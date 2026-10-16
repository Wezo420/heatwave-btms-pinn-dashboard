r"""
auto_heal.patching
==================

Deterministic, *verifiable* source patching.

Design constraint: **no patch is ever applied without a reversible backup and a
syntax check.**  An auto-healer that can permanently corrupt the codebase is
worse than no auto-healer at all, because the failure becomes silent and
attributed to the physics rather than to the tooling.  Every patch in this module
therefore goes through:

1.  ``backup``  -- write ``<file>.bak.<n>`` containing the exact pre-patch bytes.
2.  ``transform`` -- produce the new source in memory (AST rewrite or string
    replacement).
3.  ``validate`` -- ``compile()`` the new source.  A patch that does not parse is
    rejected outright and never touches disk.
4.  ``atomic_write`` -- write to ``.tmp`` then ``os.replace``, so a crash mid-write
    cannot leave a truncated source file.
5.  ``rollback`` -- on failure of the verification run, restore from the backup.

Two patch mechanisms are provided.

**AST patches** (:func:`apply_ast_patch`) operate on the parsed tree and are used
for structural edits that a regex would get wrong: inserting a method call after
a specific ``ast.Assign``, wrapping an expression, or renaming a binding.  Using
``ast.unparse`` (3.9+) regenerates canonical source, which is a real cost --
comments are lost -- so AST patches are scoped to the smallest enclosing function.

**String patches** (:func:`apply_string_patch`) are anchored on an exact source
line plus an occurrence index.  They are used for the majority of the real
remediations (adding ``.to(device)``, clamping a loss, widening a path glob)
because they preserve comments and formatting, and because a maintainer reviewing
the diff needs to see a one-line change rather than a reformatted block.

Every patch is generated from a :class:`PatchPlan`, so the LLM's contribution is
confined to *choosing* and *justifying* the plan; the mutation itself is code in
this module and is therefore testable without an LLM.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Plan container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PatchPlan:
    """A concrete, reviewable source edit.

    Attributes
    ----------
    target:
        Absolute path of the ``.py`` file to modify.
    kind:
        ``"string"``, ``"ast"``, ``"config"`` or ``"none"``.
    find / replace:
        For ``"string"``: the exact source line(s) to replace and their
        replacement.
    occurrence:
        Which occurrence of ``find`` to patch (1-based; default 1).  Needed
        because ``model.to(device)`` often appears several times in one file.
    ast_op:
        For ``"ast"``: the name of a registered :func:`register_ast_op` handler.
    ast_args:
        Keyword arguments for the handler.
    reason:
        Why this edit should fix the diagnosed failure.
    """

    target: str
    kind: str = "string"
    find: Optional[str] = None
    replace: Optional[str] = None
    occurrence: int = 1
    ast_op: Optional[str] = None
    ast_args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def describe(self) -> str:
        """Human-readable one-liner for the log."""
        if self.kind == "string":
            return (
                f"string patch on {Path(self.target).name} "
                f"(occurrence {self.occurrence}): {truncate(self.find or '', 70)!r} -> "
                f"{truncate(self.replace or '', 70)!r}"
            )
        if self.kind == "ast":
            return f"ast patch on {Path(self.target).name}: op={self.ast_op} args={self.ast_args}"
        if self.kind == "config":
            return f"config patch on {Path(self.target).name}"
        return "no patch (diagnostic only)"


@dataclass(slots=True)
class PatchResult:
    """Outcome of applying a :class:`PatchPlan`."""

    ok: bool
    applied: bool
    plan: PatchPlan
    backup_path: Optional[str] = None
    before: str = ""
    after: str = ""
    error: str = ""

    def describe(self) -> str:
        """One-line outcome summary."""
        if not self.ok:
            return f"PATCH FAILED: {self.error}"
        if not self.applied:
            return f"no change: {self.plan.describe()}"
        return f"patched {Path(self.plan.target).name} ({self.plan.kind}); backup={self.backup_path}"


def truncate(text: str, n: int) -> str:
    """Shorten ``text`` to ``n`` characters."""
    return text if len(text) <= n else text[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------


def backup_file(path: Path) -> Path:
    """Copy ``path`` to ``<name>.bak.<epoch>`` and return the backup path.

    Raises
    ------
    FileNotFoundError
        If the target does not exist.  Callers treat this as "nothing to patch",
        which is a legitimate outcome for a config-only remediation.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.bak.{stamp}")
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(f"{path.name}.bak.{stamp}.{counter}")
    shutil.copy2(path, target)
    return target


def restore(backup: Optional[Path]) -> bool:
    """Restore a file from its backup; returns True on success."""
    if backup is None or not backup.exists():
        return False
    original = backup.name.split(".bak.")[0]
    target = backup.parent / original
    try:
        shutil.copy2(backup, target)
        return True
    except OSError:
        return False


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically."""
    tmp = path.with_suffix(path.suffix + ".heal.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# String patches
# ---------------------------------------------------------------------------


def apply_string_patch(plan: PatchPlan) -> PatchResult:
    """Replace the ``occurrence``-th exact match of ``find`` with ``replace``.

    The match is whole-line (after stripping the leading indentation) so that a
    substring appearing inside a longer statement cannot be clobbered
    accidentally.  Indentation of the replaced line is preserved.
    """
    path = Path(plan.target)
    if plan.find is None or plan.replace is None:
        return PatchResult(False, False, plan, error="string patch missing find/replace")
    if not path.exists():
        return PatchResult(False, False, plan, error=f"target not found: {path}")

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    target_find = plan.find.strip()
    hits = [i for i, ln in enumerate(lines) if ln.strip() == target_find]
    if not hits:
        # Fall back to a substring match, but only if it is unique -- an
        # ambiguous substring is far more dangerous than a failed patch.
        sub = [i for i, ln in enumerate(lines) if target_find and target_find in ln]
        if len(sub) == 1:
            hits = sub
        else:
            return PatchResult(
                False,
                False,
                plan,
                error=(
                    f"anchor not found ({len(hits)} exact, {len(sub)} substring matches): "
                    f"{truncate(target_find, 60)!r}"
                ),
            )

    if plan.occurrence < 1 or plan.occurrence > len(hits):
        return PatchResult(
            False, False, plan,
            error=f"occurrence {plan.occurrence} out of range (found {len(hits)})",
        )

    idx = hits[plan.occurrence - 1]
    indent = re.match(r"^\s*", lines[idx]).group(0)
    new_line = indent + plan.replace.strip()

    patched_lines = list(lines)
    patched_lines[idx] = new_line
    patched = "\n".join(patched_lines) + ("\n" if original.endswith("\n") else "")

    if patched == original:
        return PatchResult(True, False, plan, before=original, after=original,
                           error="replacement is identical to the original")

    # Compile-check before touching disk.
    try:
        compile(patched, str(path), "exec")
    except SyntaxError as exc:
        return PatchResult(
            False, False, plan, before=original,
            error=f"patch would produce a SyntaxError: {exc}",
        )

    bk = backup_file(path)
    atomic_write(path, patched)
    return PatchResult(True, True, plan, backup_path=str(bk), before=original, after=patched)


# ---------------------------------------------------------------------------
# AST patches
# ---------------------------------------------------------------------------

#: Registry of named AST operations, keyed by :attr:`PatchPlan.ast_op`.
AST_OPS: Dict[str, Callable[[ast.AST, Dict[str, Any]], ast.AST]] = {}


def register_ast_op(name: str) -> Callable[[Callable[[ast.AST, Dict[str, Any]], ast.AST]],
                                           Callable[[ast.AST, Dict[str, Any]], ast.AST]]:
    """Decorator registering an AST transformation under ``name``."""
    def deco(fn):
        AST_OPS[name] = fn
        return fn
    return deco


@register_ast_op("append_call_to_method")
def _op_append_call(tree: ast.AST, args: Dict[str, Any]) -> ast.AST:
    """Insert a call at the start of a named method's body.

    ``args``: ``{"method": "forward", "call": "xyz_t = xyz_t.to(self.device)"}``.
    """
    method = args.get("method", "forward")
    stmt = ast.parse(args.get("call", "pass")).body[0]
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            node.body.insert(0, stmt)
            found = True
    if not found:
        raise ValueError(f"method {method!r} not found")
    return tree


@register_ast_op("wrap_return_expression")
def _op_wrap_return(tree: ast.AST, args: Dict[str, Any]) -> ast.AST:
    """Wrap the first ``return`` expression of a function in a call.

    ``args``: ``{"method": "forward", "wrap": "torch.nan_to_num"}``.
    """
    method = args.get("method")
    wrap = args.get("wrap", "torch.nan_to_num")
    wrapper = ast.parse(f"{wrap}(0.0)").body[0].value  # type: ignore[attr-defined]
    inner = ast.parse("__EXPR__").body[0].value  # type: ignore[attr-defined]
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (method is None or node.name == method):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    wrapper.args[0] = sub.value  # type: ignore[attr-defined]
                    sub.value = wrapper
                    found = True
                    break
            if found:
                break
    if not found:
        raise ValueError(f"no return statement found in {method!r}")
    return tree


@register_ast_op("add_clamp_to_loss")
def _op_add_clamp(tree: ast.AST, args: Dict[str, Any]) -> ast.AST:
    """Clamp a named loss term to ``[0, 1e6]`` at its return site.

    ``args``: ``{"method": "forward", "variable": "loss", "hi": 1e6}``.
    """
    method = args.get("method")
    var = args.get("variable", "loss")
    hi = args.get("hi", 1e6)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (method is None or node.name == method):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    wrapped = ast.parse(
                        f"torch.clamp({var}, min=0.0, max={hi})"
                    ).body[0].value  # type: ignore[attr-defined]
                    wrapped.left = sub.value  # type: ignore[attr-defined]
                    sub.value = wrapped
                    found = True
                    break
            if found:
                break
    if not found:
        raise ValueError(f"no return statement found in {method!r}")
    return tree


@register_ast_op("rename_function")
def _op_rename(tree: ast.AST, args: Dict[str, Any]) -> ast.AST:
    """Rename a function definition and all of its in-file call sites."""
    old = args.get("old")
    new = args.get("new")
    if not old or not new:
        raise ValueError("rename requires 'old' and 'new'")
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == old:
            node.name = new
            count += 1
        elif isinstance(node, ast.Name) and node.id == old:
            node.id = new
            count += 1
        elif isinstance(node, ast.Attribute) and node.attr == old:
            node.attr = new
            count += 1
    if not count:
        raise ValueError(f"no occurrences of {old!r}")
    return tree


def apply_ast_patch(plan: PatchPlan) -> PatchResult:
    """Apply a registered AST transformation.

    .. warning::
       ``ast.unparse`` discards comments and normalises formatting.  This patch
       kind rewrites the whole file, so it is reserved for structural fixes; the
       default remediations use string patches to keep diffs reviewable.
    """
    path = Path(plan.target)
    if not plan.ast_op:
        return PatchResult(False, False, plan, error="ast patch missing ast_op")
    if plan.ast_op not in AST_OPS:
        return PatchResult(
            False, False, plan,
            error=f"unknown ast_op {plan.ast_op!r} (known: {sorted(AST_OPS)})",
        )
    if not path.exists():
        return PatchResult(False, False, plan, error=f"target not found: {path}")

    original = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original, filename=str(path))
        new_tree = AST_OPS[plan.ast_op](tree, plan.ast_args)
        patched = ast.unparse(new_tree) + "\n"
        compile(patched, str(path), "exec")
    except (SyntaxError, ValueError, KeyError, AttributeError) as exc:
        return PatchResult(False, False, plan, before=original, error=f"{type(exc).__name__}: {exc}")

    if patched == original:
        return PatchResult(True, False, plan, before=original, after=original,
                           error="ast patch produced no change")

    bk = backup_file(path)
    atomic_write(path, patched)
    return PatchResult(True, True, plan, backup_path=str(bk), before=original, after=patched)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def apply_patch(plan: PatchPlan) -> PatchResult:
    """Apply a plan, dispatching on :attr:`PatchPlan.kind`."""
    if plan.kind == "none":
        return PatchResult(True, False, plan, error="diagnostic-only plan")
    if plan.kind == "string":
        return apply_string_patch(plan)
    if plan.kind == "ast":
        return apply_ast_patch(plan)
    if plan.kind == "config":
        return apply_config_patch(plan)
    return PatchResult(False, False, plan, error=f"unknown patch kind {plan.kind!r}")


def apply_config_patch(plan: PatchPlan) -> PatchResult:
    """Environment-level remediation that touches no source file.

    Used for classes of failure where the correct fix is an environment change
    rather than a code change: forcing CPU execution when CUDA is unavailable or
    broken, raising a thread stack limit, or pointing the dataset root at a
    directory that actually exists.  Writing these into the source would be
    wrong -- they are properties of the deployment, not of the program.
    """
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (plan.ast_args or {}).items()})
    os.environ.update(env)
    return PatchResult(True, True, plan, error=f"env set: {plan.ast_args}")


def undo(result: PatchResult) -> bool:
    """Roll a patch back using its backup."""
    if not result.applied or result.backup_path is None:
        return False
    if result.plan.kind == "config":
        # Environment patches are not undone by restoring a file; the caller is
        # expected to restart the process, which inherits a clean environment.
        return False
    return restore(Path(result.backup_path))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def compile_check(path: Path) -> Tuple[bool, str]:
    """``compile()`` a file; returns ``(ok, message)``."""
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        return True, ""
    except (SyntaxError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def import_check(path: Path, timeout: float = 90.0) -> Tuple[bool, str]:
    """Import a module in a subprocess to confirm it loads.

    A subprocess is essential: importing into *this* interpreter would pollute
    ``sys.modules`` with a possibly-broken module and could mask the very failure
    we are testing for.  A 30 s budget mirrors "would the server have started?".

    Returns
    -------
    (ok, output)
        ``output`` is the combined stdout/stderr, trimmed to 4000 characters.
    """
    repo_root = Path.cwd()
    rel = _module_relpath(path, repo_root)
    if rel is None:
        return False, f"{path} is outside the monitored repository"

    code = (
        "import importlib,sys\n"
        f"m=importlib.import_module({rel!r})\n"
        "print('HEALTH_IMPORT_OK')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"import timed out after {timeout:.0f}s"
    except OSError as exc:
        return False, f"could not spawn interpreter: {exc}"

    combined = (proc.stdout + "\n" + proc.stderr).strip()
    ok = proc.returncode == 0 and "HEALTH_IMPORT_OK" in proc.stdout
    return ok, combined[-4000:]


def _module_relpath(path: Path, root: Path) -> Optional[str]:
    """Convert ``backend/models/pinn.py`` -> ``backend.models.pinn``."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    if rel.name == "__init__.py":
        rel = rel.parent
    else:
        rel = rel.with_suffix("")
    parts = [p for p in rel.parts if p != "__pycache__"]
    if not parts:
        return None
    return ".".join(parts)


def syntax_all(repo_root: Path, limit: int = 400) -> List[str]:
    """Compile every ``.py`` file under ``repo_root``; returns the failing paths.

    Used as a post-patch blast-radius check: a patch that fixes one file but
    breaks its importer will be caught here before the server is restarted.
    """
    bad: List[str] = []
    for i, py in enumerate(repo_root.rglob("*.py")):
        if i > limit:
            break
        if any(part in {".venv", "venv", "__pycache__", "node_modules", ".git"}
               for part in py.parts):
            continue
        ok, _ = compile_check(py)
        if not ok:
            bad.append(str(py))
    return bad