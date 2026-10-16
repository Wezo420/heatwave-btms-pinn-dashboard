r"""
auto_heal.py
============

Command-line entry point for the autonomous self-healing loop.

    python auto_heal.py                       # watch the default backend
    python auto_heal.py --dry-run             # diagnose only, write nothing
    python auto_heal.py --inject nan_loss     # inject a fault, then heal it
    python auto_heal.py --command "python -m uvicorn backend.main:app --port 8001"

Exit codes
----------
``0``  the service was healthy, or was healed and is now serving.
``1``  the service could not be healed; an incident report was written.
``2``  the monitor itself failed (bad configuration, LangGraph missing).

Fault injection
---------------
``--inject`` is the demonstration entry point. It copies the target module to a
temporary backup, applies a deliberately broken one-line edit, and then hands
control to the graph. The injected faults mirror the failure taxonomy the
monitor claims to handle:

===================  =========================================================
``cuda_shape``       removes the ``reshape(-1, condition_dim)`` normaliser
``nan_loss``         removes the ``nan_to_num`` guard on the Arrhenius argument
``dtype``            changes the residual collocation batch to float32
``dataset``          points the NASA data root at a non-existent directory
``module``           introduces a bad import
``kill``             starts a script that raises on the second line
===================  =========================================================

Every injection restores the original on exit (via ``atexit``), so a
demonstration cannot leave the repository broken.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auto_heal.graph import LANGGRAPH_AVAILABLE, build_graph  # noqa: E402
from auto_heal.llm import MIN_APPLY_CONFIDENCE  # noqa: E402
from auto_heal.logger import HealLogger  # noqa: E402
from auto_heal.nodes import Runtime  # noqa: E402
from auto_heal.patching import backup_file, restore  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_COMMAND: List[str] = [
    sys.executable, "-m", "uvicorn", "backend.main:app",
    "--host", "127.0.0.1", "--port", "8000", "--log-level", "info",
]

LOG_DIR = REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------

#: name -> (target module relative to repo root, original anchor, replacement)
INJECTIONS: Dict[str, Tuple[str, str, str]] = {
    "cuda_shape": (
        "backend/models/pinn.py",
        "xyz_t = self.prepare_inputs(xyz_t, t_hours)",
        "xyz_t = self.prepare_inputs(xyz_t, t_hours)  # INJECTED: shape normaliser removed",
    ),
    "nan_loss": (
        "backend/models/physics.py",
        "t_k = torch.clamp(temp_c + 273.15, min=200.0)",
        "t_k = temp_c + 273.15  # INJECTED: non-finite guard removed",
    ),
    "dtype": (
        "backend/models/pinn.py",
        "collocation = collocation.detach().requires_grad_(True)",
        "collocation = collocation.detach().to(torch.float32).requires_grad_(True)",
    ),
    "dataset": (
        "backend/config.py",
        'NASA_DATA_ROOT: Final[Path] = Path(',
        'NASA_DATA_ROOT: Final[Path] = Path("/nonexistent/nasa/BatteryData") or Path(',
    ),
    "module": (
        "backend/config.py",
        "import math",
        "import math\nimport definitely_not_installed_xyz  # INJECTED: bad import",
    ),
}


def inject_fault(name: str, root: Path) -> List[Path]:
    """Apply an injected fault; returns the backup paths for restoration.

    Raises
    ------
    ValueError
        If ``name`` is not a known injection.
    FileNotFoundError
        If the target module or the anchor line is missing.
    """
    if name == "kill":
        return []
    if name not in INJECTIONS:
        raise ValueError(
            f"unknown injection {name!r}; choose from {sorted(INJECTIONS) + ['kill']}"
        )

    rel, anchor, replacement = INJECTIONS[name]
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"injection target not found: {path}")

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.strip() == anchor.strip()]
    if not hits:
        raise FileNotFoundError(
            f"injection anchor not found in {rel}: {anchor.strip()!r}\n"
            f"(the module may have been edited since the injection was written)"
        )

    bk = backup_file(path)
    idx = hits[0]
    indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
    lines[idx] = indent + replacement.strip()
    path.write_text("\n".join(lines) + ("\n" if source.endswith("\n") else ""),
                    encoding="utf-8")
    return [bk]


def write_kill_script(tmpdir: Path) -> List[str]:
    r"""Write a script that crashes deterministically and return its command.

    This exercises the monitor end-to-end without needing a real fault in the
    codebase: the script prints a realistic torch error and exits, which is
    exactly the shape of input ``node_classify`` is built to parse.

    The crash is deliberately **not** gated on a CUDA device.  An earlier
    version used ``device="cuda"``, so on a CPU-only box it died at the
    ``randn`` call with "Torch not compiled with CUDA enabled" rather than at
    the multiplication with the intended shape mismatch.  The monitor then had
    no rule for the error it actually got, and the demonstration of
    "recognise a known failure" turned into "fail to recognise an unknown one"
    by accident rather than by design.  A CPU tensor pair raises the same
    ``mat1 and mat2 shapes cannot be multiplied`` on every machine, which is
    what makes the injection reproducible.
    """
    script = tmpdir / "inject_crash.py"
    script.write_text(
        "import torch\n"
        "import sys\n"
        "\n"
        'print("starting injected-crash backend", flush=True)\n'
        "# 8x16 times 8x7: the inner dimensions disagree, so this is a\n"
        "# RuntimeError whether or not a CUDA device is present.\n"
        "x = torch.randn(8, 16)\n"
        "y = torch.randn(8, 7)\n"
        "z = x @ y\n"
        'print(z, flush=True)\n'
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(
        prog="auto_heal.py",
        description=(
            "Autonomous self-healing monitor for the heatwave-BTMS FastAPI backend. "
            "Watches a running server, classifies any crash, formulates a root-cause "
            "hypothesis, applies a verified source patch, and restarts the service."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python auto_heal.py\n"
            "  python auto_heal.py --dry-run\n"
            "  python auto_heal.py --inject nan_loss --max-attempts 2\n"
            "  python auto_heal.py --prefer-llm --llm-model gpt-4o-mini\n"
            "  python auto_heal.py --command \"python -m uvicorn backend.main:app --port 8001\"\n"
        ),
    )
    p.add_argument(
        "--command", nargs=argparse.REMAINDER,
        help="command to supervise (everything after this flag). Defaults to "
             "`python -m uvicorn backend.main:app --port 8000`.",
    )
    p.add_argument("--cwd", default=str(REPO_ROOT),
                   help="repository root to supervise (default: this file's directory)")
    p.add_argument("--log-dir", default=str(LOG_DIR),
                   help="directory for auto_heal.log / auto_heal.jsonl (default: ./logs)")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="healing attempts before escalating (default: 3)")
    p.add_argument("--dry-run", action="store_true",
                   help="diagnose and report, but never write to a source file")
    p.add_argument("--prefer-llm", action="store_true",
                   help="use the LLM hypothesiser instead of the deterministic rules")
    p.add_argument("--llm-model", default=None,
                   help="model name for the LLM (default: $HEAL_LLM_MODEL or gpt-4o-mini)")
    p.add_argument("--llm-base-url", default=None,
                   help="OpenAI-compatible base URL (default: $HEAL_LLM_BASE_URL)")
    p.add_argument("--llm-api-key", default=None,
                   help="API key (default: $OPENAI_API_KEY or $GROQ_API_KEY)")
    p.add_argument("--startup-timeout", type=float, default=120.0,
                   help="seconds to wait for /api/health before declaring failure")
    p.add_argument("--health-port", type=int, default=8000,
                   help="port to poll for /api/health (default: 8000)")
    p.add_argument("--inject", choices=sorted(INJECTIONS) + ["kill"],
                   help="inject a fault before starting the monitor (demonstration aid)")
    p.add_argument("--quiet", action="store_true", help="do not echo log lines to stdout")
    p.add_argument("--graph-diagram", action="store_true",
                   help="print the Mermaid graph diagram and exit")
    p.add_argument("--no-checkpointer", action="store_true",
                   help="disable the LangGraph MemorySaver (no time-travel debugging)")
    return p


def _resolve_command(args: argparse.Namespace) -> List[str]:
    """Determine the supervised command."""
    if args.command:
        cmd = list(args.command)
        # argparse.REMAINDER keeps a leading '--' if the user used one.
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            print("error: --command was given but empty", file=sys.stderr)
            raise SystemExit(2)
        return cmd
    if args.inject == "kill":
        # A crashing stub needs no port; use a temp dir so we never write into
        # the repository.
        return write_kill_script(Path(tempfile.gettempdir()))
    return list(DEFAULT_COMMAND)


def _banner(text: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    if args.graph_diagram:
        if not LANGGRAPH_AVAILABLE:
            print("langgraph is not installed; the graph diagram is unavailable.")
            return 2
        rt = Runtime(
            cwd=args.cwd, command=DEFAULT_COMMAND, log_dir=args.log_dir,
            echo=not args.quiet,
        )
        print(build_graph(rt, checkpointer=False).describe())
        return 0

    root = Path(args.cwd).resolve()
    if not (root / "backend").exists():
        print(f"error: {root} does not look like the repository root "
              f"(no backend/ directory). Pass --cwd.", file=sys.stderr)
        return 2

    logger = HealLogger.create(args.log_dir, echo=not args.quiet)
    logger.info(
        "auto-heal monitor starting",
        event="startup",
        cwd=str(root),
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        prefer_llm=args.prefer_llm,
        min_apply_confidence=MIN_APPLY_CONFIDENCE,
    )
    logger.log_environment()

    # -- fault injection -------------------------------------------------
    backups: List[Path] = []
    if args.inject:
        try:
            backups = inject_fault(args.inject, root)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: could not inject fault: {exc}", file=sys.stderr)
            return 2
        _banner(f"INJECTED FAULT: {args.inject}")
        print(f"Injected '{args.inject}' into the monitored code.", flush=True)
        print(f"Backups: {[str(b) for b in backups] or '(none needed)'}\n", flush=True)

        def _restore() -> None:
            for bk in backups:
                if restore(bk):
                    print(f"[auto-heal] restored {bk.name}", flush=True)

        atexit.register(_restore)

    command = _resolve_command(args)

    _banner("SUPERVISED COMMAND")
    print(" ".join(command), flush=True)
    print(f"repository : {root}", flush=True)
    print(f"logs       : {logger.log_path}", flush=True)
    print(f"max attempt: {args.max_attempts}"
          f"{'   (DRY RUN -- no source will be modified)' if args.dry_run else ''}\n",
          flush=True)

    # -- build the graph --------------------------------------------------
    runtime = Runtime(
        cwd=str(root),
        command=command,
        log_dir=args.log_dir,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        prefer_llm=args.prefer_llm,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        startup_timeout_s=args.startup_timeout,
        health_timeout_s=args.startup_timeout,
        health_port=args.health_port,
        echo=not args.quiet,
    )

    if not LANGGRAPH_AVAILABLE:
        logger.error(
            "langgraph is not installed; the state machine cannot run",
            event="missing_dependency",
            remedy="pip install langgraph langchain-core",
        )
        print(
            "error: langgraph is not installed.\n"
            "       pip install langgraph langchain-core\n"
            "       (The deterministic hypothesiser in auto_heal/llm.py needs no LLM,\n"
            "        but the orchestrated graph requires langgraph.)",
            file=sys.stderr,
        )
        return 2

    started = time.time()
    graph = build_graph(runtime, max_attempts=args.max_attempts,
                        checkpointer=not args.no_checkpointer)

    _banner("HEALING GRAPH")
    print(graph.describe(), flush=True)
    print("", flush=True)

    try:
        final = graph.run(
            command=command,
            cwd=str(root),
            log_path=str(logger.log_path),
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        runtime.stop_server()
        logger.warn("interrupted by user; service stopped", event="interrupted")
        print("\ninterrupted; service stopped", flush=True)
        return 1
    except Exception as exc:  # the monitor itself failed
        runtime.stop_server()
        logger.error("monitor crashed", event="monitor_crash", error=repr(exc))
        print(f"error: the monitor itself failed: {exc!r}", file=sys.stderr)
        return 2
    finally:
        runtime.stop_server()

    status = final.get("status", "unknown")
    elapsed = time.time() - started

    _banner("RESULT")
    print(f"status        : {status}", flush=True)
    print(f"attempts used : {final.get('attempt', 0)} of {args.max_attempts}", flush=True)
    print(f"elapsed       : {elapsed:.1f} s", flush=True)
    if final.get("failure_class"):
        print(f"failure class : {final['failure_class']}", flush=True)
    if final.get("hypothesis"):
        hyp = str(final["hypothesis"])
        print(f"\nhypothesis:\n  {hyp[:600]}", flush=True)
    if final.get("patch_applied"):
        print(f"\npatch applied : {final.get('patch_kind')} -> "
              f"{final.get('patch_target')}", flush=True)
        print(f"backup        : {final.get('backup_path')}", flush=True)
    if final.get("verification"):
        print(f"\nverification: {final['verification']}", flush=True)
    print(f"\nlog           : {logger.log_path}", flush=True)
    if getattr(logger, "jsonl_path", None):
        print(f"audit trail   : {logger.jsonl_path}", flush=True)

    # Restore the injected fault so the repository is left clean.
    if backups:
        print("", flush=True)
        for bk in backups:
            if restore(bk):
                print(f"restored injected fault from {bk.name}", flush=True)

    return 0 if status in ("healed", "running") else 1


if __name__ == "__main__":
    raise SystemExit(main())