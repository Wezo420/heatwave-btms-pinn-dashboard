r"""
auto_heal.nodes
===============

The individual LangGraph nodes.

Each node is a pure-ish function ``(state) -> partial_state`` that reads from and
returns a partial :class:`~auto_heal.graph_state.HealState`.  Keeping them small
and side-effect-explicit makes the graph legible: the control flow in
:mod:`auto_heal.graph` reads like a sentence, and each node can be unit-tested by
handing it a synthetic state.

Graph shape
-----------

.. code-block:: text

    start
      |
      v
    monitor            run the server, wait for a crash
      |
      v
    classify           parse the traceback, assign a FailureClass
      |                  \
      | no crash           \ UNKNOWN -> diagnose_llm
      |                           /
      v                          /
    restart  <-----------------
      |
      v (attempts exhausted)
    escalate

    restart --> verify --> (healthy) report
                          (still failing) --> rollback --> classify (next attempt)

The two "escape hatches" that matter for safety are:

* :func:`node_restart` runs *before* any patch is considered, because a large
  fraction of "crashes" are transient (a reload race, a port taken by a previous
  instance). Patching a file because a server started twice is how an auto-healer
  breaks working code.
* :func:`node_rollback` restores the backup before the next attempt, so attempt
  N+1 always starts from the same baseline as attempt 1. Without this, a failed
  patch compounds with the next one.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph_state import FailureClass, HealState, TracebackInfo
from .llm import Hypothesis, MIN_APPLY_CONFIDENCE, build_hypothesiser
from .logger import HealLogger
from .patching import (
    PatchPlan,
    PatchResult,
    apply_patch,
    compile_check,
    import_check,
    syntax_all,
)
from .traceback_parser import classify, parse_traceback, summarise

# ---------------------------------------------------------------------------
# Module-level runtime context
# ---------------------------------------------------------------------------


class Runtime:
    """Mutable context shared by the nodes (the graph state is pure data).

    LangGraph checkpoints state, not objects; holding the logger, the
    hypothesiser and the server handle here means the nodes stay simple and the
    state remains JSON-serialisable, which is what makes ``graph.get_state()``
    and time-travel debugging work.
    """

    def __init__(
        self,
        cwd: str,
        command: List[str],
        log_dir: str,
        max_attempts: int = 3,
        dry_run: bool = False,
        prefer_llm: bool = False,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        startup_timeout_s: float = 120.0,
        health_timeout_s: float = 180.0,
        health_port: int = 8000,
        echo: bool = True,
    ) -> None:
        self.cwd = str(Path(cwd).resolve())
        self.command = list(command)
        self.max_attempts = max(1, int(max_attempts))
        self.dry_run = bool(dry_run)
        self.startup_timeout_s = startup_timeout_s
        self.health_timeout_s = health_timeout_s
        self.health_port = health_port
        self.logger = HealLogger.create(log_dir, echo=echo)
        self.hypothesiser = build_hypothesiser(
            cwd=self.cwd,
            prefer_llm=prefer_llm,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
        self.process: Optional[subprocess.Popen] = None
        self.last_patch: Optional[PatchResult] = None
        self.captured: Optional[TracebackInfo] = None

    # -- server lifecycle --------------------------------------------------
    def start_server(self) -> subprocess.Popen:
        """Launch the monitored command as a child process."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # New process group so we can kill the whole tree (uvicorn spawns
            # reload workers, which would otherwise survive a kill of the parent).
            start_new_session=(os.name != "nt"),
        )
        return self.process

    def stop_server(self, sig: int = signal.SIGTERM) -> None:
        """Terminate the child, escalating to SIGKILL after a grace period."""
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.process = None

    def drain(self, limit: int = 4000) -> str:
        """Read whatever the child has written so far, without blocking."""
        proc = self.process
        if proc is None or proc.stdout is None:
            return ""
        try:
            import fcntl  # POSIX only

            fd = proc.stdout.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            chunk = proc.stdout.read() or ""
            return chunk[-limit:]
        except ImportError:
            # Windows: use a non-blocking read via a reader thread.
            return self._drain_windows(limit)

    def _drain_windows(self, limit: int) -> str:
        """Read the child's output on Windows without blocking."""
        proc = self.process
        if proc is None or proc.stdout is None:
            return ""
        import io
        import threading

        buffer: List[str] = []

        def reader() -> None:
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    buffer.append(line)
            except (ValueError, OSError):
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=0.5)
        return "".join(buffer)[-limit:]

    # -- health ------------------------------------------------------------
    def wait_healthy(self, timeout_s: float) -> bool:
        """Poll ``/api/health`` until it returns 200 or ``timeout_s`` elapses."""
        import json as _json
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self.health_port}/api/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            proc = self.process
            if proc is not None and proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        body = _json.loads(resp.read().decode("utf-8", "replace"))
                        self.logger.event("health_ok", {"status": body.get("status")})
                        return True
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(1.0)
        return False


#: Process-wide runtime for the graph; set by :func:`auto_heal.graph.build_graph`.
RUNTIME: Optional[Runtime] = None


def set_runtime(rt: Runtime) -> None:
    """Install the runtime used by the graph's nodes."""
    global RUNTIME
    RUNTIME = rt


def _rt() -> Runtime:
    if RUNTIME is None:
        raise RuntimeError("Runtime not initialised; call auto_heal.nodes.set_runtime() first")
    return RUNTIME


# ---------------------------------------------------------------------------
# Node 1: monitor
# ---------------------------------------------------------------------------


def node_monitor(state: HealState) -> Dict[str, Any]:
    """Start the server and wait for either a crash or a healthy response.

    Returns ``failed=False`` when the service came up, in which case the graph
    terminates successfully -- the system is healthy and there is nothing to heal.
    """
    rt = _rt()
    rt.last_patch = None

    if rt.start_server() is None:  # pragma: no cover - Popen raises instead
        return {"status": "aborted"}

    healthy = rt.wait_healthy(rt.startup_timeout_s)

    if healthy:
        rt.logger.info(
            "service is healthy; no healing required",
            event="healthy",
            attempt=state.get("attempt", 0) + 1,
            command=" ".join(rt.command),
        )
        rt.logger.event("healthy", {"attempt": state.get("attempt", 0) + 1})
        return {
            "failed": False,
            "status": "running",
            "classified": True,
            "verified": True,
            "verification": "service reached a healthy state without intervention",
            "finished_at": time.time(),
        }

    # Drain whatever the child wrote and look for a traceback.
    output = rt.drain()
    proc = rt.process
    exit_code = proc.poll() if proc is not None else None
    rt.logger.warn(
        "service did not become healthy",
        event="failure_detected",
        attempt=state.get("attempt", 0) + 1,
        exit_code=exit_code,
        output_tail=_tail(output),
    )

    # If the process is still alive but not serving, treat that as a hang: stop it
    # so the next attempt starts from a clean state.
    if proc is not None and exit_code is None:
        rt.stop_server()

    return {
        "failed": True,
        "server_stdout": output,
        "server_stderr": "",
        "server_pid": proc.pid if proc is not None else None,
        "server_exit_code": exit_code,
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Node 2: classify
# ---------------------------------------------------------------------------


def node_classify(state: HealState) -> Dict[str, Any]:
    """Parse the captured output into a traceback and assign a failure class."""
    rt = _rt()
    raw = state.get("server_stdout", "") or ""
    attempt = int(state.get("attempt", 0)) + 1

    info, cls = _extract(raw)
    if info is None:
        # No traceback in the output at all: the server died without a Python
        # error (killed by the OS, segfault, OOM killer). There is nothing to
        # patch; report and stop.
        rt.logger.error(
            "no Python traceback in the captured output; nothing patchable",
            event="unclassified_crash",
            attempt=attempt,
            exit_code=state.get("server_exit_code"),
            output_tail=_tail(raw),
        )
        rt.logger.event("unclassified_crash",
                        {"attempt": attempt, "output": raw[-4000:]})
        return {
            "classified": True,
            "failure_class": FailureClass.UNKNOWN.value,
            "traceback": None,
            "status": "exhausted",
            "finished_at": time.time(),
        }

    rt.captured = info
    line = summarise(info, cls)
    # NOTE: the field is spelled ``message_text``, ``message`` is taken.
    # Passing ``message=`` here raises TypeError -- the first positional parameter
    # of :meth:`HealLogger.error` is itself named ``message``, so the kwarg
    # arrives twice and crashes the monitor while it is reporting the crash it
    # just classified. The logger call is the very last thing this node does,
    # so the failure surfaces as "the monitor itself failed" with no hint that
    # the underlying service was diagnosable all along.
    rt.logger.error(
        line,
        event="classified",
        attempt=attempt,
        failure_class=cls.value,
        location=f"{info.failing_file}:{info.failing_line}",
        exception_type=info.exception_type,
        message_text=info.message[:400],
    )
    rt.logger.event("classified", {
        "attempt": attempt,
        "failure_class": cls.value,
        "traceback": info.to_dict(),
        "summary": line,
    })

    return {
        "classified": True,
        "attempt": attempt,
        "failure_class": cls.value,
        "traceback": info.to_dict(),
        "raw_traceback": info.raw,
    }


# ---------------------------------------------------------------------------
# Node 3: hypothesise (rule base, then LLM for UNKNOWN)
# ---------------------------------------------------------------------------


def node_hypothesise(state: HealState) -> Dict[str, Any]:
    """Form the root-cause hypothesis and the concrete remediation plan."""
    rt = _rt()
    info = _traceback_from_state(state)
    if info is None:
        return {"hypothesis": "no traceback available", "confidence": 0.0,
                "patch_kind": "none", "llm_used": False}

    cls = FailureClass(state.get("failure_class", FailureClass.UNKNOWN.value))
    hyp = rt.hypothesiser.hypothesise(info, cls)

    rt.logger.info(
        hyp.describe(),
        event="hypothesis",
        attempt=state.get("attempt", 0),
        failure_class=cls.value,
        confidence=round(hyp.confidence, 3),
        source=hyp.source,
        patch_kind=hyp.plan.kind,
        patch_target=_short(hyp.plan.target),
    )
    rt.logger.event("hypothesis", {
        "attempt": state.get("attempt", 0),
        "failure_class": cls.value,
        "hypothesis": hyp.explanation,
        "confidence": hyp.confidence,
        "source": hyp.source,
        "alternatives": hyp.alternatives,
        "plan": {
            "kind": hyp.plan.kind,
            "target": _short(hyp.plan.target),
            "find": hyp.plan.find,
            "replace": hyp.plan.replace,
            "occurrence": hyp.plan.occurrence,
            "ast_op": hyp.plan.ast_op,
            "reason": hyp.plan.reason,
        },
    })

    return {
        "hypothesis": hyp.explanation,
        "rationale": hyp.plan.reason,
        "confidence": hyp.confidence,
        "llm_used": hyp.source.startswith("llm"),
        "patch_target": hyp.plan.target,
        "patch_kind": hyp.plan.kind,
        "patch_find": hyp.plan.find,
        "patch_replace": hyp.plan.replace,
        # The occurrence index must be carried across the hop. Leaving it out
        # makes node_patch fall back to 1 and patch the first match even when
        # the hypothesiser named the third.
        "patch_occurrence": int(hyp.plan.occurrence or 1),
    }


def node_diagnose_llm(state: HealState) -> Dict[str, Any]:
    """Second-chance LLM diagnosis for failures the rule base cannot classify.

    Kept as a separate node rather than a branch inside
    :func:`node_hypothesise` so the escalation decision is visible in the graph
    topology and so a future rule can override it.
    """
    rt = _rt()
    info = _traceback_from_state(state)
    if info is None:
        return {"hypothesis": "no traceback available", "confidence": 0.0,
                "patch_kind": "none", "llm_used": False}

    from .llm import LLMHypothesiser

    hypothesiser = LLMHypothesiser(cwd=rt.cwd)
    hyp = hypothesiser.hypothesise(info, FailureClass.UNKNOWN)

    rt.logger.info(
        hyp.describe(),
        event="llm_diagnosis",
        attempt=state.get("attempt", 0),
        confidence=round(hyp.confidence, 3),
        source=hyp.source,
        patch_kind=hyp.plan.kind,
    )
    rt.logger.event("llm_diagnosis", {
        "attempt": state.get("attempt", 0),
        "hypothesis": hyp.explanation,
        "confidence": hyp.confidence,
        "source": hyp.source,
        "plan_kind": hyp.plan.kind,
        "plan_target": _short(hyp.plan.target),
    })

    return {
        "hypothesis": hyp.explanation,
        "rationale": hyp.plan.reason,
        "confidence": hyp.confidence,
        "llm_used": True,
        "patch_target": hyp.plan.target,
        "patch_kind": hyp.plan.kind,
        "patch_find": hyp.plan.find,
        "patch_replace": hyp.plan.replace,
        # Same reason as node_hypothesise: node_patch reads this back, and an
        # undeclared or unset key defaults to occurrence 1.
        "patch_occurrence": int(hyp.plan.occurrence or 1),
    }


# ---------------------------------------------------------------------------
# Node 4: patch
# ---------------------------------------------------------------------------


def node_patch(state: HealState) -> Dict[str, Any]:
    """Apply the patch, after the confidence and dry-run gates."""
    rt = _rt()
    attempt = int(state.get("attempt", 0))
    kind = state.get("patch_kind", "none")
    confidence = float(state.get("confidence", 0.0))

    if kind == "none":
        rt.logger.warn(
            "hypothesis proposes no source change; escalating",
            event="no_action",
            attempt=attempt,
            failure_class=state.get("failure_class"),
            confidence=round(confidence, 3),
        )
        return {"patch_applied": False, "status": "exhausted", "finished_at": time.time()}

    if confidence < MIN_APPLY_CONFIDENCE:
        rt.logger.warn(
            "hypothesis confidence below the apply threshold; not patching",
            event="low_confidence",
            attempt=attempt,
            confidence=round(confidence, 3),
            threshold=MIN_APPLY_CONFIDENCE,
        )
        rt.logger.event("low_confidence", {"attempt": attempt, "confidence": confidence})
        return {"patch_applied": False, "status": "exhausted", "finished_at": time.time()}

    if rt.dry_run:
        plan = PatchPlan(
            target=state.get("patch_target", ""), kind=kind,
            find=state.get("patch_find"), replace=state.get("patch_replace"),
            occurrence=int(state.get("patch_occurrence", 1) or 1),
            reason=state.get("rationale", ""),
        )
        rt.logger.info(
            "DRY RUN: would apply " + plan.describe(),
            event="dry_run",
            attempt=attempt,
            patch_target=_short(plan.target),
        )
        rt.logger.event("dry_run", {"attempt": attempt, "plan": plan.describe()})
        return {"patch_applied": False, "verification": "dry run: no change written",
                "status": "exhausted", "finished_at": time.time()}

    plan = PatchPlan(
        target=state.get("patch_target", ""),
        kind=kind,
        find=state.get("patch_find"),
        replace=state.get("patch_replace"),
        occurrence=int(state.get("patch_occurrence", 1) or 1),
        reason=state.get("rationale", ""),
    )
    result = apply_patch(plan)
    rt.last_patch = result

    rt.logger.info(
        result.describe(),
        event="patch_applied" if result.applied else "patch_skipped",
        attempt=attempt,
        patch_kind=kind,
        patch_target=_short(plan.target),
        error=result.error or None,
    )
    rt.logger.event("patch_result", {
        "attempt": attempt,
        "applied": result.applied,
        "ok": result.ok,
        "backup": result.backup_path,
        "error": result.error,
        "before_sha": _sha(result.before),
        "after_sha": _sha(result.after),
    })

    if not result.ok or not result.applied:
        return {
            "patch_applied": False,
            "verification": f"patch not applied: {result.error}",
            "status": "exhausted",
            "finished_at": time.time(),
        }

    return {
        "patch_applied": True,
        "backup_path": result.backup_path,
        "patch_before": result.before,
        "patch_after": result.after,
    }


# ---------------------------------------------------------------------------
# Node 5: verify (import check + blast-radius compile sweep)
# ---------------------------------------------------------------------------


def node_verify(state: HealState) -> Dict[str, Any]:
    """Statically validate the patch before spending a restart on it.

    Two checks, cheapest first:

    1.  ``import_check`` -- spawn a *subprocess* that imports the patched module.
        A subprocess is mandatory: importing into this interpreter would leave a
        possibly-broken module in ``sys.modules`` and could mask the very
        failure being tested.
    2.  ``syntax_all`` -- compile every ``.py`` in the repository.  A patch can
        parse fine and still break its importer (e.g. renaming a helper that
        three modules import), and catching that before the restart saves a
        full crash-restart cycle.
    """
    rt = _rt()
    attempt = int(state.get("attempt", 0))
    target = state.get("patch_target", "")

    if not state.get("patch_applied"):
        return {"verified": False,
                "verification": "no patch was applied; nothing to verify"}

    ok_import, out_import = import_check(Path(target))
    if not ok_import:
        rt.logger.error(
            "patched module failed its import check; rolling back",
            event="verify_failed",
            attempt=attempt,
            stage="import",
            target=_short(target),
            output=_tail(out_import),
        )
        rt.logger.event("verify_failed", {"attempt": attempt, "stage": "import",
                                          "output": out_import[-4000:]})
        return {
            "verified": False,
            "verification": f"import check failed: {_tail(out_import, 8)}",
            "verification_output": out_import,
        }

    broken = syntax_all(Path(rt.cwd))
    if broken:
        rt.logger.error(
            "post-patch compile sweep found broken files; rolling back",
            event="verify_failed",
            attempt=attempt,
            stage="compile_sweep",
            broken=broken,
        )
        rt.logger.event("verify_failed", {"attempt": attempt, "stage": "compile_sweep",
                                          "broken": broken})
        return {
            "verified": False,
            "verification": f"compile sweep found {len(broken)} broken file(s): {broken[:3]}",
            "verification_output": "\n".join(broken),
        }

    rt.logger.info(
        "patch verified statically (import + repository compile sweep)",
        event="verify_ok",
        attempt=attempt,
        target=_short(target),
    )
    rt.logger.event("verify_ok", {"attempt": attempt, "target": _short(target)})
    return {
        "verified": True,
        "verification": "import check passed and the full repository compiles",
        "verification_output": out_import[-2000:],
    }


# ---------------------------------------------------------------------------
# Node 6: rollback
# ---------------------------------------------------------------------------


def node_rollback(state: HealState) -> Dict[str, Any]:
    """Restore the pre-patch source so the next attempt starts clean."""
    rt = _rt()
    attempt = int(state.get("attempt", 0))
    backup = state.get("backup_path")

    if not backup:
        rt.logger.warn("no backup to roll back to", event="rollback", attempt=attempt)
        return {"rolled_back": False}

    from .patching import restore

    ok = restore(Path(backup))
    rt.logger.info(
        "rolled the patch back" if ok else "ROLLBACK FAILED -- manual intervention required",
        event="rollback",
        attempt=attempt,
        backup=_short(backup),
        success=ok,
    )
    rt.logger.event("rollback", {"attempt": attempt, "backup": _short(backup), "success": ok})
    return {"rolled_back": ok}


# ---------------------------------------------------------------------------
# Node 7: restart
# ---------------------------------------------------------------------------

#: Markers that indicate the server has actually begun serving.
_READY_MARKERS = (
    "Application startup complete",
    "Uvicorn running on",
    "started server process",
    "Running on http",
)


def node_restart(state: HealState) -> Dict[str, Any]:
    """Stop any surviving process and relaunch the monitored command."""
    rt = _rt()
    attempt = int(state.get("attempt", 0))
    rt.stop_server()

    if rt.start_server() is None:  # pragma: no cover
        return {"status": "aborted"}

    healthy = rt.wait_healthy(rt.startup_timeout_s)
    output = rt.drain()
    exit_code = rt.process.poll() if rt.process is not None else None

    if healthy:
        rt.logger.info(
            "service recovered after restart",
            event="restarted",
            attempt=attempt,
            patch_applied=state.get("patch_applied", False),
        )
        rt.logger.event("restarted", {"attempt": attempt, "healthy": True})
        return {
            "failed": False,
            "status": "healed",
            "classified": True,
            "verified": True,
            "verification": "service reached a healthy state after the patch and restart",
            "finished_at": time.time(),
        }

    rt.logger.warn(
        "service still unhealthy after restart",
        event="restart_failed",
        attempt=attempt,
        exit_code=exit_code,
        ready_markers=[m for m in _READY_MARKERS if m in output],
        output_tail=_tail(output),
    )
    rt.logger.event("restart_failed", {"attempt": attempt, "exit_code": exit_code,
                                       "output": output[-4000:]})
    rt.stop_server()

    return {
        "failed": True,
        "server_stdout": output,
        "server_exit_code": exit_code,
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Node 8: report / escalate
# ---------------------------------------------------------------------------


def node_report(state: HealState) -> Dict[str, Any]:
    """Terminal success node: summarise the healing action."""
    rt = _rt()
    attempt = int(state.get("attempt", 0))
    plan_desc = ""
    if state.get("patch_applied"):
        plan_desc = (
            f"{state.get('patch_kind')} patch on "
            f"{_short(state.get('patch_target', ''))}"
        )
    rt.logger.info(
        "HEALED: " + (plan_desc or "service restarted without a source change"),
        event="healed",
        attempts=attempt,
        failure_class=state.get("failure_class"),
        verification=state.get("verification"),
    )
    rt.logger.event("healed", {
        "attempts": attempt,
        "failure_class": state.get("failure_class"),
        "hypothesis": state.get("hypothesis"),
        "patch_kind": state.get("patch_kind"),
        "patch_target": _short(state.get("patch_target", "")),
        "backup": state.get("backup_path"),
        "verification": state.get("verification"),
        "server_pid": state.get("server_pid"),
        "elapsed_s": _elapsed(state),
    })
    _write_report(rt, state, healed=True)
    return {"status": "healed", "finished_at": time.time()}


def node_escalate(state: HealState) -> Dict[str, Any]:
    """Terminal failure node: report that the agent could not fix it.

    Leaving the service *stopped* is the right default.  A self-healer that
    leaves a broken server running in front of a safety-adjacent dashboard is
    worse than one that stops and says so; the operator decides what to do next.
    """
    rt = _rt()
    rt.stop_server()
    attempt = int(state.get("attempt", 0))
    rt.logger.error(
        "EXHAUSTED: could not heal automatically; intervention required",
        event="exhausted",
        attempts=attempt,
        failure_class=state.get("failure_class"),
        hypothesis=(state.get("hypothesis") or "")[:400],
        verification=state.get("verification"),
    )
    rt.logger.event("exhausted", {
        "attempts": attempt,
        "failure_class": state.get("failure_class"),
        "hypothesis": state.get("hypothesis"),
        "confidence": state.get("confidence"),
        "patch_kind": state.get("patch_kind"),
        "verification": state.get("verification"),
        # The crashed process's PID, so an operator reading the JSONL audit
        # trail after the fact can correlate with any OS-level log.
        "server_pid": state.get("server_pid"),
        "server_exit_code": state.get("server_exit_code"),
        "elapsed_s": _elapsed(state),
    })
    _write_report(rt, state, healed=False)
    return {"status": "exhausted", "finished_at": time.time()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract(output: str) -> tuple[Optional[TracebackInfo], FailureClass]:
    """Extract the *last* traceback from a captured output stream.

    A crashing server can print several tracebacks (the original, then one from
    the reloader). The last one is the most recent and therefore the most
    relevant.
    """
    if not output:
        return None, FailureClass.UNKNOWN
    marker = "Traceback (most recent call last)"
    if marker not in output:
        return None, FailureClass.UNKNOWN
    chunks = output.split(marker)
    raw = marker + chunks[-1]
    info = parse_traceback(raw)
    return info, classify(info)


def _traceback_from_state(state: HealState) -> Optional[TracebackInfo]:
    """Rehydrate a :class:`TracebackInfo` from the serialised state."""
    tb = state.get("traceback")
    if not tb:
        return None
    return TracebackInfo(
        raw=state.get("raw_traceback", ""),          # type: ignore[arg-type]
        exception_type=tb.get("exception_type", "Exception"),
        message=tb.get("message", ""),
        failing_file=tb.get("failing_file"),
        failing_line=int(tb.get("failing_line", 0)),
        failing_source=tb.get("failing_source", ""),
        frames=list(tb.get("frames", [])),
    )


def _tail(text: str, n: int = 24) -> str:
    """Last ``n`` lines of ``text``."""
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:])


def _short(path: str, cwd: Optional[str] = None) -> str:
    r"""Repository-relative path for logging; falls back to the basename.

    The obvious implementation -- strip the ``C:\`` drive prefix -- produces a
    path that is neither absolute nor repo-relative
    (``Users/you/proj/backend/config.py``), which is useless to a reader trying
    to locate the file and unusable for copy-paste. What the log actually wants
    is ``backend/config.py``.

    ``patches_under_repo`` was already the behaviour every caller assumed, so
    this takes the repo root when it can and degrades to the basename rather
    than to a mangled absolute path.  ``cwd`` defaults to the live runtime's
    root, which is what every production call site wants.
    """
    p = str(path or "")
    if not p:
        return ""
    root = cwd if cwd is not None else (RUNTIME.cwd if RUNTIME is not None else None)
    if root:
        try:
            return Path(p).resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    # Not under the repo root (a backup, a temp file): keep the name.
    return Path(p).name


def _sha(text: str) -> str:
    """Short content hash, for detecting whether a patch changed anything."""
    import hashlib

    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _elapsed(state: HealState) -> float:
    """Seconds between the first and last event."""
    started = float(state.get("started_at", time.time()))
    finished = float(state.get("finished_at", time.time()) or time.time())
    return max(0.0, finished - started)


def _write_report(rt: Runtime, state: HealState, healed: bool) -> Path:
    """Write a standalone Markdown incident report next to the log."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(rt.log_dir if hasattr(rt, "log_dir") else rt.logger.log_path.parent) / f"incident-{stamp}.md"
    lines = [
        f"# Auto-heal incident report -- {'HEALED' if healed else 'EXHAUSTED'}",
        "",
        f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Attempts: {state.get('attempt', 0)} of {rt.max_attempts}",
        f"- Monitored command: `{' '.join(rt.command)}`",
        f"- Failure class: `{state.get('failure_class')}`",
        f"- Failure location: `{state.get('traceback', {}).get('failing_file', '?')}:"
        f"{state.get('traceback', {}).get('failing_line', '?')}`" if state.get("traceback") else "",
        "",
        "## Root cause hypothesis",
        "",
        str(state.get("hypothesis", "")).strip() or "_(none)_",
        "",
        "## Remediation",
        "",
        f"- Kind: `{state.get('patch_kind')}`",
        f"- Target: `{_short(state.get('patch_target', ''))}`",
        f"- Rationale: {state.get('rationale', '')}",
        f"- Applied: {state.get('patch_applied', False)}",
        f"- Backup: `{state.get('backup_path')}`",
        "",
        "## Verification",
        "",
        str(state.get("verification", "")).strip() or "_(none)_",
        "",
        "## Diff",
        "",
        "```diff",
        _diff(state.get("patch_before", ""), state.get("patch_after", "")),
        "```",
        "",
    ]
    try:
        path.write_text("\n".join(x for x in lines if x is not None), encoding="utf-8")
    except OSError:
        pass
    return path


def _diff(before: str, after: str) -> str:
    """A minimal unified diff (avoids importing difflib into every log line)."""
    import difflib

    if not before and not after:
        return "(no textual change recorded)"
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile="before", tofile="after", lineterm="", n=3,
        )
    )