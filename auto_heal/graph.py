r"""
auto_heal.graph
===============

The LangGraph state machine.

Topology
--------

.. code-block:: text

    START
      |
      v
    monitor --(healthy)--> report --> END
      |
      v
    classify --(no traceback)--> escalate --> END
      |
      v
    hypothesise --(confidence < 0.5 or no patch)--> escalate --> END
      |
      v
    patch
      |
      v
    verify --(static check fails)--> rollback --> [attempts_left?] restart_loop
      |                                                        ^
      v                                                        |
    restart --(healthy)--> report --> END                      |
      |                                                        |
      v                                                        |
    restart_loop <---------------------------------------------┘

Why this shape
--------------
Three properties are deliberate:

1.  **Restart precedes patching.** The first cycle through
    ``monitor -> classify -> hypothesise -> patch`` is preceded by a restart
    attempt inside ``monitor`` itself. A large share of "crashes" in a
    ``uvicorn --reload`` workflow are transient (a stale worker, a port still
    held by the previous instance). Patching source for those would break working
    code.

2.  **Rollback precedes retry.** ``rollback`` restores the backup before control
    returns to ``monitor``. Attempt *N+1* therefore always begins from the same
    baseline as attempt 1, so a failed patch cannot compound.

3.  **Escalation is a first-class terminal state, not an exception.** An agent
    that exhausts its budget and stops, loudly and with a report, is behaving
    correctly. Silently continuing to poke at source is not.

Attempt counting
----------------
``attempt`` is incremented in :func:`~auto_heal.nodes.node_classify`, so it
reflects *classified failures* rather than graph iterations. The conditional edge
``route_after_restart`` compares it against ``max_attempts`` and terminates when
the budget is exhausted.

Checkpointing
-------------
``MemorySaver`` gives time-travel debugging: the graph can be inspected at any
node after the fact. ``stream_mode="updates"`` yields one dict per node so the
CLI can print progress live.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from .graph_state import FailureClass, HealState
from .nodes import (
    Runtime,
    node_classify,
    node_diagnose_llm,
    node_escalate,
    node_hypothesise,
    node_monitor,
    node_patch,
    node_report,
    node_restart,
    node_rollback,
    node_verify,
    set_runtime,
)

# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def route_after_monitor(state: HealState) -> Literal["classify", "report"]:
    """A healthy service goes straight to the report; otherwise classify.

    The check is an explicit ``is False`` rather than a truthiness test. An
    absent key reads as ``None``, which is falsy -- so ``state.get("failed")``
    on a partially-merged state would take the "healthy" branch and report
    success on a service that crashed. ``failed`` is absent precisely when the
    state has not been updated correctly, so the safe reading of "unknown" is
    *failure*, and that is the branch this deliberately takes.
    """
    return "report" if state.get("failed") is False else "classify"


def route_after_classify(state: HealState) -> Literal["hypothesise", "escalate"]:
    """A crash with no Python traceback is not patchable."""
    return "hypothesise" if state.get("traceback") else "escalate"


def route_after_hypothesise(state: HealState) -> Literal["patch", "diagnose_llm", "escalate"]:
    """Decide whether to patch, ask the LLM for a second opinion, or stop.

    The escalation to :func:`node_diagnose_llm` fires only for
    ``UNKNOWN`` -- a failure the rule base has no entry for. A *known* class with
    a rejected patch is a different situation: the rule already knows the cause,
    so a second opinion adds nothing and the correct action is to report.
    """
    if state.get("failure_class") == FailureClass.UNKNOWN.value:
        return "diagnose_llm"
    return "patch"


def route_after_patch(state: HealState) -> Literal["verify", "escalate"]:
    """Only a patch that actually landed gets verified."""
    return "verify" if state.get("patch_applied") else "escalate"


def route_after_verify(state: HealState) -> Literal["restart", "rollback"]:
    """Static verification passed -> restart; failed -> roll back first."""
    return "restart" if state.get("verified") else "rollback"


def route_after_restart(state: HealState) -> Literal["report", "monitor"]:
    """Recovered -> report; otherwise loop if attempts remain, else escalate."""
    if not state.get("failed"):
        return "report"
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3) or 3)
    return "monitor" if attempt < max_attempts else "escalate"


def route_after_rollback(state: HealState) -> Literal["monitor"]:
    """Always re-enter the loop after a rollback."""
    return "monitor"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

#: Fallback used when LangGraph is not installed, so the module still imports and
#: a clear error is raised at build time rather than an obscure ImportError.
try:  # pragma: no cover - import guard
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment]
    MemorySaver = None  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]


class HealGraph:
    """Compiled LangGraph plus the helpers needed to drive it.

    Parameters
    ----------
    runtime:
        The shared :class:`~auto_heal.nodes.Runtime` (logger, hypothesiser,
        server handle).  Installed into the node module on construction so the
        nodes can be plain functions.
    max_attempts:
        Overrides ``runtime.max_attempts`` in the initial state.
    checkpointer:
        ``True`` enables ``MemorySaver`` for time-travel debugging.
    """

    def __init__(
        self,
        runtime: Runtime,
        max_attempts: Optional[int] = None,
        checkpointer: bool = True,
    ) -> None:
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "langgraph is not installed. Install it with:\n"
                "    pip install langgraph langchain-core\n"
                "The deterministic rule-based hypothesiser does not require it, but "
                "the orchestrated state machine does."
            )
        set_runtime(runtime)
        self.runtime = runtime
        self.graph = self._build(bool(checkpointer))

    # -- construction ------------------------------------------------------
    def _build(self, checkpointer: bool) -> Any:
        """Assemble and compile the state graph."""
        g = StateGraph(HealState)

        g.add_node("monitor", node_monitor)
        g.add_node("classify", node_classify)
        g.add_node("hypothesise", node_hypothesise)
        g.add_node("diagnose_llm", node_diagnose_llm)
        g.add_node("patch", node_patch)
        g.add_node("verify", node_verify)
        g.add_node("rollback", node_rollback)
        g.add_node("restart", node_restart)
        g.add_node("report", node_report)
        g.add_node("escalate", node_escalate)

        g.add_edge(START, "monitor")
        g.add_conditional_edges("monitor", route_after_monitor,
                                {"classify": "classify", "report": "report"})
        g.add_conditional_edges("classify", route_after_classify,
                                {"hypothesise": "hypothesise", "escalate": "escalate"})
        g.add_conditional_edges("hypothesise", route_after_hypothesise,
                                {"patch": "patch", "diagnose_llm": "diagnose_llm",
                                 "escalate": "escalate"})
        g.add_edge("diagnose_llm", "patch")
        g.add_conditional_edges("patch", route_after_patch,
                                {"verify": "verify", "escalate": "escalate"})
        g.add_conditional_edges("verify", route_after_verify,
                                {"restart": "restart", "rollback": "rollback"})
        g.add_conditional_edges("restart", route_after_restart,
                                {"report": "report", "monitor": "monitor"})
        g.add_edge("rollback", "monitor")
        g.add_edge("report", END)
        g.add_edge("escalate", END)

        saver = MemorySaver() if checkpointer else None
        return g.compile(checkpointer=saver)

    # -- execution ---------------------------------------------------------
    def initial_state(
        self,
        command: List[str],
        cwd: str,
        log_path: str,
        max_attempts: Optional[int] = None,
        dry_run: bool = False,
    ) -> HealState:
        """Build the starting state dictionary."""
        attempts = max_attempts if max_attempts is not None else self.runtime.max_attempts
        return {
            "command": list(command),
            "cwd": cwd,
            "log_path": log_path,
            "max_attempts": int(attempts),
            "dry_run": bool(dry_run),
            "attempt": 0,
            # Declared here as well as in the TypedDict, so the conditional
            # edges have a defined value to read on the very first hop even if
            # a future LangGraph version stops merging unset keys into the
            # channel defaults.
            "failed": True,
            "traceback": None,
            "failure_class": FailureClass.UNKNOWN.value,
            "classified": False,
            "server_stdout": "",
            "server_stderr": "",
            "server_exit_code": None,
            "hypothesis": "",
            "confidence": 0.0,
            "rationale": "",
            "llm_used": False,
            "patch_target": None,
            "patch_kind": "none",
            "patch_before": None,
            "patch_after": None,
            "patch_find": None,
            "patch_replace": None,
            "patch_applied": False,
            "backup_path": None,
            "rolled_back": False,
            "verification": "",
            "verified": False,
            "verification_output": "",
            "status": "pending",
            "history": [],
            "started_at": time.time(),
            "finished_at": None,
            "report_path": None,
        }

    def run(
        self,
        command: List[str],
        cwd: str,
        log_path: str,
        max_attempts: Optional[int] = None,
        dry_run: bool = False,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """Execute the graph to completion.

        Parameters
        ----------
        stream:
            When true, print a line per node transition as it happens. The
            per-node updates are also accumulated into ``state['history']`` so
            the final report contains the full path.

        Returns
        -------
        dict
            The final state, including ``status`` (``"healed"``, ``"exhausted"``
            or ``"running"``) and ``history``.
        """
        initial = self.initial_state(command, cwd, log_path, max_attempts, dry_run)
        self.runtime.logger.info(
            "starting auto-heal graph",
            event="graph_start",
            command=" ".join(command),
            max_attempts=initial["max_attempts"],
            dry_run=dry_run,
            cwd=cwd,
        )
        self.runtime.logger.event("graph_start", {
            "command": command, "max_attempts": initial["max_attempts"],
            "dry_run": dry_run, "cwd": cwd,
        })

        history: List[Dict[str, Any]] = []
        final: Dict[str, Any] = dict(initial)

        # LangGraph >= 1.x requires a thread_id in the configurable block before
        # it will touch a checkpointer. Omitting it raises a ValueError from
        # MemorySaver the moment the first checkpoint is written, which aborts
        # the whole healing run at the graph's *first* node -- the diagnostic
        # state is fine and the monitor dies anyway.
        #
        # The id is derived from the log path and the start time so concurrent
        # monitors of the same repository land on distinct threads, and so a
        # re-run of the same invocation can be resumed by name.
        thread_id = f"heal-{Path(log_path).stem}-{int(initial['started_at'] * 1000)}"
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.graph.stream(initial, config=config,
                                       stream_mode="updates"):
            for node_name, update in (chunk or {}).items():
                if not isinstance(update, dict):
                    continue
                final.update(update)
                history.append({"node": node_name, "update": _slim(update)})
                if stream:
                    print(f"  [graph] {node_name:<14} -> {_summarise_update(update)}",
                          flush=True)

        final["history"] = history
        final["thread_id"] = thread_id
        return final

    def stream(self, command: List[str], cwd: str, log_path: str,
               max_attempts: Optional[int] = None, dry_run: bool = False):
        """Yield raw ``(node, update)`` pairs; useful for custom CLIs."""
        initial = self.initial_state(command, cwd, log_path, max_attempts, dry_run)
        thread_id = f"heal-{Path(log_path).stem}-{int(time.time() * 1000)}"
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self.graph.stream(initial, config=config,
                                       stream_mode="updates"):
            for node_name, update in (chunk or {}).items():
                yield node_name, update

    def invoke(self, **kwargs: Any) -> Dict[str, Any]:
        """One-shot invocation returning the final state."""
        return self.run(**kwargs)

    def describe(self) -> str:
        """A Mermaid diagram of the compiled graph (hand-maintained, stable)."""
        return (
            "```mermaid\n"
            "graph TD\n"
            "  START([START]) --> monitor[monitor: run server]\n"
            "  monitor -->|healthy| report[report: healed]\n"
            "  monitor -->|crash| classify[classify: parse traceback]\n"
            "  classify -->|no traceback| escalate[escalate: exhausted]\n"
            "  classify --> hypothesise[hypothesise: rules]\n"
            "  hypothesise -->|UNKNOWN| diagnose[diagnose_llm]\n"
            "  hypothesise -->|known class| patch[patch: apply]\n"
            "  diagnose --> patch\n"
            "  patch -->|not applied| escalate\n"
            "  patch -->|applied| verify[verify: import + compile sweep]\n"
            "  verify -->|static ok| restart[restart: relaunch]\n"
            "  verify -->|static fail| rollback[rollback: restore backup]\n"
            "  rollback --> monitor\n"
            "  restart -->|healthy| report\n"
            "  restart -->|still failing| monitor\n"
            "  report --> END([END])\n"
            "  escalate --> END\n"
            "```"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slim(update: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky fields from a node update before storing it in history."""
    drop = {"patch_before", "patch_after", "raw_traceback", "server_stdout", "history"}
    return {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 400 else v)
            for k, v in update.items() if k not in drop}


def _summarise_update(update: Dict[str, Any]) -> str:
    """One-line summary of a node's update for the CLI stream."""
    if update.get("failure_class"):
        return f"class={update['failure_class']}"
    if update.get("patch_applied"):
        return f"applied {update.get('patch_kind', '?')} patch"
    if "verified" in update:
        return f"verified={update['verified']}"
    if "hypothesis" in update:
        return f"confidence={update.get('confidence', 0):.2f}"
    if "status" in update:
        return f"status={update['status']}"
    if "failed" in update:
        return f"healthy={not update['failed']}"
    return "ok"


def build_graph(
    runtime: Runtime,
    max_attempts: Optional[int] = None,
    checkpointer: bool = True,
) -> HealGraph:
    """Factory: install ``runtime`` and compile the graph."""
    return HealGraph(runtime, max_attempts=max_attempts, checkpointer=checkpointer)


__all__ = [
    "HealGraph",
    "build_graph",
    "LANGGRAPH_AVAILABLE",
    "Runtime",
    "route_after_classify",
    "route_after_hypothesise",
    "route_after_monitor",
    "route_after_patch",
    "route_after_restart",
    "route_after_rollback",
    "route_after_verify",
]