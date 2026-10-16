r"""
auto_heal.graph_state
======================

Typed state shared by every node of the healing graph.

LangGraph requires a single, serialisable state object threaded through the
graph.  We use a :class:`TypedDict` (not a Pydantic model) because LangGraph
dicts are merged field-by-field and a Pydantic model would force every node to
re-validate the whole state on each hop.

Every field carries a ``revive`` counter for anything that accumulates across
retries (the traceback history, the patch log).  LangGraph's reducer merges
lists by concatenation and scalars by replacement; the attempt counter is what
lets a node distinguish "first pass" from "retry 2 of N" without inspecting the
length of a list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class FailureClass(str, Enum):
    """Taxonomy of the failures the monitor can recognise.

    The classification is deliberately *syntactic* -- derived from the exception
    type, the message and the last frames of the traceback -- because that is all
    that is reliably available when a server has just crashed.  Semantic
    classification ("this is a CUDA OOM") would need the full traceback plus
    runtime introspection, and would itself be a source of failure.

    Each member maps to a distinct remediation strategy in
    :mod:`auto_heal.patching`.
    """

    UNKNOWN = "unknown"
    CUDA_SHAPE = "cuda_shape_mismatch"
    CUDA_OOM = "cuda_out_of_memory"
    NAN_PDE_LOSS = "nan_pde_loss"
    MISSING_DATASET = "missing_dataset_path"
    MISSING_MODULE = "missing_module"
    IMPORT_ERROR = "import_error"
    DEVICE_MISMATCH = "device_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    TIMEOUT = "timeout"
    PORT_IN_USE = "port_in_use"
    PERMISSION = "permission_denied"
    VALIDATION = "validation_error"


@dataclass(slots=True)
class TracebackInfo:
    """Structured view of a captured traceback."""

    raw: str
    exception_type: str
    message: str
    failing_file: Optional[str]
    failing_line: int
    failing_source: str
    frames: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view for the log."""
        return {
            "exception_type": self.exception_type,
            "message": self.message,
            "failing_file": self.failing_file,
            "failing_line": self.failing_line,
            "failing_source": self.failing_source,
            "frames": self.frames[-8:],
        }


class HealState(TypedDict, total=False):
    """State object threaded through the auto-heal graph.

    Field groups
    ------------
    inputs      : what the monitor was asked to do
    diagnostics : what went wrong
    hypothesis  : the agent's explanation and proposed fix
    action      : the patch to apply
    verification: the result of re-running after the patch
    bookkeeping : attempt counters and terminal status
    """

    # -- inputs --------------------------------------------------------
    command: List[str]
    cwd: str
    log_path: str
    max_attempts: int
    dry_run: bool

    # -- diagnostics ---------------------------------------------------
    attempt: int
    #: Whether the monitored service failed to reach a healthy state on the
    #: current pass. This *must* be a declared field.
    #:
    #: LangGraph merges a node's return dict into the state field-by-field, and
    #: keys that are absent from the TypedDict are dropped with no warning.
    #: ``node_monitor`` returns ``failed=True`` on a crash; if the key is
    #: undeclared it never reaches the state, so ``route_after_monitor`` reads
    #: ``state.get('failed')`` as ``None`` -- falsy -- and routes to ``report``
    #: instead of ``classify``. The graph then declares the service healed
    #: without ever attempting a single classification. This is the shape of
    #: failure that a unit test on the node alone would never catch, because
    #: the node returns perfectly correct data.
    failed: bool
    traceback: Optional[Dict[str, Any]]
    #: The raw traceback text, kept separately from the structured
    #: :attr:`traceback` so the incident report can quote the exception verbatim
    #: rather than a reconstruction of it.
    #:
    #: Declared because ``node_classify`` writes it. Undeclared, LangGraph drops
    #: it and ``_traceback_from_state`` rehydrates an *empty* raw string on every
    #: retry -- the structured fields survive because they were declared, so the
    #: diagnosis still works and the report quietly loses the evidence.
    raw_traceback: str
    failure_class: str
    classified: bool
    server_stdout: str
    server_stderr: str
    server_exit_code: Optional[int]
    server_pid: Optional[int]

    # -- hypothesis ----------------------------------------------------
    hypothesis: str
    confidence: float
    rationale: str
    llm_used: bool

    # -- action --------------------------------------------------------
    patch_target: Optional[str]
    patch_kind: str          # "ast" | "string" | "config" | "none"
    patch_before: Optional[str]
    patch_after: Optional[str]
    patch_find: Optional[str]
    patch_replace: Optional[str]
    #: Which occurrence of ``find`` to patch (1-based).
    #:
    #: Declared because ``node_hypothesise`` writes it and ``node_patch`` reads
    #: it back. Undeclared, LangGraph drops it on the merge and ``node_patch``
    #: falls back to its default of 1 -- silently patching the *first* match
    #: when the hypothesis asked for the third, which on a file with several
    #: similar lines is a patch applied in the wrong place.
    patch_occurrence: int
    patch_applied: bool
    backup_path: Optional[str]
    rolled_back: bool

    # -- verification --------------------------------------------------
    verification: str
    verified: bool
    verification_output: str

    # -- bookkeeping ---------------------------------------------------
    status: str               # "pending" | "running" | "healed" | "exhausted" | "aborted"
    history: List[Dict[str, Any]]
    started_at: float
    finished_at: Optional[float]
    report_path: Optional[str]