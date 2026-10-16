r"""
auto_heal
=========

Autonomous self-healing monitor for the FastAPI + PyTorch backend.

Public surface
--------------
:class:`HealGraph`
    The compiled LangGraph state machine.
:class:`Runtime`
    Shared context: logger, hypothesiser, server handle.
:func:`build_graph`
    Factory that wires a :class:`Runtime` into a graph.
:class:`RuleBasedHypothesiser` / :class:`LLMHypothesiser`
    Hypothesis generators (deterministic by default, LLM optional).
:class:`FailureClass`
    The failure taxonomy the monitor recognises.
"""

from .graph_state import FailureClass, HealState, TracebackInfo
from .graph import HealGraph, build_graph, LANGGRAPH_AVAILABLE
from .llm import (
    Hypothesis,
    LLMHypothesiser,
    RuleBasedHypothesiser,
    build_hypothesiser,
    MIN_APPLY_CONFIDENCE,
    PATCHABLE_PREFIXES,
)
from .logger import HealLogger
from .nodes import Runtime, set_runtime
from .patching import (
    PatchPlan,
    PatchResult,
    apply_patch,
    backup_file,
    compile_check,
    import_check,
    restore,
    undo,
)
from .traceback_parser import (
    classify,
    classify_exception,
    extract_traceback,
    parse_traceback,
    summarise,
)

__all__ = [
    "FailureClass",
    "HealState",
    "TracebackInfo",
    "HealGraph",
    "build_graph",
    "LANGGRAPH_AVAILABLE",
    "Hypothesis",
    "LLMHypothesiser",
    "RuleBasedHypothesiser",
    "build_hypothesiser",
    "MIN_APPLY_CONFIDENCE",
    "PATCHABLE_PREFIXES",
    "HealLogger",
    "Runtime",
    "set_runtime",
    "PatchPlan",
    "PatchResult",
    "apply_patch",
    "backup_file",
    "compile_check",
    "import_check",
    "restore",
    "undo",
    "classify",
    "classify_exception",
    "extract_traceback",
    "parse_traceback",
    "summarise",
]

__version__ = "1.0.0"