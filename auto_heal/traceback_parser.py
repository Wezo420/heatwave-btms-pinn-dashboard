r"""
auto_heal.traceback_parser
==========================

Parse a raw traceback into a :class:`auto_heal.graph_state.TracebackInfo` and
classify it into a :class:`auto_heal.graph_state.FailureClass`.

Why a hand-rolled parser
------------------------
``traceback`` in the standard library gives structured frames, but the *last frame
in our own repository* -- the one whose source we can patch -- is not the same as
the last frame overall.  Under uvicorn the failing frame is often inside ``torch``
or ``fastapi``, and the actionable line is several frames up.
``TracebackInfo.failing_file`` therefore resolves to the innermost frame whose path
lies inside the monitored repository root; if none does, it falls back to the
innermost frame overall so the classifier still has something to work with.

The classifier is a priority-ordered rule list.  Order matters: a
``RuntimeError: CUDA error: device-side assert`` contains the word "index" in
torch's string, and a naive substring scan would mis-classify it as a shape
mismatch when it is really a device-side assert from an out-of-bounds index.
Specific patterns therefore come before general ones.
"""

from __future__ import annotations

import re
import traceback as tb_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .graph_state import FailureClass, TracebackInfo

#: Patterns matched against ``"{exception_type}: {message}"`` plus recent source.
#: Order is significant -- first match wins.
_RULES: List[Tuple[FailureClass, str]] = [
    # --- NaN / numerical ------------------------------------------------
    (
        FailureClass.NAN_PDE_LOSS,
        r"\bnan\b|\binf\b|non-?finite|loss became|numerical overflow|"
        r"invalid value encountered|float\('-?inf'\)",
    ),
    (FailureClass.NAN_PDE_LOSS, r"grad(?:ient)?\s+(?:is\s+)?nan"),

    # --- CUDA device assertions (before the general shape rules) -------
    (FailureClass.CUDA_SHAPE, r"device[- ]side assert triggered"),
    (FailureClass.CUDA_SHAPE, r"index(?:select|out of bounds|out of range)"),
    (
        FailureClass.CUDA_OOM,
        r"out of memory|cuda.*oom|allocat\w*\s+\d+\s*GiB|"
        r"cudaerroroutofmemory|failed to allocate|cannot allocate",
    ),

    # --- shape / broadcasting ------------------------------------------
    (
        FailureClass.CUDA_SHAPE,
        r"shape|dimension|size mismatch|broadcast|must match|"
        r"expected\s+\d+\s+dimensions|inconsistent tensor|"
        r"too many values to unpack|not enough values to unpack",
    ),
    (
        FailureClass.CUDA_SHAPE,
        r"mat1 and mat2 shapes cannot be multiplied|expected input batch",
    ),

    # --- dtype ----------------------------------------------------------
    (
        FailureClass.DTYPE_MISMATCH,
        r"expected\s+(?:dtype|torch\.\w+)|can only (?:be )?(?:converted|cast)|"
        r"expected scalar type|"
        r"expected dtype .* got",
    ),
    (FailureClass.DTYPE_MISMATCH, r"bfloat16|float16.*(?:not supported|unsupported)"),

    # --- device placement ------------------------------------------------
    (
        FailureClass.DEVICE_MISMATCH,
        r"expected all tensors to be on the same device|expected device cuda|"
        r"found device cpu|device cuda:\d|does not match the device of",
    ),

    # --- dataset / files --------------------------------------------------
    (
        FailureClass.MISSING_DATASET,
        r"no such file or directory.*\.(?:mat|csv|h5|hdf5|npz|npy)|"
        r"batterydata|run-to-failure",
    ),
    (FailureClass.MISSING_DATASET, r"filenotfounderror.*\.(?:mat|csv|h5|hdf5)"),
    (
        FailureClass.PERMISSION,
        r"permission denied|access is denied|read-only file system",
    ),
    (FailureClass.MISSING_DATASET, r"cannot find the files|unable to (?:find|open|decode)"),

    # --- imports ----------------------------------------------------------
    (FailureClass.MISSING_MODULE, r"modulenotfounderror|no module named"),
    (FailureClass.IMPORT_ERROR, r"importerror|cannot import name"),

    # --- infra ------------------------------------------------------------
    (
        FailureClass.PORT_IN_USE,
        r"address already in use|error while attempting to bind on address|"
        r"only one usage of each socket address",
    ),
    (FailureClass.TIMEOUT, r"timed? ?out|timeout|deadline exceeded"),

    # --- request validation ----------------------------------------------
    (FailureClass.VALIDATION, r"validationerror|unprocessable"),
]

_COMPILED_RULES: List[Tuple[FailureClass, "re.Pattern[str]"]] = [
    (cls, re.compile(pat, re.IGNORECASE)) for cls, pat in _RULES
]

#: ``File "path", line 42, in fn`` -- the standard CPython traceback frame header.
_FRAME_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')


def extract_traceback(exc: BaseException) -> TracebackInfo:
    """Build a :class:`TracebackInfo` from a live exception."""
    raw = "".join(tb_module.format_exception(type(exc), exc, exc.__traceback__))
    return parse_traceback(raw)


def parse_traceback(raw: str) -> TracebackInfo:
    """Parse a formatted traceback string into structured form.

    Parameters
    ----------
    raw:
        Output of ``traceback.format_exc()`` or an equivalent.

    Returns
    -------
    TracebackInfo
        ``failing_file`` is the innermost frame inside the repository when one
        exists; otherwise the innermost frame overall.  ``failing_source`` is the
        displayed source line for that frame, or ``""``.
    """
    lines = raw.rstrip().splitlines()

    exception_type = "Exception"
    message = ""
    type_pat = r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit)"
    tail: List[str] = []
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith(("File ", "Traceback")):
            continue
        if re.match(type_pat, stripped):
            if ":" in stripped:
                exception_type, _, message = stripped.partition(":")
                message = message.strip()
                # Continuation lines follow the exception line. FileNotFoundError
                # puts the offending path there, and that path *is* the diagnostic
                # the classifier needs -- stopping at the first line would report
                # only "No such file or directory:" and classify as UNKNOWN.
                tail = [ln.strip() for ln in lines[i + 1:] if ln.strip()]
            else:
                exception_type = stripped
            break
        tail.append(stripped)
    if tail:
        message = (message + " " + " ".join(reversed(tail))).strip()
    if not message and lines:
        message = lines[-1].strip()

    frames: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    for i, line in enumerate(lines):
        m = _FRAME_RE.search(line)
        if m:
            frame: Dict[str, Any] = {
                "file": m.group("file"),
                "line": int(m.group("line")),
                "func": m.group("func"),
                "source": "",
            }
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if nxt.strip() and nxt.startswith((" ", "\t")) and not _FRAME_RE.search(nxt):
                    frame["source"] = nxt.strip()
            frames.append(frame)
            pending = frame
        elif pending is not None and line.strip().startswith(("raise ", "assert ")):
            pending["source"] = line.strip()
            pending = None

    repo_root = Path.cwd().resolve()
    in_repo = [f for f in frames if _is_within(f["file"], repo_root)]
    target = (in_repo or frames or [{"file": "", "line": 0, "func": "?", "source": ""}])[-1]

    return TracebackInfo(
        raw=raw,
        exception_type=exception_type,
        message=message,
        failing_file=target["file"],
        failing_line=int(target["line"]),
        failing_source=target["source"],
        frames=frames,
    )


def _is_within(path: str, root: Path) -> bool:
    """True when ``path`` resolves inside ``root`` (best effort)."""
    if not path:
        return False
    try:
        p = Path(path)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.is_file():
            return False
        return root == p or root in p.parents
    except (OSError, ValueError):
        return False


def classify(info: TracebackInfo) -> FailureClass:
    """Map a traceback to a :class:`FailureClass`.

    The haystack is ``"{exception_type}: {message}"`` plus the last few source
    lines, because some failures (a bare ``assert T.shape == (...)``) put the
    diagnostic only in the source, not in the message.
    """
    tail = "\n".join(f.get("source", "") for f in info.frames[-4:])
    haystack = f"{info.exception_type}: {info.message}\n{tail}"
    for cls, pattern in _COMPILED_RULES:
        if pattern.search(haystack):
            return cls
    return FailureClass.UNKNOWN


def classify_exception(exc: BaseException) -> FailureClass:
    """Convenience: classify a live exception."""
    return classify(extract_traceback(exc))


def summarise(info: TracebackInfo, cls: FailureClass) -> str:
    """One-line human summary used in the log and the CLI banner."""
    if info.failing_file:
        where = f"{Path(info.failing_file).name}:{info.failing_line}"
    else:
        where = "<unknown>"
    return f"[{cls.value}] {info.exception_type} at {where} -- {info.message[:160]}"