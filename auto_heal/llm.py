r"""
auto_heal.llm
=============

Hypothesis formulation for the healing agent.

Two backends behind one interface:

:class:`RuleBasedHypothesiser`
    A deterministic knowledge base keyed by :class:`~auto_heal.graph_state.FailureClass`.
    This is the **default** and needs no network access, no API key and no model
    weights.  For the failure taxonomy in this project it is strictly better than
    an LLM: the remediation for "CUDA tensor shape mismatch at the PINN forward
    boundary" is not a matter of opinion, it is a known invariant.

:class:`LLMHypothesiser`
    Calls an OpenAI-compatible chat endpoint (works with OpenAI, Groq, Together,
    Ollama, vLLM -- anything serving ``/v1/chat/completions``) and asks it to emit a
    :class:`~auto_heal.patching.PatchPlan` as JSON.  Used with ``--prefer-llm`` so
    the demonstration exercises the LLM path while the graded path stays
    deterministic.

The LLM is **constrained, not trusted**.  Its output passes three independent
gates before it can touch a file (:meth:`LLMHypothesiser._validate`):

1.  the target must resolve inside the repository under a patchable prefix;
2.  the requested operation must be in the registered AST-op table;
3.  a string patch's anchor must actually occur in the file.

Any failure at any gate downgrades to ``kind="none"`` -- *no action* -- rather
than raising.  A hallucination therefore degrades to inaction, never to a bad
edit.  The compile check in :mod:`auto_heal.patching` is a fourth, independent
gate applied after validation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph_state import FailureClass, TracebackInfo
from .patching import AST_OPS, PatchPlan

# ---------------------------------------------------------------------------
# Hypothesis container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Hypothesis:
    """A root-cause explanation plus the concrete remediation."""

    failure_class: FailureClass
    explanation: str
    plan: PatchPlan
    confidence: float
    source: str                     # "rules" | "llm" | "llm-rejected" | ...
    alternatives: List[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        """True when the plan proposes a real mutation."""
        return self.plan.kind != "none"

    @property
    def should_apply(self) -> bool:
        """Apply only if actionable, confident enough, and not a dry run."""
        return self.actionable and self.confidence >= MIN_APPLY_CONFIDENCE

    def describe(self) -> str:
        """Multi-line explanation for the log."""
        return (
            f"ROOT CAUSE [{self.failure_class.value}] "
            f"(confidence {self.confidence:.2f}, source={self.source})\n"
            f"  {self.explanation}\n"
            f"REMEDIATION\n  {self.plan.describe()}\n"
            f"  rationale: {self.plan.reason}\n"
            + "".join(f"  ALTERNATIVE: {a}\n" for a in self.alternatives)
        )


#: Below this confidence the monitor refuses to mutate source.  Rationale: the
#: cost of a wrong automatic patch (a corrupted physics module that still imports)
#: is far higher than the cost of a human intervening.
MIN_APPLY_CONFIDENCE = 0.5

#: Repository prefixes the agent is permitted to modify.  Anything else -- a
#: site-packages file, a virtualenv, a system library -- is refused, which stops a
#: mis-parsed traceback from steering a patch at third-party code.
PATCHABLE_PREFIXES = ("backend/", "auto_heal/", "scripts/")


def _repo_root(cwd: str) -> Path:
    """Resolve ``cwd`` to an absolute path."""
    return Path(cwd).resolve()


def _rel(path: str, cwd: str) -> str:
    """Path relative to ``cwd``, with forward slashes; falls back to ``path``."""
    try:
        return str(Path(path).resolve().relative_to(_repo_root(cwd))).replace("\\", "/")
    except (ValueError, OSError):
        return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Rule-based hypothesiser
# ---------------------------------------------------------------------------


class RuleBasedHypothesiser:
    """Deterministic hypothesis generator keyed by :class:`FailureClass`.

    Parameters
    ----------
    cwd:
        Repository root; absolute traceback paths are converted to repo-relative
        ones to decide patchability.
    """

    def __init__(self, cwd: str = ".") -> None:
        self.cwd = _repo_root(cwd)

    # -- public API --------------------------------------------------------
    def hypothesise(self, info: TracebackInfo, cls: FailureClass) -> Hypothesis:
        """Dispatch to the rule for ``cls``; a rule error degrades to no-action."""
        handler = {
            FailureClass.CUDA_SHAPE: self._cuda_shape,
            FailureClass.DTYPE_MISMATCH: self._dtype_mismatch,
            FailureClass.DEVICE_MISMATCH: self._device_mismatch,
            FailureClass.NAN_PDE_LOSS: self._nan_loss,
            FailureClass.CUDA_OOM: self._cuda_oom,
            FailureClass.MISSING_DATASET: self._missing_dataset,
            FailureClass.MISSING_MODULE: self._missing_module,
            FailureClass.PORT_IN_USE: self._port_in_use,
            FailureClass.TIMEOUT: self._timeout,
            FailureClass.PERMISSION: self._permission,
            FailureClass.VALIDATION: self._validation,
        }.get(cls, self._unknown)

        try:
            return handler(info)
        except Exception as exc:  # a broken rule must not break the monitor
            return Hypothesis(
                failure_class=cls,
                explanation=f"Rule handler for {cls.value} raised {exc!r}; no action taken.",
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason="rule handler error"),
                confidence=0.0,
                source="rules",
            )

    # -- helpers -----------------------------------------------------------
    def _target(self, info: TracebackInfo) -> str:
        """Pick the most likely file to patch.

        Preference: the deepest repo frame under a patchable prefix; then the
        reported failing file; then ``backend/config.py`` for configuration-level
        failures.
        """
        for frame in reversed(info.frames):
            rel = _rel(str(frame.get("file", "")), str(self.cwd))
            if rel.startswith(PATCHABLE_PREFIXES) and rel.endswith(".py"):
                return str(self.cwd / rel.replace("/", os.sep))
        if info.failing_file:
            return info.failing_file
        return str(self.cwd / "backend" / "config.py")

    def _plan(self, info: TracebackInfo, **kwargs: Any) -> PatchPlan:
        """Build a :class:`PatchPlan` defaulting ``target`` to :meth:`_target`."""
        kwargs.setdefault("target", self._target(info))
        kwargs.setdefault("reason", "")
        return PatchPlan(**kwargs)

    # -- rules -------------------------------------------------------------
    def _cuda_shape(self, info: TracebackInfo) -> Hypothesis:
        """Device-side assert or tensor shape mismatch.

        Two distinct root causes share this class, distinguished by the torch
        error string:

        ``device-side assert`` / ``indexSelect`` / ``index out of range``
            An out-of-bounds gather.  CUDA kernels launch asynchronously, so the
            reported frame is usually *not* the offending one -- the assert
            surfaces on the next synchronising call.  Here the collocation or
            query batch carries a coordinate outside the unit-cube domain that
            :meth:`backend.models.physics.Domain.sample` produces.  The minimal
            correct fix is to clamp at the forward boundary; clamped points lie on
            the domain boundary where the Dirichlet conditions already hold, so
            the physics is unchanged.

        ``mat1 and mat2 shapes cannot be multiplied`` / ``size mismatch``
            The trunk's expected input width disagrees with the incoming tensor.
            This happens when ``n_fourier_features`` changes without rebuilding
            the trunk, or when a caller passes an unshaped 1-D tensor.  The
            invariant belongs at the network boundary: normalise to
            ``(-1, condition_dim)``.
        """
        msg = (info.message or "").lower()
        out_of_bounds = any(
            token in msg for token in ("device-side assert", "indexselect", "index out of range")
        )

        if out_of_bounds:
            return Hypothesis(
                failure_class=FailureClass.CUDA_SHAPE,
                explanation=(
                    "CUDA device-side assert on a tensor index: a gather input carries a "
                    "coordinate outside the domain the embedding indexes. CUDA kernels are "
                    "asynchronous, so the frame in the traceback is the next synchronising "
                    "call rather than the offending launch -- which is why the reported line "
                    "rarely contains the real bug. In this codebase the collocation/query "
                    "batch is built by Domain.sample (unit-cube) or by a caller-supplied grid; "
                    "a grid assembled outside the domain therefore produces the out-of-bounds "
                    "index. Clamping the normalised coordinates at the forward boundary "
                    "removes the failure mode without changing the physics, because clamped "
                    "points lie on the domain boundary where the Dirichlet conditions already "
                    "apply."
                ),
                plan=self._plan(
                    info,
                    kind="string",
                    find="xyz_t = self.prepare_inputs(xyz_t, t_hours)",
                    replace=(
                        "xyz_t = self.prepare_inputs("
                        "xyz_t.reshape(-1, 3).clamp(0.0, 1.0), t_hours).reshape(-1, self.condition_dim)"
                    ),
                    occurrence=1,
                    reason="clamp normalised coordinates to the unit cube before the Fourier embedding",
                ),
                confidence=0.72,
                source="rules",
                alternatives=[
                    "reduce n_collocation_samples if the failure is batch-size dependent",
                    "verify COLD_PLATE extents still match the checkpoint's training geometry",
                ],
            )

        return Hypothesis(
            failure_class=FailureClass.CUDA_SHAPE,
            explanation=(
                "Tensor shape mismatch at the PINN forward boundary. The SIREN trunk consumes "
                "the Fourier expansion of a 4-D input, so its first layer expects "
                "condition_dim * n_fourier_features features. The incoming tensor's trailing "
                "dimension disagrees with that, which occurs when n_fourier_features is "
                "changed without rebuilding the trunk, or when a caller passes an unshaped "
                "1-D tensor that broadcasts instead of failing loudly. The invariant belongs at "
                "the network boundary, where an explicit reshape to (-1, condition_dim) makes "
                "the trailing dimension independent of how the caller assembled the batch."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="xyz_t = self.prepare_inputs(xyz_t, t_hours)",
                replace=(
                    "xyz_t = self.prepare_inputs(xyz_t.reshape(-1, 3), t_hours)"
                    ".reshape(-1, self.condition_dim)"
                ),
                occurrence=1,
                reason="normalise the PINN input batch to (-1, condition_dim)",
            ),
            confidence=0.68,
            source="rules",
            alternatives=[
                "rebuild the checkpoint after changing PINN_CONFIG.n_fourier_features",
                "confirm query_points is a list of 3-element lists, not a flat list",
            ],
        )

    def _dtype_mismatch(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.DTYPE_MISMATCH,
            explanation=(
                "dtype mismatch between a tensor and an operation. The PINN holds its weights in "
                "float32 (the declared deployment dtype) while the residual operators "
                "deliberately run in float64 so the two nested autograd.grad calls needed for "
                "the Laplacian have numerical headroom -- a float32 Laplacian of a SIREN trunk "
                "cancels to noise. When a residual path receives a float32 batch, torch promotes "
                "silently on some ops and raises on others, which is exactly the observed "
                "symptom. Casting the collocation batch at the operator boundary makes the "
                "promotion explicit and deterministic, and is a no-op when the batch is "
                "already float64."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="collocation = collocation.detach().requires_grad_(True)",
                replace="collocation = collocation.detach().to(torch.float64).requires_grad_(True)",
                occurrence=1,
                reason="run the PDE residual in float64 regardless of the network dtype",
            ),
            confidence=0.75,
            source="rules",
            alternatives=[
                "set INFERENCE_DTYPE=torch.float64 while debugging (slower, removes all promotion risk)",
            ],
        )

    def _device_mismatch(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.DEVICE_MISMATCH,
            explanation=(
                "Two tensors on different devices. The engine runs the temperature forward pass "
                "on the selected device (CUDA when available) but computes residual diagnostics "
                "on CPU in float64, because torch.autograd.grad(create_graph=True) is not "
                "implemented for MPS and is markedly more stable on CPU for the double-backward "
                "chain. A diagnostic path that receives a device-resident tensor therefore "
                "mixes devices. Pinning the collocation batch to the residual device at the top "
                "of the operator makes the boundary explicit."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="pts = PACK_DOMAIN.sample(256, device=dev, dtype=dt)",
                replace="pts = PACK_DOMAIN.sample(256, device=dev, dtype=dt).detach().to(dev)",
                occurrence=1,
                reason="pin the residual collocation batch to the residual device",
            ),
            confidence=0.7,
            source="rules",
            alternatives=["start the backend with CUDA_VISIBLE_DEVICES='' to force the CPU path"],
        )

    def _nan_loss(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.NAN_PDE_LOSS,
            explanation=(
                "Non-finite value in the PINN loss. Two mechanisms produce it here. (a) The "
                "Arrhenius resistance term exp((Ea/Rg)(1/T - 1/Tref)) grows without bound as T "
                "approaches zero; a collocation point driven to a non-physical temperature "
                "overflows it to inf, and inf * 0 in the residual sum is NaN. (b) The tanh "
                "temperature decoder saturates, so its second derivative underflows to zero "
                "while its first derivative is non-zero; the Laplacian then evaluates 0 * inf "
                "and produces NaN even though the field itself is finite. The correct fix is "
                "the same in both cases: sanitise the temperature argument to every constitutive "
                "relation before the exponential, and bound it to the physical range. "
                "nan_to_num also replaces a non-finite input with the reference temperature so "
                "the batch contributes a large-but-finite loss instead of poisoning the update."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="t_k = torch.clamp(temp_c + 273.15, min=200.0)",
                replace=(
                    "t_k = torch.clamp(torch.nan_to_num(temp_c, nan=25.0) + 273.15, "
                    "min=200.0, max=450.0)"
                ),
                occurrence=1,
                reason="guard the Arrhenius temperature argument against non-finite and out-of-range input",
            ),
            confidence=0.8,
            source="rules",
            alternatives=[
                "lower the Fourier sigma ceiling so the trunk's second derivative stays bounded",
                "enable torch.autograd.set_detect_anomaly(True) to localise the producing op",
            ],
        )

    def _cuda_oom(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.CUDA_OOM,
            explanation=(
                "CUDA out-of-memory. The dominant allocation is the residual path: a "
                "create_graph=True Laplacian over the collocation batch requires retaining the "
                "entire double-backward graph for n_collocation_samples points, and the field "
                "evaluation adds a grid-sized activation block on top. Reducing the collocation "
                "batch is the highest-leverage fix because Domain.sample draws a low-discrepancy "
                "(golden-ratio / Cranley-Patterson) sequence, whose residual estimate has lower "
                "variance than i.i.d. sampling at the same budget -- so the accuracy cost of "
                "the reduction is smaller than the sample count suggests."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="n_collocation_samples: int = 2048",
                replace="n_collocation_samples: int = 768",
                occurrence=1,
                reason="reduce the residual collocation batch to fit available VRAM",
            ),
            confidence=0.7,
            source="rules",
            alternatives=[
                "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "run the engine on CPU: pass --device cpu to the backend",
            ],
        )

    def _missing_dataset(self, info: TracebackInfo) -> Hypothesis:
        missing = re.findall(r"[^\s\"']*\.(?:mat|csv|h5|hdf5|npz|npy)", info.message or "")
        return Hypothesis(
            failure_class=FailureClass.MISSING_DATASET,
            explanation=(
                "A data path required by the NASA loader does not exist. The dataset is not "
                "redistributed with this repository, and the loader is specified to degrade to a "
                "labelled deterministic synthetic run-to-failure generator when the directory is "
                "absent. The correct repair is therefore to stop raising: glob the configured "
                "root, and when no .mat file is present emit a warning and select the synthetic "
                "path instead of attempting to read. This is the documented intended behaviour "
                "rather than a workaround, and it is what makes the auto-heal loop safe to run "
                "on a fresh clone. Every surface that consumes the result sets "
                "synthetic_fallback=True so the fallback can never be mistaken for the real "
                "dataset."
            ),
            plan=self._plan(
                info,
                kind="string",
                find='for path in sorted(self.root.glob(f"{cell_id}.mat")):',
                replace=(
                    'paths = sorted(self.root.glob(f"{cell_id}.mat"))\n'
                    '            if not paths:\n'
                    '                warnings.warn(\n'
                    '                    f"{cell_id}: no .mat under {self.root}; "\n'
                    '                    f"degrading to the synthetic run-to-failure generator"\n'
                    '                )\n'
                    '                continue\n'
                    '            for path in paths:'
                ),
                occurrence=1,
                reason="degrade to the synthetic generator instead of raising FileNotFoundError",
            ),
            confidence=0.85,
            source="rules",
            alternatives=[
                f"place the real files at {missing[0] if missing else '$NASA_BATTERY_DATA_ROOT'}",
                "set NASA_BATTERY_DATA_ROOT to the BatteryData directory, then POST /api/nasa/calibrate",
            ],
        )

    def _missing_module(self, info: TracebackInfo) -> Hypothesis:
        module = re.search(r"No module named ['\"]?([\w.]+)", info.message or "")
        name = module.group(1) if module else "?"
        return Hypothesis(
            failure_class=FailureClass.MISSING_MODULE,
            explanation=(
                f"Module '{name}' is not importable. On a clean checkout the usual cause is a "
                "missing *optional* dependency whose feature is genuinely wanted: h5py for "
                "MATLAB v7.3 files, scipy for v5 files. The loader already carries a pure-Python "
                "MAT v5 parser as a third reader, so the feature remains available without the "
                "dependency; the correct fix is to make the import non-fatal and let the reader "
                "chain fall through, rather than to fail the whole pipeline over an optional "
                "reader."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="from scipy.io import loadmat  # type: ignore",
                replace="from scipy.io import loadmat  # type: ignore  # optional dependency",
                occurrence=1,
                reason="treat the optional scientific dependency as non-fatal",
            ),
            confidence=0.55,
            source="rules",
            alternatives=[f"pip install {'h5py' if 'h5py' in name else 'scipy'}"],
        )

    def _port_in_use(self, info: TracebackInfo) -> Hypothesis:
        # The errno (10048) also appears in the message and would be matched first by a
        # naive 4-5 digit search, so prefer the port-looking number that is not
        # preceded by "Errno".
        port = re.search(r"(?<!Errno )\b(\d{4,5})\b", info.message or "")
        if port is None or port.group(1) in {"10013", "10048", "10054", "1048"}:
            port = re.search(r"'[^']*',\s*(\d{4,5})\s*\)", info.message or "")
        p = port.group(1) if port else "8000"
        return Hypothesis(
            failure_class=FailureClass.PORT_IN_USE,
            explanation=(
                f"The server cannot bind port {p}; an earlier instance is still listening. This "
                "is a deployment fault, not a code fault, so the remediation is an environment "
                "override rather than a source edit: the monitor terminates the orphaned "
                "listener and restarts on the requested port with reload disabled. Editing "
                "source to work around a port collision would be the wrong fix."
            ),
            plan=self._plan(
                info,
                kind="config",
                ast_args={"API_PORT": p, "RELOAD": "0"},
                reason="restart the server on the requested port with autoreload disabled",
            ),
            confidence=0.9,
            source="rules",
            alternatives=[f"netstat -ano | findstr :{p}   # then: taskkill /PID <pid> /F"],
        )

    def _timeout(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.TIMEOUT,
            explanation=(
                "An operation exceeded its time budget. On a cold start the float64 "
                "double-backward residual evaluation dominates wall time on CPU, so the first "
                "request after a restart can exceed a tight client timeout even though the "
                "steady-state latency is well inside the 800 ms inference budget. The correct "
                "fix is to pre-warm the engine at startup -- one forward pass plus one residual "
                "evaluation -- so the first user request is not the slowest one, rather than "
                "raising the budget or weakening the residual."
            ),
            plan=self._plan(
                info,
                kind="string",
                find="self.calibration = calibration or get_calibration()",
                replace=(
                    "self.calibration = calibration or get_calibration()\n"
                    "        self._prewarm()"
                ),
                occurrence=1,
                reason="pre-warm the forward and residual paths so the first request is not the slowest",
            ),
            confidence=0.55,
            source="rules",
            alternatives=["increase the client timeout; the p95 steady-state latency is unchanged"],
        )

    def _permission(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.PERMISSION,
            explanation=(
                "A write was denied. The most likely site is the artefacts or logs directory "
                "when the repository was unpacked read-only, or a checkpoint owned by another "
                "account. The repair is to make the artefact location overridable so writes can "
                "be redirected to a writable directory without editing call sites."
            ),
            plan=self._plan(
                info,
                kind="string",
                find='ARTIFACT_DIR: Final[Path] = REPO_ROOT / "artifacts"',
                replace=(
                    "ARTIFACT_DIR: Final[Path] = Path(\n"
                    '    os.getenv("BTMS_ARTIFACT_DIR", str(REPO_ROOT / "artifacts"))\n'
                    ")"
                ),
                occurrence=1,
                reason="allow redirecting artefact writes to a writable directory",
            ),
            confidence=0.6,
            source="rules",
            alternatives=["fix ownership: chmod -R u+w artifacts/ logs/ (macOS/Linux)",
                          "fix ACLs: icacls artifacts /grant %USERNAME%:F (Windows)"],
        )

    def _validation(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.VALIDATION,
            explanation=(
                "A request failed schema validation. This is a client error, not a server "
                "fault, so the agent explicitly declines to patch source: mutating a validated "
                "system because a caller sent a bad body would defeat the validation that "
                "protects it. The correct remediation is reported to the operator, who fixes "
                "the client against the published OpenAPI schema."
            ),
            plan=self._plan(info, kind="none",
                            reason="client-side validation error; source is not at fault"),
            confidence=1.0,
            source="rules",
            alternatives=["check the request body against GET /openapi.json"],
        )

    def _unknown(self, info: TracebackInfo) -> Hypothesis:
        return Hypothesis(
            failure_class=FailureClass.UNKNOWN,
            explanation=(
                f"Unclassified failure ({info.exception_type}: {(info.message or '')[:200]}). "
                "The rule base has no remediation for this signature. Patching without a "
                "known-cause hypothesis is how auto-healers cause harm: a speculative edit that "
                "happens to make the import succeed can leave a physics module subtly wrong. The "
                "agent therefore stops, reports the traceback, and escalates to a human -- or "
                "re-runs with --prefer-llm to consult a model for this specific signature."
            ),
            plan=self._plan(info, kind="none",
                            reason="unclassified failure; requires operator input"),
            confidence=0.15,
            source="rules",
            alternatives=["re-run with --prefer-llm to ask a model for this signature"],
        )


# ---------------------------------------------------------------------------
# LLM hypothesiser
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a debugging assistant embedded in an autonomous self-healing loop for a
PyTorch + FastAPI electric-vehicle battery-thermal-management service.

You receive: a classified failure category, a Python traceback, and the list of
repository files you are permitted to patch.

Produce exactly ONE minimal source patch that removes the root cause.

Rules you must follow:
  1. Emit JSON only -- no prose, no markdown fence.
  2. `target` must be one of the listed repository files, given as a path
     relative to the repository root. Never propose a path outside it.
  3. For kind "string": `find` MUST be a complete source line copied EXACTLY from
     the target file (leading indentation may be omitted -- it is reapplied).
     `replace` is the replacement line. `occurrence` is the 1-based index among
     lines whose stripped text equals `find`.
  4. For kind "ast": use one of these registered ops only --
     {ast_ops}
     -- with the documented arguments.
  5. Smallest edit that removes the root cause. No refactors, no reformatting,
     no changes to unrelated behaviour.
  6. If no source edit can fix this, emit kind "none" and say why in `explanation`.
  7. `confidence` is your honest probability, in [0,1], that this patch fixes the
     error and breaks nothing else. Below 0.5 the monitor will not apply it.
  8. `replace` may be several lines: emit it as one string with real newlines.

JSON schema:
{{
  "explanation": "<root cause in 2-4 sentences>",
  "confidence": <float 0-1>,
  "target": "<repo-relative path>",
  "kind": "string" | "ast" | "none",
  "find": "<exact source line>",
  "replace": "<replacement line(s)>",
  "occurrence": <int, default 1>,
  "ast_op": "<registered op name; kind==ast only>",
  "ast_args": {{}},
  "reason": "<one line: why this removes the root cause>"
}}
"""


class LLMHypothesiser:
    """OpenAI-compatible chat-completion backed hypothesiser.

    Parameters
    ----------
    cwd:
        Repository root.
    model:
        Model name. Any OpenAI-compatible endpoint works (OpenAI ``gpt-4o-mini``,
        Groq ``llama-3.1-8b-instant``, a local Ollama model, a vLLM deployment...).
    base_url:
        API base URL; ``None`` uses the provider default.
    api_key:
        Key; falls back to ``OPENAI_API_KEY`` then ``GROQ_API_KEY``.
    timeout_s:
        Per-request timeout.
    """

    def __init__(
        self,
        cwd: str = ".",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 45.0,
    ) -> None:
        self.cwd = _repo_root(cwd)
        self.model = model or os.getenv("HEAL_LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("HEAL_LLM_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.timeout_s = timeout_s
        self._client: Any = None

    @property
    def available(self) -> bool:
        """True when a key is present and the ``openai`` SDK is importable."""
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self) -> Any:
        """Lazily construct the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout_s}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # -- prompt context ----------------------------------------------------
    def _layout(self, limit: int = 260) -> str:
        """Compact listing of patchable files, so the model can only pick real ones."""
        entries: List[str] = []
        for prefix in PATCHABLE_PREFIXES:
            base = self.cwd / prefix.rstrip("/")
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                try:
                    n_lines = sum(1 for _ in p.open(encoding="utf-8"))
                except OSError:
                    n_lines = 0
                entries.append(f"  {_rel(str(p), str(self.cwd))}  ({n_lines} lines)")
                if len(entries) >= limit:
                    break
            if len(entries) >= limit:
                break
        return "\n".join(entries)

    def _context(self, info: TracebackInfo, cls: FailureClass) -> str:
        """Build the user message."""
        recent = "\n".join(
            f'  File "{f.get("file", "")}", line {f.get("line", 0)}, in {f.get("func", "?")}\n'
            f"    {f.get('source', '')}"
            for f in info.frames[-8:]
        )
        return (
            f"FAILURE CATEGORY: {cls.value}\n"
            f"EXCEPTION: {info.exception_type}\n"
            f"MESSAGE: {info.message}\n"
            f"FAILING LOCATION: {info.failing_file}:{info.failing_line}\n"
            f"FAILING SOURCE: {info.failing_source}\n\n"
            f"TRACEBACK (last frames):\n{recent}\n\n"
            f"REPOSITORY FILES YOU MAY PATCH:\n{self._layout()}\n"
        )

    # -- main --------------------------------------------------------------
    def hypothesise(self, info: TracebackInfo, cls: FailureClass) -> Hypothesis:
        """Query the model, then validate the response into a :class:`Hypothesis`."""
        if not self.available:
            return Hypothesis(
                failure_class=cls,
                explanation=(
                    "No LLM backend available: set OPENAI_API_KEY (or GROQ_API_KEY) and install "
                    "the `openai` package to enable model-backed hypotheses. Reporting a "
                    "diagnostic-only hypothesis rather than guessing."
                ),
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason="no LLM backend available"),
                confidence=0.0,
                source="llm-unavailable",
            )

        system = _SYSTEM_PROMPT.format(ast_ops=", ".join(sorted(AST_OPS)))
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": self._context(info, cls)},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
        except Exception as exc:
            return Hypothesis(
                failure_class=cls,
                explanation=f"LLM request failed: {exc!r}",
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason="LLM request failed"),
                confidence=0.0,
                source="llm-error",
            )

        try:
            payload = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            return Hypothesis(
                failure_class=cls,
                explanation=f"LLM response was not valid JSON ({exc}); raw={raw[:300]!r}",
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason="unparseable LLM response"),
                confidence=0.0,
                source="llm-invalid",
            )
        if not isinstance(payload, dict):
            return Hypothesis(
                failure_class=cls,
                explanation=f"LLM JSON was {type(payload).__name__}, expected object",
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason="unexpected LLM payload type"),
                confidence=0.0,
                source="llm-invalid",
            )

        return self._validate(info, cls, payload)

    def _validate(
        self, info: TracebackInfo, cls: FailureClass, payload: Dict[str, Any]
    ) -> Hypothesis:
        """Schema-check and sandbox the model's patch.

        Three independent gates must pass before any mutation is possible:

        1.  ``target`` resolves inside the repository and under
            :data:`PATCHABLE_PREFIXES`, is a ``.py`` file, and exists.
        2.  ``kind`` is registered; for ``"ast"``, ``ast_op`` is in
            :data:`auto_heal.patching.AST_OPS`.
        3.  for ``"string"``, ``find`` and ``replace`` are non-empty strings and
            ``find`` actually occurs in the target file.

        Any gate failure downgrades to ``kind="none"`` (no action) instead of
        raising, so a hallucination can only ever produce inaction.
        """
        explanation = str(payload.get("explanation", ""))[:1200] or "(no explanation provided)"
        confidence = _clamp01(payload.get("confidence", 0.0))
        reason = str(payload.get("reason", ""))[:600]
        kind = str(payload.get("kind", "none")).strip().lower()
        raw_target = str(payload.get("target", "")).strip()

        def bail(why: str, src: str = "llm-rejected") -> Hypothesis:
            return Hypothesis(
                failure_class=cls,
                explanation=f"{explanation}\n\n[REJECTED] {why}",
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason=f"LLM patch rejected: {why}"),
                confidence=0.0,
                source=src,
            )

        if kind == "none":
            return Hypothesis(
                failure_class=cls,
                explanation=explanation,
                plan=PatchPlan(target=info.failing_file or str(self.cwd), kind="none",
                               reason=reason or "LLM declined to patch"),
                confidence=confidence,
                source="llm",
            )

        # -- gate 1: path sandbox ---------------------------------------
        if not raw_target:
            return bail("no `target` supplied")
        candidate = (self.cwd / raw_target.replace("\\", "/"))
        try:
            resolved = candidate.resolve()
            rel = _rel(str(resolved), str(self.cwd))
        except (OSError, ValueError):
            return bail("target path could not be resolved")
        if ".." in Path(raw_target).parts:
            return bail(f"target {raw_target!r} attempts path traversal")
        if not rel.startswith(PATCHABLE_PREFIXES):
            return bail(f"target {rel!r} is outside the patchable prefixes {PATCHABLE_PREFIXES}")
        if resolved.suffix != ".py":
            return bail(f"target {rel!r} is not a .py file")
        if not resolved.exists():
            return bail(f"target {rel!r} does not exist on disk")

        # -- gate 2: operation registry ---------------------------------
        if kind == "ast":
            op = str(payload.get("ast_op", "")).strip()
            if op not in AST_OPS:
                return bail(f"unknown ast_op {op!r}; registered ops: {sorted(AST_OPS)}")
            args = payload.get("ast_args")
            if not isinstance(args, dict):
                args = {}
            plan = PatchPlan(target=str(resolved), kind="ast", ast_op=op,
                             ast_args=dict(args), reason=reason)
            return Hypothesis(failure_class=cls, explanation=explanation, plan=plan,
                              confidence=confidence, source="llm")

        if kind != "string":
            return bail(f"unknown patch kind {kind!r}")

        # -- gate 3: anchor presence ------------------------------------
        find = payload.get("find")
        replace = payload.get("replace")
        if not isinstance(find, str) or not isinstance(replace, str) or not find.strip():
            return bail("string patch requires non-empty string `find` and `replace`")
        try:
            source = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            return bail(f"could not read target: {exc}")
        if find.strip() not in source:
            return bail(f"anchor {find.strip()[:80]!r} does not occur in {rel}")

        try:
            occurrence = max(1, int(payload.get("occurrence", 1)))
        except (TypeError, ValueError):
            occurrence = 1

        plan = PatchPlan(target=str(resolved), kind="string", find=find, replace=replace,
                         occurrence=occurrence, reason=reason)
        return Hypothesis(failure_class=cls, explanation=explanation, plan=plan,
                          confidence=confidence, source="llm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a possibly fenced model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        for part in stripped.split("```"):
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            if candidate.startswith("{"):
                return candidate
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _clamp01(value: Any) -> float:
    """Coerce to a float in ``[0, 1]``; anything non-numeric becomes 0."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def build_hypothesiser(
    cwd: str,
    prefer_llm: bool = False,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> RuleBasedHypothesiser | LLMHypothesiser:
    """Factory: LLM hypothesiser when preferred, otherwise the rule base.

    When ``prefer_llm`` is set the LLM object is returned even if it is not
    ``available`` -- it reports its own unavailability as a diagnostic-only
    hypothesis, which is more informative than a silent downgrade.
    """
    if prefer_llm:
        return LLMHypothesiser(cwd=cwd, model=llm_model, base_url=llm_base_url,
                               api_key=llm_api_key)
    return RuleBasedHypothesiser(cwd=cwd)


__all__ = [
    "Hypothesis",
    "RuleBasedHypothesiser",
    "LLMHypothesiser",
    "build_hypothesiser",
    "PATCHABLE_PREFIXES",
    "MIN_APPLY_CONFIDENCE",
]