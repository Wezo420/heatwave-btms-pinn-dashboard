"""Tests for the auto-heal traceback parser, patcher and hypothesiser."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from auto_heal.graph_state import FailureClass, TracebackInfo
from auto_heal.llm import (
    MIN_APPLY_CONFIDENCE,
    PATCHABLE_PREFIXES,
    Hypothesis,
    LLMHypothesiser,
    RuleBasedHypothesiser,
)
from auto_heal.patching import (
    PatchPlan,
    apply_patch,
    backup_file,
    compile_check,
    restore,
    syntax_all,
    undo,
)
from auto_heal.traceback_parser import classify, classify_exception, parse_traceback

SHAPE_TB = textwrap.dedent('''
    Traceback (most recent call last):
      File "C:/proj/backend/models/pinn.py", line 512, in forward
        feature, _ = self.encode(xyz_t, chem_latent)
      File "C:/proj/backend/models/pinn.py", line 470, in encode
        h = self.trunk[0](feats)
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x4 and 1x96)
''')

DEVICE_TB = textwrap.dedent('''
    Traceback (most recent call last):
      File "C:/proj/backend/services/inference.py", line 300, in _residuals
        r_solid = residual.solid_residual(colloc, current_a, soc, heat_watts)
    RuntimeError: Expected all tensors to be on the same device, but found at least two
    devices, cuda:0 and cpu!
''')

NAN_TB = textwrap.dedent('''
    Traceback (most recent call last):
      File "C:/proj/backend/models/physics.py", line 214, in internal_resistance
        arrhenius = torch.exp((spec.ea / R_GAS) * (1.0 / T_REF_K - 1.0 / t_k))
    RuntimeError: PDE residual produced a non-finite value (nan) in the loss term
''')

DATASET_TB = textwrap.dedent('''
    Traceback (most recent call last):
      File "C:/proj/backend/models/nasa_dataset.py", line 300, in _load_cycles
        raw = _read_mat(path)
    FileNotFoundError: [Errno 2] No such file or directory:
    'C:/proj/backend/data/nasa/BatteryData/B0005.mat'
''')

PORT_TB = textwrap.dedent('''
    Traceback (most recent call last):
      File "C:/proj/uvicorn/server.py", line 71, in startup
        sock.bind(sockaddr)
    OSError: [Errno 10048] error while attempting to bind on address
    ('127.0.0.1', 8000): address already in use
''')

VALIDATION_TB = textwrap.dedent('''
    Traceback (most recent call last):
      File "C:/proj/backend/main.py", line 120, in handler
        raise exc
    fastapi.exceptions.RequestValidationError: body -> operating_point -> coolant_flow_lpm
      Input should be less than or equal to 14
''')


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestTracebackParser:
    def test_parses_type_message_and_location(self):
        info = parse_traceback(SHAPE_TB)
        assert info.exception_type == "RuntimeError"
        assert "mat1 and mat2 shapes cannot be multiplied" in info.message
        assert info.failing_file.endswith("pinn.py")
        assert info.failing_line > 0
        assert len(info.frames) == 2

    def test_prefers_the_innermost_repo_frame(self, monkeypatch, tmp_path):
        """The actionable frame is the deepest one inside the repo."""
        monkeypatch.chdir(tmp_path)
        tb = (
            'Traceback (most recent call last):\n'
            '  File "C:/Python311/site-packages/torch/nn/modules/linear.py", line 1, in f\n'
            '    x\n'
            '  File "C:/proj/backend/models/pinn.py", line 42, in g\n'
            '    y\n'
            'RuntimeError: boom\n'
        )
        info = parse_traceback(tb)
        assert info.failing_file.endswith("pinn.py")
        assert info.failing_line == 42

    def test_multiline_message_is_captured(self):
        info = parse_traceback(DEVICE_TB)
        assert "device" in info.message.lower()
        assert "cuda" in info.message.lower()

    def test_source_line_is_captured(self):
        """``failing_source`` is the source of the *deepest* repo frame.

        SHAPE_TB has two frames in the repository; the reported one is the
        innermost (``pinn.py:470``, inside ``encode``), not the outermost
        caller -- the whole point of preferring the deepest repo frame is that
        the outer frame is usually not where the fix goes.
        """
        info = parse_traceback(SHAPE_TB)
        assert info.failing_line == 470
        assert "self.trunk[0]" in info.failing_source

    def test_classify_exception_from_live_exception(self):
        try:
            raise ValueError("something broke badly in the encoder")
        except ValueError as exc:
            cls = classify_exception(exc)
        assert cls in FailureClass


class TestClassifier:
    @pytest.mark.parametrize("tb,expected", [
        (SHAPE_TB, FailureClass.CUDA_SHAPE),
        (DEVICE_TB, FailureClass.DEVICE_MISMATCH),
        (NAN_TB, FailureClass.NAN_PDE_LOSS),
        (DATASET_TB, FailureClass.MISSING_DATASET),
        (PORT_TB, FailureClass.PORT_IN_USE),
        (VALIDATION_TB, FailureClass.VALIDATION),
    ])
    def test_priority_order(self, tb, expected):
        assert classify(parse_traceback(tb)) is expected

    def test_cuda_oom_beats_generic_shape(self):
        """OOM mentions 'size'/'shape' -- the specific rule must win."""
        tb = textwrap.dedent('''
            Traceback (most recent call last):
              File "C:/proj/backend/main.py", line 10, in f
                x
            torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.20 GiB
            (GPU 0; 8.00 GiB total capacity)
        ''')
        assert classify(parse_traceback(tb)) is FailureClass.CUDA_OOM

    def test_device_side_assert_is_a_shape_failure(self):
        tb = textwrap.dedent('''
            Traceback (most recent call last):
              File "C:/proj/backend/main.py", line 10, in f
                x
            RuntimeError: CUDA error: device-side assert triggered
        ''')
        assert classify(parse_traceback(tb)) is FailureClass.CUDA_SHAPE

    def test_unrecognised_signature_is_unknown(self):
        tb = (
            'Traceback (most recent call last):\n'
            '  File "C:/proj/backend/main.py", line 10, in f\n'
            '    x\n'
            'SystemExit: the flux capacitor melted\n'
        )
        assert classify(parse_traceback(tb)) is FailureClass.UNKNOWN

    def test_missing_module_is_detected(self):
        tb = (
            'Traceback (most recent call last):\n'
            '  File "C:/proj/backend/main.py", line 10, in f\n'
            '    x\n'
            "ModuleNotFoundError: No module named 'h5py'\n"
        )
        assert classify(parse_traceback(tb)) is FailureClass.MISSING_MODULE


# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------


class TestStringPatch:
    def test_replaces_the_matching_line_and_preserves_indent(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def f():\n    x = 1\n    return x\n", encoding="utf-8")
        plan = PatchPlan(target=str(f), kind="string", find="x = 1", replace="x = 2")
        result = apply_patch(plan)
        assert result.ok and result.applied
        assert "    x = 2" in f.read_text(encoding="utf-8")

    def test_backup_is_written_and_undo_restores(self, tmp_path):
        f = tmp_path / "m.py"
        original = "x = 1\n"
        f.write_text(original, encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="string",
                                       find="x = 1", replace="x = 2"))
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()
        assert f.read_text(encoding="utf-8") == "x = 2\n"
        assert undo(result) is True
        assert f.read_text(encoding="utf-8") == original

    def test_occurrence_index_selects_the_right_occurrence(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("v = 1\nv = 1\nv = 1\n", encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="string",
                                       find="v = 1", replace="v = 9", occurrence=2))
        assert result.ok and result.applied
        assert f.read_text(encoding="utf-8") == "v = 1\nv = 9\nv = 1\n"

    def test_ambiguous_substring_anchor_is_refused(self, tmp_path):
        """A substring matching many lines is more dangerous than a failure."""
        f = tmp_path / "m.py"
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="string",
                                       find="a", replace="X"))
        assert not result.ok
        assert "anchor not found" in result.error

    def test_syntax_error_producing_patch_is_rejected_before_write(self, tmp_path):
        f = tmp_path / "m.py"
        original = "x = 1\n"
        f.write_text(original, encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="string",
                                       find="x = 1", replace="x = ((("))
        assert not result.ok
        assert "SyntaxError" in result.error
        assert f.read_text(encoding="utf-8") == original   # disk untouched

    def test_no_op_replacement_is_reported_but_successful(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="string",
                                       find="x = 1", replace="x = 1"))
        assert result.ok
        assert not result.applied

    def test_out_of_range_occurrence_is_refused(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="string",
                                       find="x = 1", replace="y", occurrence=5))
        assert not result.ok
        assert "out of range" in result.error

    def test_missing_target_is_reported(self, tmp_path):
        result = apply_patch(PatchPlan(target=str(tmp_path / "nope.py"), kind="string",
                                       find="x", replace="y"))
        assert not result.ok
        assert "target not found" in result.error

    def test_multiline_replacement_is_supported(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def f():\n    pass\n", encoding="utf-8")
        result = apply_patch(PatchPlan(
            target=str(f), kind="string", find="pass",
            replace="a = 1\n    b = 2\n    return a + b",
        ))
        assert result.ok and result.applied
        ok, msg = compile_check(f)
        assert ok, msg

    def test_backup_refuses_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            backup_file(tmp_path / "absent.py")


class TestAstPatch:
    def test_wrap_return_expression(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text(
            "import torch\n"
            "def loss(x):\n"
            "    return x.pow(2).mean()\n",
            encoding="utf-8",
        )
        result = apply_patch(PatchPlan(
            target=str(f), kind="ast", ast_op="wrap_return_expression",
            ast_args={"method": "loss", "wrap": "torch.nan_to_num"},
        ))
        assert result.ok and result.applied
        src = f.read_text(encoding="utf-8")
        assert "nan_to_num" in src
        ok, _ = compile_check(f)
        assert ok

    def test_unknown_op_is_refused(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = apply_patch(PatchPlan(target=str(f), kind="ast", ast_op="rm_rf"))
        assert not result.ok
        assert "unknown ast_op" in result.error

    def test_rename_function_updates_call_sites(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def old(a):\n    return a\n\ndef caller():\n    return old(1)\n",
                     encoding="utf-8")
        result = apply_patch(PatchPlan(
            target=str(f), kind="ast", ast_op="rename_function",
            ast_args={"old": "old", "new": "renamed"},
        ))
        assert result.ok and result.applied
        src = f.read_text(encoding="utf-8")
        assert "def renamed" in src
        assert "old(" not in src


class TestConfigPatch:
    def test_config_patch_sets_environment_only(self, tmp_path):
        import os

        result = apply_patch(PatchPlan(
            target=str(tmp_path / "unused.py"), kind="config",
            ast_args={"BTMS_TEST_KEY": "value"},
        ))
        assert result.ok and result.applied
        assert os.environ.get("BTMS_TEST_KEY") == "value"
        os.environ.pop("BTMS_TEST_KEY", None)

    def test_undo_of_config_patch_returns_false(self, tmp_path):
        result = apply_patch(PatchPlan(
            target=str(tmp_path / "unused.py"), kind="config",
            ast_args={"BTMS_TEST_KEY2": "v"},
        ))
        assert undo(result) is False


class TestCompileSweep:
    def test_detects_a_broken_file(self, tmp_path):
        (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
        broken = syntax_all(tmp_path)
        assert any("bad.py" in b for b in broken)

    def test_clean_tree_passes(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        assert syntax_all(tmp_path) == []

    def test_import_check_runs_in_a_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pkg = tmp_path / "pkgmod"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
        ok, out = __import__("auto_heal.patching", fromlist=["x"]).import_check(
            pkg / "__init__.py"
        )
        assert ok, out
        assert "HEALTH_IMPORT_OK" in out

    def test_import_check_fails_on_a_broken_module(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pkg = tmp_path / "brokenmod"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        ok, out = __import__("auto_heal.patching", fromlist=["x"]).import_check(
            pkg / "__init__.py"
        )
        assert not ok
        assert "boom" in out


# ---------------------------------------------------------------------------
# Hypothesiser
# ---------------------------------------------------------------------------


class TestRuleBasedHypothesiser:
    @pytest.mark.parametrize("tb,expected", [
        (SHAPE_TB, FailureClass.CUDA_SHAPE),
        (DEVICE_TB, FailureClass.DEVICE_MISMATCH),
        (NAN_TB, FailureClass.NAN_PDE_LOSS),
        (DATASET_TB, FailureClass.MISSING_DATASET),
        (PORT_TB, FailureClass.PORT_IN_USE),
        (VALIDATION_TB, FailureClass.VALIDATION),
    ])
    def test_every_class_yields_a_hypothesis(self, tb, expected):
        h = RuleBasedHypothesiser(cwd=".").hypothesise(
            parse_traceback(tb), classify(parse_traceback(tb))
        )
        assert h.failure_class is expected
        assert h.explanation
        assert h.confidence >= 0.0

    def test_validation_failure_refuses_to_patch(self):
        """A client error must never mutate the model."""
        info = parse_traceback(VALIDATION_TB)
        h = RuleBasedHypothesiser(cwd=".").hypothesise(
            info, classify(info)
        )
        assert h.plan.kind == "none"
        assert h.should_apply is False

    def test_unknown_class_declines_to_patch(self):
        tb = (
            'Traceback (most recent call last):\n'
            '  File "C:/proj/backend/main.py", line 3, in f\n'
            '    x\n'
            'SystemExit: unusual\n'
        )
        info = parse_traceback(tb)
        h = RuleBasedHypothesiser(cwd=".").hypothesise(info, FailureClass.UNKNOWN)
        assert h.plan.kind == "none"
        assert "human" in h.explanation.lower()

    def test_confidence_gate_blocks_low_confidence_patches(self):
        hyp = Hypothesis(
            failure_class=FailureClass.CUDA_SHAPE,
            explanation="weak guess",
            plan=PatchPlan(target="x.py", kind="string", find="a", replace="b"),
            confidence=MIN_APPLY_CONFIDENCE - 0.01,
            source="rules",
        )
        assert hyp.actionable is True
        assert hyp.should_apply is False

    def test_port_in_use_uses_a_config_patch(self):
        info = parse_traceback(PORT_TB)
        h = RuleBasedHypothesiser(cwd=".").hypothesise(info, FailureClass.PORT_IN_USE)
        assert h.plan.kind == "config"
        assert h.plan.ast_args["API_PORT"] == "8000"

    def test_alternatives_are_recorded(self):
        info = parse_traceback(SHAPE_TB)
        h = RuleBasedHypothesiser(cwd=".").hypothesise(info, FailureClass.CUDA_SHAPE)
        assert len(h.alternatives) >= 1


class TestLLMHypothesiserValidation:
    """The trust boundary: a hallucinated patch must degrade to no action."""

    def _h(self, payload, tmp_path):
        hypothesiser = LLMHypothesiser(cwd=str(tmp_path))
        info = TracebackInfo(raw="", exception_type="RuntimeError", message="x",
                             failing_file=str(tmp_path / "backend" / "x.py"),
                             failing_line=1, failing_source="x")
        return hypothesiser._validate(info, FailureClass.CUDA_SHAPE, payload)

    def test_path_traversal_is_rejected(self, tmp_path):
        h = self._h({"kind": "string", "target": "../../etc/passwd.py",
                     "find": "a", "replace": "b", "confidence": 0.99}, tmp_path)
        assert h.plan.kind == "none"
        assert h.confidence == 0.0

    def test_outside_prefix_is_rejected(self, tmp_path):
        secret = tmp_path / "secret.py"
        secret.write_text("a = 1\n", encoding="utf-8")
        h = self._h({"kind": "string", "target": "secret.py",
                     "find": "a = 1", "replace": "a = 2", "confidence": 0.99}, tmp_path)
        assert h.plan.kind == "none"

    def test_nonexistent_target_is_rejected(self, tmp_path):
        h = self._h({"kind": "string", "target": "backend/ghost.py",
                     "find": "a", "replace": "b", "confidence": 0.99}, tmp_path)
        assert h.plan.kind == "none"

    def test_missing_anchor_is_rejected(self, tmp_path):
        backend = tmp_path / "backend"
        backend.mkdir()
        f = backend / "real.py"
        f.write_text("actual_line = 1\n", encoding="utf-8")
        h = self._h({"kind": "string", "target": "backend/real.py",
                     "find": "hallucinated", "replace": "x", "confidence": 0.99},
                    tmp_path)
        assert h.plan.kind == "none"

    def test_unknown_ast_op_is_rejected(self, tmp_path):
        backend = tmp_path / "backend"
        backend.mkdir()
        (backend / "real.py").write_text("x = 1\n", encoding="utf-8")
        h = self._h({"kind": "ast", "target": "backend/real.py",
                     "ast_op": "delete_everything", "confidence": 0.99}, tmp_path)
        assert h.plan.kind == "none"

    def test_valid_patch_is_accepted(self, tmp_path):
        backend = tmp_path / "backend"
        backend.mkdir()
        (backend / "real.py").write_text("a = 1\n", encoding="utf-8")
        h = self._h({"kind": "string", "target": "backend/real.py",
                     "find": "a = 1", "replace": "a = 2", "confidence": 0.8,
                     "reason": "fix"}, tmp_path)
        assert h.plan.kind == "string"
        assert h.plan.find == "a = 1"
        assert h.should_apply is True

    def test_non_python_target_is_rejected(self, tmp_path):
        backend = tmp_path / "backend"
        backend.mkdir()
        (backend / "data.mat").write_bytes(b"\x00")
        h = self._h({"kind": "string", "target": "backend/data.mat",
                     "find": "a", "replace": "b", "confidence": 0.99}, tmp_path)
        assert h.plan.kind == "none"

    def test_no_kind_is_diagnostic_only(self, tmp_path):
        h = self._h({"kind": "none", "explanation": "cannot fix"}, tmp_path)
        assert h.plan.kind == "none"
        assert h.confidence == 0.0

    def test_availability_is_false_without_a_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert LLMHypothesiser(cwd=".").available is False


    @pytest.mark.skipif(
        not __import__("auto_heal.graph", fromlist=["LANGGRAPH_AVAILABLE"])
        .LANGGRAPH_AVAILABLE,
        reason="langgraph is not installed",
    )
    def test_package_and_script_do_not_shadow_each_other(self):
        """The repo carries both ``auto_heal/`` and ``auto_heal.py``.

        Python's import rules give the package precedence, so ``import
        auto_heal`` binds to the directory and the script is unreachable by
        name. Anything that needs the CLI must therefore load it by *path* --
        which is what ``auto_heal/__main__.py`` does, making ``python -m
        auto_heal`` work. Without that shim, ``-m auto_heal`` would silently
        import the package and do nothing at all: no error, no monitor.
        """
        repo = Path(__file__).resolve().parents[2]
        script = repo / "auto_heal.py"
        package = repo / "auto_heal"
        assert script.is_file(), f"expected the CLI script at {script}"
        assert package.is_dir(), f"expected the package at {package}"

        main_py = package / "__main__.py"
        assert main_py.is_file(), (
            "auto_heal/__main__.py is missing. `python -m auto_heal` would "
            "import the package and exit without running the monitor, silently."
        )
        text = main_py.read_text(encoding="utf-8")
        assert "runpy.run_path" in text, "the shim must load the CLI by path"

        # And the imported package must not accidentally be the script.
        import auto_heal
        assert Path(auto_heal.__file__).parent == package

    def test_cli_parser_exposes_the_documented_flags(self):
        """The flags SETUP.md tells a presenter to type must exist.

        A typo in a flag name surfaces at demo time as an argparse traceback in
        front of an audience, which is the worst possible moment to discover
        it.
        """
        import subprocess
        import sys

        repo = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            [sys.executable, str(repo / "auto_heal.py"), "--help"],
            capture_output=True, text=True, cwd=str(repo), timeout=120,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        for flag in ("--inject", "--dry-run", "--max-attempts",
                     "--graph-diagram", "--startup-timeout", "--prefer-llm"):
            assert flag in out.stdout, f"{flag} missing from --help"
        for choice in ("cuda_shape", "nan_loss", "dtype", "dataset",
                       "module", "kill"):
            assert choice in out.stdout, f"injection {choice!r} missing"


def test_patchable_prefixes_are_scoped_to_the_repo():
    """The agent must never be allowed to patch site-packages."""
    assert all(p.startswith(("backend/", "auto_heal/", "scripts/"))
               for p in PATCHABLE_PREFIXES)
    assert not any("site-packages" in p or "/" not in p for p in PATCHABLE_PREFIXES)


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


class TestGraphWiring:
    """The state machine is wired correctly -- not just each node in isolation.

    The nodes below can all be unit-correct and the graph can still be wrong,
    because LangGraph merges a node's returned dict into the state
    *field-by-field* and silently discards keys that are absent from the state
    TypedDict. ``node_monitor`` returning ``failed=True`` therefore did nothing
    at all: ``route_after_monitor`` read ``state.get('failed')`` as ``None``,
    took the falsy branch straight to ``report``, and the monitor declared the
    service healed without ever classifying a single crash. A node-level test
    cannot see that, which is what makes a graph-level test the only place it
    can be caught.
    """

    @staticmethod
    def _route(st: dict) -> Optional[str]:
        """Run one conditional edge against a partial state."""
        import auto_heal.graph as g
        return g.route_after_monitor(st)

    def test_failed_false_reports_success(self):
        assert self._route({"failed": False}) == "report"

    def test_failed_true_classifies(self):
        assert self._route({"failed": True}) == "classify"

    def test_absent_failed_is_treated_as_failure_not_success(self):
        """The one routing decision the old bug inverted.

        An unset key must not silently mean "healthy" -- that is the direction
        that turns a crashed service into a reported success. If LangGraph ever
        stops merging channel defaults, this test fails.
        """
        assert self._route({}) == "classify"

    @pytest.mark.skipif(
        not __import__("auto_heal.graph", fromlist=["LANGGRAPH_AVAILABLE"])
        .LANGGRAPH_AVAILABLE,
        reason="langgraph is not installed",
    )
    def test_state_keys_are_read_written_and_declared_consistently(self, tmp_path):
        """All three directions, checked at once.

        LangGraph merges a node's returned dict into the state field-by-field
        and silently discards keys absent from the ``TypedDict``. Two bugs in
        this repo came from that, and a node-level test cannot see either:

        * ``failed`` was returned by ``node_monitor`` but never declared, so the
          conditional edge read ``None``, took the falsy branch and reported a
          crashed service as **healed** without classifying anything.
        * ``patch_occurrence`` was read by ``node_patch`` but never written, so
          every patch silently used the default occurrence 1 regardless of what
          the hypothesiser asked for.
        * ``raw_traceback`` was written by ``node_classify`` but never declared,
          so the structured diagnosis survived while the incident report quietly
          lost the verbatim traceback.

        This test walks the node source's AST and asserts all three directions
        hold, so any fourth instance of the same class fails here rather than in
        front of a reviewer.
        """
        import ast
        from auto_heal.graph import HealGraph
        from auto_heal.nodes import Runtime, set_runtime

        rt = Runtime(cwd=".", command=[sys.executable, "-c", "pass"],
                     log_dir=str(tmp_path / "logs"), echo=False)
        set_runtime(rt)

        nodes_py = Path(__file__).resolve().parents[2] / "auto_heal" / "nodes.py"
        tree = ast.parse(nodes_py.read_text(encoding="utf-8"), filename=str(nodes_py))

        reads: set = set()
        writes: set = set()
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("node_")]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "state"
                        and node.args and isinstance(node.args[0], ast.Constant)):
                    reads.add(node.args[0].value)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant):
                            writes.add(k.value)

        # Keys the CLI seeds in initial_state() rather than a node.
        writes |= {"command", "cwd", "log_path", "max_attempts", "dry_run",
                   "attempt", "failed", "traceback", "failure_class",
                   "classified", "server_stdout", "server_stderr",
                   "server_exit_code", "status", "history", "started_at",
                   "finished_at", "report_path"}

        from auto_heal import graph_state as gs
        declared = set(gs.HealState.__annotations__)

        orphans = reads - writes
        assert not orphans, (
            f"state keys read but never written: {sorted(orphans)}. Every reader "
            f"sees its own default, so the value is whatever the node guessed "
            f"rather than what an upstream node decided."
        )

        undeclared_writes = writes - declared
        assert not undeclared_writes, (
            f"state keys written but not declared in HealState: "
            f"{sorted(undeclared_writes)}. LangGraph drops them on the merge, "
            f"so any downstream reader sees a default instead of this value."
        )

        # Dead fields are worth a NOTE rather than a failure: they may be
        # written by a node this AST walk does not see. But they should not
        # accumulate silently either.
        dead = declared - reads - writes
        assert not dead, (
            f"state fields declared but never read or written: {sorted(dead)}. "
            f"They are documentation of behaviour that does not exist."
        )

    @pytest.mark.skipif(
        not __import__("auto_heal.graph", fromlist=["LANGGRAPH_AVAILABLE"])
        .LANGGRAPH_AVAILABLE,
        reason="langgraph is not installed",
    )
    def test_state_declares_every_key_the_nodes_return(self, tmp_path):
        """Every key returned by each node is a declared state field.

        This is the runtime complement to the AST-based test above: it calls
        each node and inspects what actually comes back, catching a key that is
        declared but whose *value* never survives the merge.
        """
        import tempfile
        from auto_heal.graph import HealGraph
        from auto_heal.nodes import Runtime, set_runtime

        rt = Runtime(cwd=".", command=[sys.executable, "-c", "pass"],
                     log_dir=str(tmp_path / "logs"), echo=False)
        set_runtime(rt)

        from auto_heal.nodes import (
            node_classify,
            node_monitor,
            node_patch,
            node_report,
            node_restart,
            node_rollback,
            node_verify,
        )
        from auto_heal import graph_state as gs

        declared = set(gs.HealState.__annotations__)
        # Each node returns only the keys it intends to update, so a returned
        # key absent from the declaration is a bug by construction: LangGraph
        # drops it without a warning and every reader sees the default.
        for fn in (node_monitor, node_classify, node_patch, node_verify,
                   node_rollback, node_restart, node_report):
            for key in fn({}).keys():
                assert key in declared, (
                    f"{fn.__name__} returns undeclared key {key!r}; LangGraph "
                    f"will drop it and any conditional edge reading it will "
                    f"see the default"
                )

        rt = Runtime(cwd=".", command=[sys.executable, "-c", "pass"],
                     log_dir=str(tmp_path / "logs"), echo=False)
        set_runtime(rt)

        from auto_heal.nodes import (
            node_classify,
            node_monitor,
            node_patch,
            node_report,
            node_restart,
            node_rollback,
            node_verify,
        )
        from auto_heal import graph_state as gs

        declared = set(gs.HealState.__annotations__)
        # Each node returns only the keys it intends to update, so a returned
        # key absent from the declaration is a bug by construction: LangGraph
        # drops it without a warning and every reader sees the default.
        for fn in (node_monitor, node_classify, node_patch, node_verify,
                   node_rollback, node_restart, node_report):
            for key in fn({}).keys():
                assert key in declared, (
                    f"{fn.__name__} returns undeclared key {key!r}; LangGraph "
                    f"will drop it and any conditional edge reading it will "
                    f"see the default"
                )

    @pytest.mark.skipif(
        not __import__("auto_heal.graph", fromlist=["LANGGRAPH_AVAILABLE"])
        .LANGGRAPH_AVAILABLE,
        reason="langgraph is not installed",
    )
    def test_initial_state_declares_the_failed_flag(self):
        import tempfile
        from auto_heal.graph import HealGraph
        from auto_heal.nodes import Runtime

        rt = Runtime(cwd=".", command=[sys.executable, "-c", "pass"],
                     log_dir=tempfile.mkdtemp(), echo=False)
        graph = HealGraph(rt, checkpointer=False)
        state = graph.initial_state([sys.executable, "-c", "pass"], ".", "x.log")
        assert "failed" in state
        assert state["failed"] is True
        assert state["attempt"] == 0
        assert state["status"] == "pending"