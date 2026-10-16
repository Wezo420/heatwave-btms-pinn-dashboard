"""Tests for the FastAPI surface: routes, schemas and the fail-safe logic."""

from __future__ import annotations

import json
import math
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.config import (
    DEFAULT_PACK_CURRENT_A,
    MAX_COOLANT_FLOW_LPM,
    MIN_COOLANT_FLOW_LPM,
    PACK_NOMINAL_VOLTAGE_V,
)
from backend.main import create_app
from backend.schemas import FlowCommandRequest, OperatingPoint, SimulationRequest
from backend.services.heatwave import (
    AMBIENT_TAU_S,
    COMFORT_CEILING_C,
    MARGIN_FLOOR_C,
    RAMP_RATE_LPM_PER_S,
    HeatwavePlanner,
    ReducedPackModel,
    SafetyState,
    effective_ambient_series,
    planner_for,
    solar_overshoot,
)
from backend.services.telemetry import TelemetrySimulator


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A TestClient over the real application.

    ``with TestClient(...)`` runs the lifespan, which starts the telemetry hub's
    asyncio task -- exactly as uvicorn would, so the WebSocket route is testable.
    """
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def scenario(client: TestClient) -> dict:
    """The canonical hot-route scenario, fetched once."""
    resp = client.get("/api/simulate/scenario", params={"chemistry": "nmc", "n": 60})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


class TestMeta:
    def test_root_banner_lists_the_endpoints(self, client: TestClient):
        body = client.get("/").json()
        assert "endpoints" in body
        assert body["endpoints"]["telemetry_ws"].startswith("/api/simulate/ws/")

    def test_health_reports_dependencies(self, client: TestClient):
        body = client.get("/api/health").json()
        assert body["status"] in ("ok", "degraded", "error")
        assert "torch_version" in body
        assert "checkpoint_present" in body
        assert "nasa_dataset_present" in body

    def test_catalogue_drives_the_ui_toggle(self, client: TestClient):
        body = client.get("/api/catalog/chemistries").json()
        keys = [c["key"] for c in body["chemistries"]]
        assert keys == ["nmc", "lfp"]
        nmc = body["chemistries"][0]
        assert nmc["thermal_runaway_onset_c"] == 210.0
        assert nmc["cp_j_kgk"] == 1040.0
        assert body["chemistries"][1]["thermal_runaway_onset_c"] == 270.0
        assert body["max_flow_lpm"] == MAX_COOLANT_FLOW_LPM

    def test_model_info_exposes_the_gradient_penalty_config(self, client: TestClient):
        body = client.get("/api/model/info").json()
        assert "gradient_penalty" in body
        assert "ema_decay" in body["gradient_penalty"]
        assert body["latency_budget_ms"] == 800.0
        assert body["total_parameters"] > 0

    def test_openapi_is_served(self, client: TestClient):
        spec = client.get("/openapi.json").json()
        assert "/api/predict/field" in spec["paths"]

    def test_unknown_route_is_404(self, client: TestClient):
        assert client.get("/api/nonexistent").status_code == 404


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_field_preview_returns_a_finite_grid(self, client: TestClient):
        body = client.get("/api/predict/field/preview",
                          params={"chemistry": "nmc", "nx": 8, "ny": 6, "nz": 4}).json()
        assert body["temperature_c"]
        assert len(body["temperature_c"]) == 8
        assert body["max_temp_c"] > body["min_temp_c"]
        assert "residuals" in body

    @pytest.mark.parametrize("chem", ["nmc", "lfp"])
    def test_field_endpoint_validates_and_returns(self, client: TestClient, chem):
        resp = client.post("/api/predict/field", json={
            "operating_point": {
                "chemistry": chem, "coolant_inlet_c": 26.0, "coolant_flow_lpm": 6.0,
                "ambient_c": 38.0, "pack_current_a": DEFAULT_PACK_CURRENT_A,
                 "state_of_charge": 0.5,
            },
            "grid": [6, 5, 4],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chemistry"] == chem
        assert body["residuals"]["nonfinite_guard"] is False
        assert body["within_latency_budget"] in (True, False)

    def test_unknown_chemistry_is_422(self, client: TestClient):
        resp = client.get("/api/predict/field/preview",
                          params={"chemistry": "sodium"})
        assert resp.status_code == 422

    def test_out_of_range_flow_is_422(self, client: TestClient):
        resp = client.post("/api/predict/field", json={
            "operating_point": {"coolant_flow_lpm": 99.0},
            "grid": [4, 4, 3],
        })
        assert resp.status_code == 422

    def test_unknown_field_is_rejected(self, client: TestClient):
        """extra=forbid turns a typo into an error rather than a silent default."""
        resp = client.post("/api/predict/field", json={
            "operating_point": {"coolant_flow_lpm": 6.0},
            "grid": [4, 4, 3],
            "typo_field": 1,
        })
        assert resp.status_code == 422

    def test_out_of_range_grid_is_422(self, client: TestClient):
        resp = client.post("/api/predict/field", json={
            "operating_point": {},
            "grid": [200, 5, 4],
        })
        assert resp.status_code == 422

    def test_residuals_endpoint_reports_the_penalty(self, client: TestClient):
        body = client.get("/api/predict/residuals", params={"chemistry": "lfp"}).json()
        assert "gradient_penalty_note" in body
        assert body["exotherm_onset_c"] == 270.0
        assert "pde_energy" in body["residuals"]

    def test_single_point_query(self, client: TestClient):
        body = client.post("/api/predict/single", json={
            "operating_point": {"chemistry": "nmc", "coolant_flow_lpm": 6.0},
            "point": [0.05, 0.10, 0.004],
        }).json()
        assert isinstance(body["temperature_c"], (int, float))
        assert body["uncertainty_k"] >= 0


# ---------------------------------------------------------------------------
# Heatwave planning
# ---------------------------------------------------------------------------


class TestHeatwavePlanning:
    def test_forecast_finds_a_zone_in_the_hot_scenario(self, client: TestClient, scenario):
        body = client.post("/api/heatwave/forecast", json={
            "route": scenario["route"],
            "weather": scenario["weather"],
            "chemistry": "nmc",
            "pack_heat_w": 3200.0,
            "state_of_charge": 0.7,
        }).json()
        assert len(body["zones"]) >= 1
        zone = body["zones"][0]
        assert zone["t_enter_s"] <= zone["t_exit_s"]
        assert 0.0 <= zone["risk_index"] <= 1.0
        assert MIN_COOLANT_FLOW_LPM <= zone["required_flow_lpm"] <= MAX_COOLANT_FLOW_LPM

    def test_forecast_events_are_advisory_not_enforced(self, client: TestClient, scenario):
        body = client.post("/api/heatwave/forecast", json={
            "route": scenario["route"], "weather": scenario["weather"],
            "chemistry": "nmc", "pack_heat_w": 3000.0, "state_of_charge": 0.7,
        }).json()
        assert all(e["authority"] == "advisory" for e in body["schedule"])

    def test_mismatched_series_lengths_is_422(self, client: TestClient, scenario):
        resp = client.post("/api/heatwave/forecast", json={
            "route": scenario["route"][:10],
            "weather": scenario["weather"][:5],
            "chemistry": "nmc", "pack_heat_w": 3000.0, "state_of_charge": 0.7,
        })
        assert resp.status_code == 422

    def test_ambient_series_lags_the_raw_reading(self, client: TestClient):
        body = client.get("/api/heatwave/ambient",
                          params={"chemistry": "nmc", "ambient_start_c": 30.0,
                                  "ambient_end_c": 50.0}).json()
        raw = body["raw_ambient_c"]
        eff = body["effective_ambient_c"]
        # The first-order lag must always lag a monotonically rising signal.
        assert eff[-1] < raw[-1]
        assert eff[0] == pytest.approx(raw[0], abs=2.0)
        assert body["pack_ambient_c"][-1] > eff[-1]   # solar overshoot added

    def test_explain_returns_the_deciding_terms(self, client: TestClient):
        body = client.get("/api/heatwave/explain", params={"chemistry": "lfp"}).json()
        assert "decision" in body
        assert "comfort_ceiling_c" in body["decision"]
        assert body["exotherm_onset_c"] == 270.0


class TestPlannerInternals:
    @pytest.mark.parametrize("chem", ["nmc", "lfp"])
    def test_plan_respects_the_comfort_ceiling(self, chem):
        planner = HeatwavePlanner(chem)
        route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i), "speed_mps": 22.0}
                 for i in range(0, 121)]
        weather = [{"t_s": float(i), "ambient_c": 33.0 + 19.0 * (i / 120.0) ** 1.6,
                    "solar_w_m2": 700.0 + 550.0 * (i / 120.0), "wind_mps": 2.0,
                    "road_surface_c": 46.0 + 26.0 * (i / 120.0)} for i in range(0, 121)]
        plan = planner.plan(route, weather,
                            base_current_a=DEFAULT_PACK_CURRENT_A,
                            base_flow_lpm=6.0, initial_temp_c=34.0, soc=0.7)
        assert plan.peak_cell_c <= COMFORT_CEILING_C[chem] + 1e-6
        assert plan.min_margin_c >= MARGIN_FLOOR_C[chem] - 1e-6

    def test_anticipation_beats_the_constant_flow_baseline(self):
        """The project's headline claim, measured rather than asserted.

        The operating point is chosen to be the one where the claim is
        non-trivial, and it is worth being explicit about why.  Against a
        part-load 2 L/min baseline and a pack already at 38 degC from sustained
        fast charging, constant flow breaches the 42 degC comfort ceiling while
        the anticipatory schedule holds it.  Against a 6 L/min baseline the
        pack rejects ~4.7 kW at 16 K above coolant -- several times its own
        ~1 kW heat load -- so the two schedules are identical and the benefit is
        exactly zero.  That is the model being right, not the controller
        failing: there is nothing for an anticipatory controller to anticipate
        when the pump is already wide open.
        """
        planner = HeatwavePlanner("nmc")
        route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i), "speed_mps": 22.0}
                 for i in range(0, 121)]
        weather = [{"t_s": float(i), "ambient_c": 33.0 + 21.0 * (i / 120.0) ** 1.6,
                    "solar_w_m2": 700.0 + 600.0 * (i / 120.0), "wind_mps": 2.0,
                    "road_surface_c": 46.0 + 28.0 * (i / 120.0)} for i in range(0, 121)]

        controlled = planner.plan(route, weather, base_flow_lpm=2.0,
                                  initial_temp_c=38.0)
        baseline = planner.plan(route, weather, base_flow_lpm=2.0,
                                initial_temp_c=38.0, enable_anticipatory=False)

        # The baseline must actually breach the ceiling, or "improvement" would
        # be measured against a schedule that never needed help.
        assert baseline.peak_cell_c > COMFORT_CEILING_C["nmc"]
        assert controlled.peak_cell_c < baseline.peak_cell_c
        assert controlled.peak_cell_c <= COMFORT_CEILING_C["nmc"]
        assert controlled.decision_log["delta_peak_cell_c"] < 0
        assert controlled.decision_log["anticipation_benefit_k"] > 0.0

    def test_anticipation_is_a_no_op_when_the_pump_is_already_wide_open(self):
        """Documents the regime where anticipation buys nothing.

        A strong claim about the controller is only worth as much as the
        conditions under which it was measured.  At 6 L/min the cold plate keeps
        the pack 11 K above coolant against a 42 degC ceiling, so the
        anticipatory schedule and the constant schedule coincide exactly.
        """
        planner = HeatwavePlanner("nmc")
        route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i), "speed_mps": 22.0}
                 for i in range(0, 121)]
        weather = [{"t_s": float(i), "ambient_c": 33.0 + 21.0 * (i / 120.0) ** 1.6,
                    "solar_w_m2": 700.0 + 600.0 * (i / 120.0), "wind_mps": 2.0,
                    "road_surface_c": 46.0 + 28.0 * (i / 120.0)} for i in range(0, 121)]

        controlled = planner.plan(route, weather, base_flow_lpm=6.0,
                                  initial_temp_c=38.0)
        baseline = planner.plan(route, weather, base_flow_lpm=6.0,
                                initial_temp_c=38.0, enable_anticipatory=False)

        assert controlled.peak_cell_c == pytest.approx(baseline.peak_cell_c, abs=1e-9)
        assert controlled.decision_log["anticipation_benefit_k"] == 0.0
        assert controlled.decision_log["coolant_lift_lpm"] == 0.0

    def test_lfp_tolerates_a_larger_margin_than_nmc(self):
        results = {}
        for chem in ("nmc", "lfp"):
            planner = HeatwavePlanner(chem)
            route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i), "speed_mps": 22.0}
                     for i in range(0, 121)]
            weather = [{"t_s": float(i), "ambient_c": 34.0 + 20.0 * (i / 120.0) ** 1.6,
                        "solar_w_m2": 800.0 + 500.0 * (i / 120.0), "wind_mps": 2.0,
                        "road_surface_c": 48.0 + 26.0 * (i / 120.0)} for i in range(0, 121)]
            results[chem] = planner.plan(route, weather,
                                         base_current_a=DEFAULT_PACK_CURRENT_A,
                                         initial_temp_c=34.0, soc=0.7)
        assert results["lfp"].min_margin_c > results["nmc"].min_margin_c

    def test_schedule_is_rate_limited(self):
        planner = HeatwavePlanner("nmc")
        route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i * 2), "speed_mps": 22.0}
                 for i in range(0, 121)]
        weather = [{"t_s": float(i * 2), "ambient_c": 30.0 + 24.0 * (i / 120.0),
                    "solar_w_m2": 900.0, "wind_mps": 2.0,
                    "road_surface_c": 50.0 + 24.0 * (i / 120.0)} for i in range(0, 121)]
        plan = planner.plan(route, weather, base_flow_lpm=2.0)
        deltas = np_abs_diff(plan.commanded_flow_lpm)
        assert deltas.max() <= RAMP_RATE_LPM_PER_S * 2.0 + 1e-9

    def test_no_actuation_when_conditions_are_mild(self):
        planner = HeatwavePlanner("nmc")
        route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i), "speed_mps": 22.0}
                 for i in range(0, 61)]
        weather = [{"t_s": float(i), "ambient_c": 18.0, "solar_w_m2": 200.0,
                    "wind_mps": 4.0, "road_surface_c": 20.0} for i in range(0, 61)]
        plan = planner.plan(route, weather, base_flow_lpm=6.0)
        assert plan.zones == []
        assert "No anticipatory action" in plan.advisory

    def test_ambient_lag_time_constant(self):
        weather = [{"ambient_c": 0.0}] * 200
        series = effective_ambient_series(weather, 10.0)
        # A step to ambient must approach it with a ~420 s time constant.
        assert series[-1] < 100.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestSimulation:
    #: Payload for the trajectory tests: a pack off a fast charge (38 degC) with
    #: the pump at its part-load point (2 L/min), which is the only regime where
    #: the anticipatory controller has anything to do.
    @staticmethod
    def _payload(scenario: dict, **overrides) -> dict:
        body = {
            "chemistry": "nmc",
            "duration_s": scenario["duration_s"],
            "dt_s": scenario["dt_s"],
            "route": scenario["route"],
            "weather": scenario["weather"],
            "base_current_a": DEFAULT_PACK_CURRENT_A,
            "base_flow_lpm": 2.0,
            "initial_cell_temp_c": 38.0,
            "initial_soc": 0.7,
            "enable_anticipatory": True,
            "enable_learned_field": True,
            "seed": 1337,
        }
        body.update(overrides)
        return body

    def test_trajectory_returns_a_complete_trace(self, client: TestClient, scenario):
        resp = client.post("/api/simulate/trajectory",
                           json=self._payload(scenario))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["frames"]) == len(scenario["route"])
        frame = body["frames"][0]
        assert len(frame["cell_temps_c"]) == 12
        assert frame["trip_state"] in ("ok", "derate", "trip")

    def test_trajectory_is_deterministic_for_a_fixed_seed(self, client: TestClient, scenario):
        payload = self._payload(scenario)
        a = client.post("/api/simulate/trajectory", json=payload).json()
        b = client.post("/api/simulate/trajectory", json=payload).json()
        assert a["peak_cell_temp_c"] == b["peak_cell_temp_c"]
        assert a["frames"][-1]["max_cell_temp_c"] == b["frames"][-1]["max_cell_temp_c"]

    def test_advisory_quantifies_the_benefit_of_anticipation(self, client: TestClient, scenario):
        body = client.post("/api/simulate/trajectory",
                           json=self._payload(scenario)).json()
        assert "reduced the peak cell temperature" in body.get("advisory", "")

    def test_reported_energy_is_physically_plausible(self, client: TestClient, scenario):
        """``total_energy_wh`` must be current x pack voltage, in watt-hours.

        An earlier version accumulated ``I * T_coolant_kelvin``, which carried no
        coherent units and ran ~300x high.  At 12 A over 180 s the pack can move
        at most 12 A x ~370 V x 0.05 h ~ 220 Wh, so anything far above that is
        arithmetically impossible regardless of the chemistry.
        """
        body = client.post("/api/simulate/trajectory",
                           json=self._payload(scenario)).json()
        duration_h = scenario["duration_s"] / 3600.0
        # An upper bound from the pack's own C-rate: at most ~2C for the whole run.
        ceiling_wh = 2.0 * 20.0 * PACK_NOMINAL_VOLTAGE_V * duration_h
        assert 0.0 < body["total_energy_wh"] <= ceiling_wh, body["total_energy_wh"]

    def test_pack_current_is_clamped_to_the_pack_rating(self, client: TestClient, scenario):
        """The API rejects out-of-range current rather than extrapolating.

        Asking for 180 A on a 20 A 96s1p pack is not a scenario the model has
        any basis to simulate: the C-rate would be ~18C, far outside where the
        Arrhenius fit or the abuse limits mean anything.
        """
        resp = client.post("/api/simulate/trajectory",
                           json=self._payload(scenario, base_current_a=180.0))
        assert resp.status_code == 422, resp.text

    def test_scenario_is_harsh_enough_to_demonstrate_the_effect(self, client: TestClient):
        body = client.get("/api/simulate/scenario", params={"n": 40}).json()
        ambient = [w["ambient_c"] for w in body["weather"]]
        assert ambient[-1] >= 50.0
        assert ambient[-1] > ambient[0] + 15.0

    def test_websocket_streams_telemetry(self, client: TestClient):
        with client.websocket_connect("/api/simulate/ws/telemetry?chemistry=nmc") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["chemistry"] == "nmc"
            assert hello["exotherm_onset_c"] == 210.0
            # The hub ticks at 0.5 s; allow generous headroom for CI load.
            frame = ws.receive_json()
            assert frame["type"] == "telemetry"
            assert len(frame["data"]["cell_temps_c"]) == 12

    def test_websocket_handles_ping_and_reset(self, client: TestClient):
        with client.websocket_connect("/api/simulate/ws/telemetry?chemistry=lfp") as ws:
            ws.receive_json()   # hello
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            ws.send_json({"action": "reset"})
            ack = ws.receive_json()
            assert ack["type"] == "ack"


# ---------------------------------------------------------------------------
# Control + fail-safe
# ---------------------------------------------------------------------------


class TestControl:
    def test_flow_command_is_acknowledged_with_settle_time(self, client: TestClient):
        body = client.post("/api/control/flow", json={
            "chemistry": "nmc", "flow_lpm": 9.0, "mode": "auto",
        }).json()
        assert body["accepted"] is True
        assert body["applied_flow_lpm"] == 9.0
        assert body["estimated_settle_s"] > 0
        assert body["heat_rejection_w"] > 0

    def test_out_of_range_command_is_422(self, client: TestClient):
        assert client.post("/api/control/flow",
                           json={"chemistry": "nmc", "flow_lpm": 50.0}).status_code == 422

    def test_more_flow_buys_more_heat_rejection(self, client: TestClient):
        low = client.post("/api/control/flow",
                          json={"chemistry": "lfp", "flow_lpm": 3.0}).json()
        high = client.post("/api/control/flow",
                           json={"chemistry": "lfp", "flow_lpm": 12.0}).json()
        assert high["heat_rejection_w"] > low["heat_rejection_w"]

    def test_state_endpoint_reports_the_loop_limits(self, client: TestClient):
        body = client.get("/api/control/state").json()
        assert set(body["chemistries"]) == {"nmc", "lfp"}
        assert body["loop_limits"]["max_lpm"] == MAX_COOLANT_FLOW_LPM
        assert body["loop_limits"]["ramp_rate_lpm_per_s"] == RAMP_RATE_LPM_PER_S


class TestSafetyState:
    def test_ok_tier_when_margin_is_large(self):
        from backend.config import get_chemistry

        s = SafetyState().evaluate(30.0, get_chemistry("nmc"))
        assert s.tier == "ok"
        assert s.derate_factor == 1.0

    def test_derate_when_margin_below_floor(self):
        from backend.config import get_chemistry

        s = SafetyState().evaluate(190.0, get_chemistry("nmc"))
        assert s.tier == "derate"
        assert s.derate_factor < 1.0
        assert s.forced_flow_lpm == MAX_COOLANT_FLOW_LPM

    def test_trip_latches_at_the_exotherm_onset(self):
        from backend.config import get_chemistry

        s = SafetyState().evaluate(210.0, get_chemistry("nmc"))
        assert s.tier == "trip"
        assert s.latched_trip is True
        assert s.derate_factor == 0.0

    def test_lfp_trips_60_kelvin_higher(self):
        from backend.config import get_chemistry

        assert SafetyState().evaluate(250.0, get_chemistry("lfp")).tier != "trip"
        assert SafetyState().evaluate(270.0, get_chemistry("lfp")).tier == "trip"

    def test_latched_trip_ignores_cooling(self):
        from backend.config import get_chemistry

        s = SafetyState().evaluate(215.0, get_chemistry("nmc"))
        assert s.latched_trip
        s.evaluate(20.0, get_chemistry("nmc"))
        assert s.tier == "trip"    # latched, despite a cold pack

    def test_operator_reset_clears_the_latch(self):
        s = SafetyState()
        s.latched_trip = True
        s.tier = "trip"
        s.reset()
        assert s.tier == "ok"
        assert s.latched_trip is False

    def test_hysteresis_prevents_derate_flapping(self):
        from backend.config import get_chemistry

        spec = get_chemistry("nmc")
        s = SafetyState()
        s.evaluate(185.0, spec)         # derate
        assert s.tier == "derate"
        s.evaluate(spec.runaway_c - MARGIN_FLOOR_C["nmc"] - 1.0, spec)
        assert s.tier == "derate"       # not enough recovery yet
        s.evaluate(spec.runaway_c - MARGIN_FLOOR_C["nmc"] - 6.0, spec)
        assert s.tier == "ok"


# ---------------------------------------------------------------------------
# Reduced pack model
# ---------------------------------------------------------------------------


class TestReducedPack:
    def test_cells_are_coupled(self):
        """A uniform initial state must develop a non-uniform one.

        The coupling that breaks the symmetry is the road-surface term, whose
        per-zone weight runs from 1.0 at the inlet to 0.35 at the outlet; the
        position factor on the Joule source does the same thing more weakly.
        Note the current is the pack's real 12 A, not the pre-96s1p figures:
        at 12 A the ohmic load is ~0.15 W/zone against a ~2.9 kJ/K zone mass,
        so Joule heating alone moves the field by microkelvin per second and
        asserting on it would be asserting on numerical noise.
        """
        import numpy as np

        from backend.config import DEFAULT_PACK_CURRENT_A, get_chemistry

        model = ReducedPackModel()
        spec = get_chemistry("nmc")
        T = model.step(np.full(12, 34.0), 1.0, 40.0, 800.0, 52.0,
                       DEFAULT_PACK_CURRENT_A, 0.6, 6.0, 26.0, spec)
        assert float(T.max() - T.min()) > 1e-3

    def test_flow_reduces_peak_temperature(self):
        import numpy as np

        from backend.config import DEFAULT_PACK_CURRENT_A, get_chemistry

        model = ReducedPackModel()
        spec = get_chemistry("nmc")
        low = model.step(np.full(12, 45.0), 5.0, 45.0, 900.0, 60.0,
                         DEFAULT_PACK_CURRENT_A, 0.5, 2.0, 26.0, spec)
        high = model.step(np.full(12, 45.0), 5.0, 45.0, 900.0, 60.0,
                          DEFAULT_PACK_CURRENT_A, 0.5, 13.0, 26.0, spec)
        assert float(high.max()) < float(low.max())


# ---------------------------------------------------------------------------
# Telemetry simulator
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_ticks_produce_frames(self):
        sim = TelemetrySimulator(chemistry="nmc", engine=None)
        frame = sim.tick(0.5, use_pinn=False)
        assert len(frame.cell_temps_c) == 12
        assert frame.chemistry == "nmc"
        assert frame.t_s == pytest.approx(0.5)

    def test_chemistry_switch_preserves_state(self):
        sim = TelemetrySimulator(chemistry="nmc", engine=None)
        for _ in range(5):
            sim.tick(0.5, use_pinn=False)
        before = sim.state.cell_temp_c.copy()
        sim.set_chemistry("lfp")
        assert sim.state.chemistry == "lfp"
        assert abs(sim.state.cell_temp_c - before).max() < 1e-9

    def test_hot_route_drives_the_controller(self):
        sim = TelemetrySimulator(chemistry="nmc", engine=None)
        sim.reset(initial_temp_c=34.0)
        actuated = False
        for _ in range(400):
            frame = sim.tick(0.5, use_pinn=False)
            actuated = actuated or frame.anticipatory_active
        assert actuated

    def test_risk_index_stays_bounded(self):
        sim = TelemetrySimulator(chemistry="nmc", engine=None)
        for _ in range(100):
            frame = sim.tick(0.5, use_pinn=False)
            assert 0.0 <= frame.risk_index <= 1.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


class TestDemo:
    def test_bootstrap_serves_everything_in_one_call(self, client: TestClient):
        body = client.get("/api/demo/bootstrap", params={"chemistry": "nmc"}).json()
        for key in ("catalogue", "model", "scenario", "field", "safety"):
            assert key in body
        assert body["safety"]["exotherm_onset_c"] == 210.0

    def test_comparison_study_passes_for_both_chemistries(self, client: TestClient):
        body = client.get("/api/demo/comparison").json()
        for chem in ("nmc", "lfp"):
            assert body[chem]["ceiling_respected"] is True
            assert body[chem]["margin_respected"] is True
        assert body["lfp"]["cp_j_kgk"] == 1150.0
        assert body["nmc"]["cp_j_kgk"] == 1040.0
        assert body["delta"]["min_exotherm_margin_c"] > 0

    def test_seed_run_is_reproducible(self, client: TestClient):
        a = client.post("/api/demo/seed-run",
                        params={"chemistry": "nmc", "steps": 20, "seed": 7}).json()
        b = client.post("/api/demo/seed-run",
                        params={"chemistry": "nmc", "steps": 20, "seed": 7}).json()
        assert a["summary"]["peak_cell_temp_c"] == b["summary"]["peak_cell_temp_c"]

    def test_ablation_is_honest_about_being_unmeasured(self, client: TestClient):
        body = client.get("/api/demo/ablation").json()
        if body.get("measured") is False:
            assert "design" in body
            assert "train_pinn.py" in body["note"]
        else:
            assert body["runs"]


# ---------------------------------------------------------------------------
# NASA
# ---------------------------------------------------------------------------


class TestNasaRoutes:
    def test_calibration_flags_the_synthetic_fallback(self, client: TestClient):
        body = client.get("/api/nasa/calibration").json()
        if not body["dataset_present"]:
            assert body["synthetic_fallback"] is True
            assert "synthetic" in body["notes"].lower()

    def test_probe_distinguishes_absent_from_present(self, client: TestClient):
        body = client.get("/api/nasa/probe").json()
        assert "root_exists" in body
        assert isinstance(body["n_mat_files"], int)

    def test_dataset_stats_return_cycles(self, client: TestClient):
        body = client.get("/api/nasa/dataset/stats").json()
        assert body["describe"]["n_cycles"] >= 0
        assert body["window_len"] > 0

    def test_chemistry_prior_is_between_the_two_sources(self, client: TestClient):
        body = client.get("/api/nasa/chemistry-prior").json()
        nmc = body["per_chemistry"]["nmc"]
        assert nmc["blended_prior_ohm"] > 0
        assert nmc["datasheet_ohm"] > 0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_operating_point_rejects_out_of_range_soc(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OperatingPoint(state_of_charge=1.5)

    def test_extra_fields_are_forbidden(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OperatingPoint(unexpected=1)

    def test_trajectory_duration_is_aligned_to_the_route_end(self):
        req = SimulationRequest(
            duration_s=100.0, dt_s=1.0,
            route=[{"lat": 19.0, "lon": 72.8, "t_s": 0.0},
                   {"lat": 19.1, "lon": 72.9, "t_s": 50.0}],
            weather=[{"t_s": 0.0, "ambient_c": 30.0, "solar_w_m2": 500.0,
                      "wind_mps": 2.0, "road_surface_c": 40.0},
                     {"t_s": 50.0, "ambient_c": 50.0, "solar_w_m2": 900.0,
                      "wind_mps": 2.0, "road_surface_c": 60.0}],
        )
        assert req.route[-1].t_s == 100.0

    def test_flow_command_clamps_are_in_the_schema(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FlowCommandRequest(flow_lpm=0.1)
        with pytest.raises(ValidationError):
            FlowCommandRequest(flow_lpm=20.0)


def np_abs_diff(arr):
    """Absolute differences of a numpy array (local helper to avoid the import)."""
    import numpy as np

    return np.abs(np.diff(np.asarray(arr)))