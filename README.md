<<<<<<< HEAD
# Heatwave-Aware AI-Assisted Microchannel Battery Thermal Management Dashboard for EVs

A dual-chemistry physics-informed neural network predicting the **3-D temperature
field** of a liquid-cooled microchannel EV battery pack, with an **anticipatory
coolant controller** and a **self-healing supervisor** for the inference service.

Three deliverables, one repository:

| Deliverable | What it is | Where |
|---|---|---|
| **Model card** | IEEEtran LaTeX, 3 pages: model details, intended use, benchmarks, ethical & safety | [`docs/Model_Card_Gen_AI.pdf`](docs/Model_Card_Gen_AI.pdf) · source `docs/model_card.tex` |
| **The system** | `MicrochannelPINN` behind a FastAPI service + a Next.js dashboard | `backend/`, `frontend/` |
| **Auto-heal loop** | LangGraph state machine: classify → hypothesise → patch → verify → restart, or escalate | `auto_heal/`, `auto_heal.py` |

---

## The central claim

**One parameter set resolves the thermal field for both NMC and LFP.**

```python
model.temperature(point_in_metres, chemistry="nmc")   # 210 °C exotherm onset
model.temperature(point_in_metres, chemistry="lfp")   # 270 °C exotherm onset
```

Chemistry enters as a 12-dimensional descriptor vector through FiLM
conditioning, not as a second network. A third chemistry is a registry entry and
one embedding row — not a retrain.

---

## Quick start

```powershell
# Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Dashboard (second terminal)
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Open <http://localhost:3000>. API docs at <http://localhost:8000/docs>.

The NASA battery dataset is **not** redistributable, so the loader falls back to
a deterministic synthetic run-to-failure generator whose statistics match the
published cells. Every dependent response carries `synthetic_fallback: true`
and the dashboard shows a banner. To use the real data:

```powershell
python scripts/download_nasa.py
```

Full setup and demo run book: **[SETUP.md](SETUP.md)**.

---

## What's in the physics

The physics is written from scratch, not approximated by a surrogate. It is in
the *loss*, not only in the data:

- **Solid energy** — anisotropic conduction `∇·(k∇T)` with an Arrhenius Joule
  source `I²R(T, SOC)` and conjugate interface continuity at the bonded
  cold-plate interface.
- **Momentum** — a reduced Navier–Stokes balance over an exact plane-Poiseuille
  stream-function ansatz. Verified to machine precision: `|∇·u| = 0` exactly,
  no-slip zero on all four channel walls, flow↔pressure-gradient round-trip
  exact.
- **Heat transfer** — Shah–London constant-heat-flux Nusselt with a Hausen
  entrance enhancement. Square-duct limit gives exactly 3.92.
- **Calibration** — `R_int` regressed out of NASA run-to-failure cycles,
  recovering `R₀ = 19.09 mΩ` and `Eₐ = 23.50 kJ/mol` against a generator seeded
  at 19.0 mΩ / 23.5 kJ/mol.

Loss terms are rebalanced by a gradient-norm penalty so the large-magnitude PDE
residual cannot starve the boundary conditions.

---

## Safety boundary

**The network is advisory. Actuation is not.**

- Exotherm thresholds (210 °C / 270 °C) are **hard-coded constants**, never
  learned. A model that learned its own runaway threshold could learn a wrong
  one.
- The trip decision belongs to a **non-learned three-tier supervisor**:
  `ok → derate → trip(latched)`. No invariant in the learned output can bypass
  it.
- Every returned field ships its **own adjoint residual** and a convergence
  flag. The model is permitted to be wrong; it is not permitted to be quietly
  unconverged.
- The **self-healing monitor cannot edit safety logic**. The failure taxonomy
  routes request-validation errors to *no action* — mutating a validated system
  because a client sent a bad body defeats the validation that protects it.

---

## Self-healing loop

```
monitor ──healthy──▶ report ──▶ END
  │ crash
  ▼
classify ──no traceback──▶ escalate ──▶ END
  ▼
hypothesise ──UNKNOWN──▶ diagnose_llm ──▶ patch
  │ known                       │
  └─────────────────────────────┘
  ▼
patch ──not applied──▶ escalate
  ▼
verify ──static fail──▶ rollback ──▶ monitor   (same baseline)
  ▼
restart ──healthy──▶ report ──▶ END
```

Three properties worth stating: **restart precedes patching** (most "crashes"
are transient — patching because a server started twice is how an auto-healer
breaks working code); **rollback precedes retry** (attempt *N+1* starts from the
same baseline as attempt 1, so a failed patch cannot compound); and
**escalation is a first-class terminal state**, not an exception.

```powershell
python auto_heal.py --graph-diagram                  # the state machine
python auto_heal.py --inject kill                    # diagnose, then refuse
python auto_heal.py --dry-run --inject module        # real diagnosis + patch
```

---

## Tests

```powershell
python -m pytest backend/tests/ -q            # 228 tests, ~4 min
python -m pytest backend/tests/test_auto_heal.py -q   # graph wiring, ~25 s
```

The graph-wiring suite is the interesting one. LangGraph merges a node's
returned dict into state field-by-field and **silently discards keys absent
from the `TypedDict`**, so a node can be individually correct while the graph
is wrong, and a node-level unit test cannot see it. Three separate production
bugs lived in exactly that gap — including one where the monitor reported a
crashed service as *healed* without ever classifying it. An AST-based test now
asserts all three directions (read ↔ written ↔ declared) hold.

---

## Repository layout

```
backend/
  config.py            chemistry registry, cold-plate geometry, pack topology
  physics.py           stream function, residual operators, constitutive laws
  pinn.py              MicrochannelPINN, gradient-norm balancer, loss
  nasa_dataset.py      MAT v5 reader, R_int regression, synthetic generator
  services/            inference, heatwave planner, telemetry simulator
  routers/             FastAPI endpoints
  tests/               228 tests
frontend/              Next.js 14 App Router + Recharts + Tailwind
auto_heal/             LangGraph state machine, traceback parser, patcher
scripts/               train_pinn.py, download_nasa.py, check_tex.py
docs/                  model card (LaTeX source + compiled PDF)
artifacts/             trained checkpoint, calibration, metrics, provenance
SETUP.md               demo run book
```

---

## Reproducibility

Every run seeds Python, NumPy and Torch from a single integer (1337).
`cudnn.deterministic` is set and benchmarking disabled. `scripts/train_pinn.py
--ablation` reproduces the metrics table exactly on the same machine. The
Fourier frequency scale σ is a **config value independent of the feature count** —
tying it to the feature count makes the initial PDE residual grow as `n²`,
because the physics loss differentiates the field twice.

```powershell
python scripts/train_pinn.py --quick          # smoke test, ~2 min
python scripts/train_pinn.py --ablation       # full ablation table
```

## Known limitations

Stated plainly, and in full in §IV of the model card:

- **Spatial accuracy is below target.** RMSE ≈ 9.7 K (NMC) / 6.5 K (LFP) against
  a ≤ 1.5 K criterion. The physics term still dominates the data term; the
  NASA-derived labels and the PDE's assumed operating point are not yet
  mutually consistent. This is a modelling inconsistency, not an optimisation
  failure.
- The NASA dataset is **synthetic** in this repository (see above).
- Coolant properties are assumed isotropic and temperature-independent over
  25–45 °C, under-predicting viscosity drift by ~12 % at the upper end.
- The controller's anticipation mis-fires in unmodelled micro-climates
  (underpass, tunnel, dense canopy). The dashboard shows predicted effective
  ambient separately from the raw reading precisely so this is visible.

## License

MIT — see [LICENSE](LICENSE).
=======
# heatwave-btms-pinn-dashboard
Heatwave-aware AI-assisted microchannel battery thermal management for EVs.  Dual-chemistry (NMC/LFP) physics-informed neural network predicting the 3-D temperature field of a liquid-cooled microchannel pack from a single shared parameter set, with an anticipatory coolant controller and a LangGraph self-healing supervisor.
>>>>>>> 30c0abc248664790f21dfbb3caa67c55f5bcc506
