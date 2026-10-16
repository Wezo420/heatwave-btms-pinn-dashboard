# Demo Setup & Run Book

**Heatwave-Aware AI-Assisted Microchannel Battery Thermal Management Dashboard for EVs**

Three deliverables, three terminals, one story. This document gets all three
running, then walks you through presenting them.

| # | Deliverable | What it is | Where it lives | Port |
|---|---|---|---|---|
| 1 | **The model** | `MicrochannelPINN`, a dual-chemistry (NMC/LFP) physics-informed neural network behind a FastAPI service | `backend/` | `8000` |
| 2 | **The dashboard** | Next.js App Router UI — chemistry toggle, live telemetry, 3-D thermal field, anticipatory heatwave alert | `frontend/` | `3000` |
| 3 | **The self-healing monitor** | LangGraph state machine that catches the backend crashing, diagnoses it, patches it, restarts it | `auto_heal/` | supervises `8000` |
| — | **The model card** | IEEEtran LaTeX document, 3 pages, endorsement-style | `docs/model_card.tex` | — |

---

## 0. What you need before you start

| Tool | Version | Why |
|---|---|---|
| Python | 3.11–3.13 | torch, FastAPI, LangGraph. **3.14 works on this machine but torch wheels lag.** |
| Node.js | ≥ 18.18 | Next.js 14, Recharts, Tailwind |
| A LaTeX distribution | TeX Live / MiKTeX | To compile the model card to PDF |

> **Not installed on this machine:** a LaTeX distribution. `docs/model_card.tex`
> is complete and passes `python scripts/check_tex.py` (structural validation:
> brace balance, environment balance, label/ref integrity, citation coverage,
> and the `\cellcolor`-needs-`xcolor[table]` check). To get a PDF, either
> install MiKTeX locally or upload `docs/model_card.tex` to
> [Overleaf](https://overleaf.com) and compile with pdfLaTeX — it has no
> external images, no custom fonts, and no `.sty` files outside a standard
> install.

### One-time install

Open **Terminal A** at the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then, if you need the real NASA dataset rather than the synthetic generator:

```powershell
python scripts/download_nasa.py
```

That fetches the NASA PCoE Battery Data Set into `backend/data/nasa/BatteryData/`.
**Skip it if you just want the demo** — every dependent endpoint reports
`synthetic_fallback: true` and the dashboard shows a banner, which is itself a
feature worth pointing out (the provenance is never faked).

### Verify the backend before you demo

```powershell
python -m pytest backend/tests/ -q
```

215 backend tests, ~3.5 min. The `auto_heal` subset (63 of them, including the
graph-wiring guards described in §4) runs in ~25 s on its own:

```powershell
python -m pytest backend/tests/test_auto_heal.py -q
```

If anything fails, do not present — the failure list is the honest answer to
"what doesn't work".

---

## 1. Start the backend

**Terminal A**, from the project root, venv activated:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info
```

Wait for:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal visible during the demo — the request log is evidence that
the dashboard is live, not a video.

Confirm it:

```powershell
curl http://127.0.0.1:8000/api/health
```

```powershell
curl http://127.0.0.1:8000/api/demo/bootstrap
```

`/api/demo/bootstrap` returns the catalogue, model info, scenario, field and
safety data in one call — it is exactly what the dashboard fetches on first
paint, so if it works, everything downstream works.

---

## 2. Start the dashboard

**Terminal B**, from the project root:

```powershell
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Wait for:

```
▲ Next.js 14.2.23
- Local:        http://localhost:3000
```

Open <http://localhost:3000>.

The `next.config.mjs` rewrites `/api/*` and `/ws/*` to `http://127.0.0.1:8000`,
so there is nothing else to configure. `.env.local` is safe to leave at its
defaults.

### What to click, in order

1. **Chemistry toggle (top left) — NMC ⇄ LFP.** This is the headline
   architectural claim. Switching is a *model* switch, not a display theme: it
   changes `c_p` (1040 → 1150 J/kg·K), the exotherm onset (210 → 270 °C), the
   comfort ceiling, and the required coolant set-point in one step, from **one
   shared parameter set**. Watch the 3-D thermal field and the exotherm-margin
   readout move together. If you only show one thing, show this.

2. **Anticipatory Heatwave Alert overlay.** The scenario is a pack that has just
   come off DC fast charging (38 °C) entering a heatwave with the coolant pump
   at its part-load 2 L/min. The overlay appears *before* the peak and shows
   the predicted effective ambient separately from the raw reading — that gap
   is the 420 s first-order ambient lag, i.e. the model "knowing" about a rising
   ambient about seven minutes before the cabin thermometer does.

3. **Telemetry charts** — cell temperatures per zone and commanded coolant
   flow. Point out that **advisory** (predicted) and **enforced** (commanded)
   actuations are drawn in distinct styles and never merge into one line.

4. **Control panel.** Command a flow manually and watch it get rate-limited
   locally, exactly as the backend limits it.

5. **Residual panel.** Every field evaluation ships its own adjoint residual and
   a convergence flag. The model is allowed to be wrong; it is not allowed to be
   silently unconverged.

---

## 3. Compile the model card

**On this machine:** no LaTeX is installed, so use Overleaf or install MiKTeX.

```powershell
# If you install MiKTeX:
pdflatex docs/model_card.tex
pdflatex docs/model_card.tex     # second pass resolves \ref and hyperref bookmarks
```

**On Overleaf:** upload `docs/model_card.tex` as `main.tex`, set the compiler to
pdfLaTeX, compile. Nothing else is needed.

### Where the code is and how to read it in a demo

`docs/model_card.tex` — 908 lines. The structure, and what to show:

| Section | Lines | What to say |
|---|---|---|
| Preamble + packages | 20–65 | `IEEEtran` `[journal,10pt,compsoc]`, `amsmath`/`graphicx`/`booktabs`/`hyperref` all present, `hyperref` loaded last, `xcolor[table]` for the shaded table rows |
| **Capability panel** | 127–178 | The endorsement face. Two columns: "What the model does" / "What it refuses to do", each traceable to a section by `\ref`. This is the page to leave on screen while you talk. |
| Reviewer note box | 180–193 | States that no number in the card is illustrative |
| §I Model Details | 215–464 | Architecture, loss formulation, gradient rebalancing, NASA calibration |
| §II Intended Use | 466–541 | Edge deployment contract, the 800 ms latency budget, HIL testing, out-of-scope uses |
| §III Metrics & Benchmarks | 543–648 | **Table I** — the acceptance-criteria table. Five shaded groups: spatial accuracy, physical consistency, operational, cross-chemistry stability, calibration. A "Target" and a "Status" column, so nobody confuses a threshold with a measurement. |
| §IV Ethical Considerations & Safety | 654–822 | The advisory/actuation boundary, the non-learned fail-safe state machine, hard-coded 210/270 °C thresholds, drift handling, self-heal limits |
| **Endorsement box** | 827–851 | The explicit "what we endorse / what we do not endorse" statement |
| Bibliography | 861–908 | 9 references, all cited |

Validate structurally any time you edit it — this is the check that catches a
dangling `\ref` before a reviewer does:

```powershell
python scripts/check_tex.py docs/model_card.tex
```

---

## 4. The self-healing loop

### Where the code is

```
auto_heal.py                  CLI entry point, fault injection, exit codes
auto_heal/
  __main__.py                 `python -m auto_heal` shim -- see the note below
  graph_state.py              HealState TypedDict + FailureClass taxonomy
  traceback_parser.py         traceback -> structured TracebackInfo + FailureClass
  llm.py                      RuleBasedHypothesiser (default) + LLMHypothesiser
  patching.py                 backup, string patch, AST patch, compile+import checks
  logger.py                   auto_heal.log (human) + auto_heal.jsonl (audit)
  nodes.py                    the nine node functions
  graph.py                    the LangGraph StateGraph + conditional edges
```

> **Why there is both `auto_heal.py` and `auto_heal/`.** Python's import rules
> give the package precedence, so `import auto_heal` binds to the directory and
> the script is unreachable by name — every intra-package import is therefore
> relative. `auto_heal/__main__.py` bridges the gap so `python -m auto_heal`
> works identically to `python auto_heal.py`. Without that shim the `-m` form
> would import the package and exit silently, running no monitor at all.
> Prefer the explicit script form when you are presenting; it is unambiguous.

### How it works

```
START
  │
  ▼
monitor ──healthy──▶ report ──▶ END
  │ crash
  ▼
classify ──no traceback──▶ escalate ──▶ END
  │
  ▼
hypothesise ──UNKNOWN──▶ diagnose_llm ──▶ patch
  │ known                       │
  └─────────────────────────────┘
  ▼
patch ──not applied──▶ escalate
  │ applied
  ▼
verify ──static fail──▶ rollback ──▶ monitor   (next attempt, same baseline)
  │ static ok
  ▼
restart ──healthy──▶ report ──▶ END
  │ still failing
  ▼
  (loop back to monitor, up to --max-attempts, else escalate)
```

Three properties worth stating out loud when you present it, because they are
the ones a sceptical reviewer asks about:

1. **Restart precedes patching.** Most "crashes" in a live workflow are
   transient — a stale worker, a port still held. Patching source because a
   server started twice is how an auto-healer breaks working code.
2. **Rollback precedes retry.** Attempt *N+1* always begins from the same
   baseline as attempt 1, so a failed patch cannot compound.
3. **Escalation is a first-class terminal state.** An agent that exhausts its
   budget and stops loudly with a report is behaving correctly. Silently
   continuing to poke at source is not.

### The safety constraint — lead with this

The monitor **may not modify safety logic**. The `FailureClass` taxonomy routes
request-validation errors to *no action*, because mutating a validated system
because a client sent a bad body defeats the validation that protects it. The
fail-safe supervisor, the exotherm thresholds and the safety margins are all
outside the patch allow-list (`backend/`, `auto_heal/`, `scripts/` only).

### Four bugs this loop shipped with, and why they are the best part of the demo

If a reviewer asks "does it actually work", this is the honest answer, and it is
a better answer than "yes". Every one of these was a **live, silent failure** in
the state wiring, and none was visible to the 44 node-level tests that passed:

| Bug | Effect | Fixed by |
|---|---|---|
| `failed` returned but never declared in `HealState` | LangGraph dropped it, the conditional edge read `None`, took the falsy branch, and reported a **crashed service as healed** without classifying anything | Declare the field **and** change the route to an explicit `is False` test |
| `patch_occurrence` read but never written | Every patch silently used occurrence 1 no matter what the hypothesis asked for — wrong place on any file with more than one similar line | Have both hypothesis nodes write it |
| `raw_traceback` written but never declared | The structured diagnosis survived while the incident report quietly lost the verbatim traceback | Declare the field |
| `HealLogger.error()` called with `message=` | The kwarg collided with the positional parameter and crashed the monitor **while reporting the crash it had just diagnosed** | Rename the kwarg |

The root cause is one property of LangGraph: it merges a node's returned dict
into the state field-by-field and **discards keys absent from the TypedDict,
silently**. Your nodes can all be individually correct while the graph is
wrong, and a node-level unit test cannot see it.

Two structural tests now enforce all three directions (read↔write↔declared) by
parsing the node source's AST, plus a runtime test that calls each node and
inspects what actually comes back. A fourth instance of the same class fails
those tests rather than failing in front of an audience. The first time I added
the AST test it immediately caught the `patch_occurrence` bug I had just
"fixed" — so I removed the write, watched the test fail, and put it back.

### Show it live

Open the diagram first — it is the clearest single artefact in the whole repo:

```powershell
python auto_heal.py --graph-diagram
```

**Demonstration 1 — a correct diagnosis, then a correct refusal.** This is the
most honest of the three, and it is the one to open with:

```powershell
python auto_heal.py --inject kill --max-attempts 1 --startup-timeout 10
```

`kill` writes a throwaway script that dies on a shape mismatch. The monitor then
does everything right and still refuses to act, which is the point:

1. `monitor` — the service never becomes healthy
2. `classify` — **`cuda_shape_mismatch`**, `RuntimeError at inject_crash.py:9`
3. `hypothesise` — root cause at **0.68 confidence** from the rule base
4. `patch` — the anchor it wants to patch does not exist in the crash script, so
   it **fails the patch rather than applying it somewhere wrong**
5. `escalate` — `EXHAUSTED`, exit code 1, incident report written

Say this explicitly: *the agent diagnosed the failure correctly and then refused
to write a patch it could not verify. A self-healer that always produces a patch
is more dangerous than one that stops.*

**Demonstration 2 — a real diagnosis and a real proposed patch:**

```powershell
python auto_heal.py --dry-run --inject module --max-attempts 1
```

`--dry-run` writes nothing, and verifies nothing either — the point of this demo
is the *diagnosis and the patch plan*, not a completed heal. The monitor
classifies it as `missing_module`, forms a hypothesis at **0.55 confidence**, and
proposes a genuine source patch: make the optional `scipy` import non-fatal so
the loader's pure-Python MAT v5 reader chain falls through instead of failing the
whole pipeline. It reports exactly what it *would* have written and restores the
injected fault on exit.

Note the target it prints is `backend/config.py` — repo-relative and
copy-pasteable. An earlier version printed a mangled absolute path
(`Users/you/proj/backend/config.py`) because it stripped the `C:\` drive prefix
instead of making the path relative to the repository root.

**Demonstration 3 — the healthy path (the one that always works):**

```powershell
python auto_heal.py --startup-timeout 45
```

No injection. The backend starts, `/api/health` returns 200, the monitor reports
`healed` with 0 attempts and exits 0. Use this to show the exit-code vocabulary:
`0` healthy-or-healed, `1` unhealable, `2` the monitor itself failed.

### Faults available for injection

| Name | What it breaks |
|---|---|
| `cuda_shape` | removes the shape normaliser in `pinn.py` |
| `nan_loss` | removes the non-finite guard on the Arrhenius argument in `physics.py` |
| `dtype` | forces the residual collocation batch to float32 |
| `dataset` | points the NASA data root at a non-existent directory |
| `module` | introduces a bad import in `config.py` |
| `kill` | a throwaway script in a temp dir that dies on a tensor shape mismatch |

`kill` is the odd one out: it injects nothing into the repository and instead
supervises a deliberately broken stub in the system temp directory. That is
deliberate — it exercises the full classify→hypothesise→patch→escalate path
without needing a real fault in the codebase, and because the stub lives outside
the repository the monitor has no legitimate file to patch, which is precisely
the situation where it should refuse. The script crashes on a CPU tensor pair
rather than `device="cuda"`, so the failure is reproducible on machines with no
GPU: a CUDA-gated crash would raise "Torch not compiled with CUDA enabled"
before ever reaching the multiplication, and the demo would fail for the wrong
reason.

Every injection restores itself via `atexit`, so a demo cannot leave the
repository broken.

### To use an LLM for the harder cases

```powershell
$env:OPENAI_API_KEY = "sk-..."
python auto_heal.py --prefer-llm --llm-model gpt-4o-mini
```

Without a key the deterministic rule-based hypothesiser runs instead — and it
is the one that produces the `missing_module` diagnosis above, so you do not
need a key for the demo.

---

## 5. The full demo script (~12 minutes)

**Setup:** Terminal A on the backend, Terminal B on `npm run dev`, browser on
`localhost:3000`, browser tab 2 on the PDF model card.

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the model card PDF, leave it on the capability panel | "One network, two chemistries. Here is what it does and what it refuses to do." |
| 1:00 | Back to the dashboard, toggle NMC → LFP | The shared-parameter-set claim. δ~5.4 K from one set of weights. |
| 2:30 | Let the heatwave scenario run to the alert overlay | "It ramps flow *before* the zone, using a 420 s ambient lag." |
| 4:00 | Point at the residual panel | "Every field ships its own residual and a convergence flag." |
| 5:30 | Comparison card: NMC vs LFP margins | "LFP's advantage is the 60 K onset gap, not lower resistance — it has *higher* r_int." |
| 6:30 | `python auto_heal.py --graph-diagram` | The state machine, on screen. |
| 7:30 | `python auto_heal.py --inject kill` | "It classifies correctly, forms a real hypothesis, then refuses to write a patch it cannot verify." |
| 9:00 | `python auto_heal.py --dry-run --inject module` | "With a matching fault it does diagnose and does propose a real patch." |
| 10:30 | Back to §IV of the model card | "And it is forbidden from touching safety logic by construction." |
| 11:30 | Endorsement box, last line | "Ready for HIL validation. Not ready to be trusted as a protection function." |

---

## 6. Things that will go wrong, and what to say

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard shows a synthetic-data banner | NASA `.mat` files not downloaded | Expected. Point at it as the provenance feature. Run `python scripts/download_nasa.py` if you want it gone. |
| `/api/demo/comparison` benefit is exactly 0.00 K | See below | Demonstrate at the part-load 2 L/min baseline — that is the regime where anticipation has work to do. |
| `--inject kill` exits 1 | The crash script is in a temp dir, so there is no repo file to patch | Correct behaviour. This is the slide, not a failure. |
| Backend takes ~700 ms on the *first* field call | Residual diagnostics warm-up in float64 | Second call is ~40 ms. The 800 ms budget includes the cold path. |
| Ablation table shows em-dashes | No training run has been committed | Say so. The card reports the ablation *design*, not invented numbers. |
| LangGraph `KeyError: thread_id` | You installed langgraph 0.2.x | `pip install "langgraph>=1,<2"`. The code passes the 1.x config. |

### The one subtlety to have ready

The incentive claim — "anticipation lowers peak temperature" — is only true in
one regime, and if a reviewer asks you to run it at 6 L/min you must not look
surprised. At 6 L/min the cold plate has roughly 4.7 kW of rejection at 16 K
above coolant, several times this pack's own ~1 kW heat load, so the
anticipatory schedule and the constant-flow baseline are **identical** and the
benefit is exactly 0.0 K. That is the model being right, not the controller
failing: there is nothing to anticipate when the pump is already wide open.

The claim is measured against a part-load 2 L/min baseline and a pack that
arrives at 38 °C from fast charging, which is when a reactive BMS would wait
until the 42 °C ceiling was already breached. There the same scenario gives
**4.33 K** of benefit for NMC and holds the ceiling. The model card states both
numbers.

---

## 7. One-screen reference

```powershell
# A) backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# B) dashboard
cd frontend; npm run dev

# C) self-heal, three demos
python auto_heal.py --graph-diagram                  # the state machine
python auto_heal.py --inject kill                    # diagnose, then refuse an unverifiable patch
python auto_heal.py --dry-run --inject module        # real diagnosis + proposed patch

# D) model card
python scripts/check_tex.py docs/model_card.tex      # structural validation

# E) the metrics behind the claims
python scripts/train_pinn.py --ablation              # populates artifacts/pinn_metrics.json
```

Useful endpoints to have in a second browser tab while you talk:

| URL | Shows |
|---|---|
| `/api/health` | Service + dependency status |
| `/api/demo/bootstrap` | Everything the first paint needs |
| `/api/demo/comparison` | NMC vs LFP side by side, with the interpretation text |
| `/api/predict/residuals` | Adjoint diagnostics + the `evaluated` flag + margin |
| `/api/simulate/scenario?n=40` | The heatwave route the demo runs |
| `/docs` | FastAPI's own OpenAPI explorer — the whole surface in one page |
