r"""
Heatwave-Aware AI-Assisted Microchannel Battery Thermal Management Dashboard for EVs
=====================================================================================

A research-grade, full-stack system for predicting and controlling the thermal
state of a liquid-cooled microchannel battery pack.

What is here
------------

``backend/``      FastAPI service + PyTorch dual-chemistry PINN
``frontend/``     Next.js dashboard (App Router, Tailwind, Recharts)
``auto_heal/``    LangGraph self-healing monitor
``auto_heal.py``  Its command-line entry point
``scripts/``      Training, dataset staging and LaTeX validation
``docs/``         IEEEtran AI model card

Quick start
-----------
::

    # 1. Backend
    python -m venv .venv && .venv\\Scripts\\activate
    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000
    #    -> http://127.0.0.1:8000/docs

    # 2. Frontend (separate terminal)
    cd frontend && npm install && npm run dev
    #    -> http://localhost:3000

    # 3. Optional: stage the NASA dataset and train the PINN
    python scripts/download_nasa.py
    python scripts/train_pinn.py --quick

    # 4. Optional: exercise the self-healing loop
    python auto_heal.py --dry-run

See README.md for the full walkthrough.
"""

__version__ = "1.0.0"

PROJECT_NAME = (
    "Heatwave-Aware AI-Assisted Microchannel Battery Thermal Management "
    "Dashboard for EVs"
)

__all__ = ["PROJECT_NAME", "__version__"]