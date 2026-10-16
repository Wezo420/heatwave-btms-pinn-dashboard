r"""
scripts.download_nasa
=====================

Fetch the NASA PCoE Battery Data Set and stage it for the loader.

Source
------
The dataset is distributed as a ~2 MB archive of MATLAB ``.mat`` files
(``B0005.mat`` ... ``B0034.mat`` plus ``Rests.mat`` and ``BicycleTest.mat``)
containing run-to-failure charge/discharge cycle data.  It is hosted publicly
and is not redistributable, which is why :mod:`backend.models.nasa_dataset`
carries a synthetic fallback.

Two acquisition routes
----------------------
1.  **Archive** -- ``--url``.  Point this at a direct download of the archive;
    it is extracted into ``backend/data/nasa/``.
2.  **Manual**  -- no URL configured.  The script prints the canonical
    instructions and verifies whatever is already on disk.

The script is deliberately non-fatal: it never raises on a download failure, it
reports the state and exits 0, because "the dataset is absent" is a supported,
documented condition of this project rather than an error.

Usage
-----
::

    python scripts/download_nasa.py                 # verify / stage what is there
    python scripts/download_nasa.py --url <direct-download-url>
    python scripts/download_nasa.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import NASA_DATA_ROOT, NMC_CELL_IDS  # noqa: E402

DEFAULT_TARGET = REPO_ROOT / "backend" / "data" / "nasa"

INSTRUCTIONS = """
The NASA Battery Data Set is not redistributable, so it is not bundled here.

To stage it manually:

  1. Download the "Battery Data Set" archive from the NASA PCoE site:
         https://www.nasa.gov/content/battery-data-set/
     (You will need to accept the data-use agreement on that page.)

  2. Extract it so that the ``.mat`` files end up in:
         backend/data/nasa/BatteryData/B0005.mat
         backend/data/nasa/BatteryData/B0006.mat
         ...

     If you put them somewhere else, set the environment variable:
         export NASA_BATTERY_DATA_ROOT=/path/to/BatteryData      # macOS / Linux
         set NASA_BATTERY_DATA_ROOT=C:\\path\\to\\BatteryData     # Windows

  3. Verify and refit:
         python scripts/download_nasa.py --verify-only
         curl -X POST http://127.0.0.1:8000/api/nasa/calibrate

Until then the loader uses a deterministic synthetic run-to-failure generator
whose statistics match the published NASA cells, and every API response that
depends on it sets ``synthetic_fallback: true``.
"""


def _mat_files(target: Path) -> List[Path]:
    """Every ``B*.mat`` under ``target`` (recursively), sorted."""
    if not target.exists():
        return []
    return sorted(p for p in target.rglob("B*.mat") if p.is_file())


def verify(target: Path) -> dict:
    """Report what is present under ``target``."""
    files = _mat_files(target)
    ids = sorted({p.stem for p in files})
    wanted = sorted(NMC_CELL_IDS)
    missing = [c for c in wanted if c not in ids]

    info = {
        "target": str(target),
        "exists": target.exists(),
        "n_mat_files": len(files),
        "cell_ids": ids,
        "expected_cells": wanted,
        "missing_cells": missing,
        "complete": not missing and bool(files),
    }

    if files:
        # Decode one file to prove it is readable, not merely present.
        try:
            from backend.models.nasa_dataset import _read_mat

            keys = sorted(_read_mat(files[0]).keys())
            info["first_file"] = files[0].name
            info["first_file_keys"] = keys
            info["decodable"] = True
        except Exception as exc:
            info["decodable"] = False
            info["decode_error"] = repr(exc)

    return info


def download(url: str, target: Path, timeout: float = 120.0) -> Optional[Path]:
    """Download ``url`` and extract it under ``target``.

    Returns the path of the downloaded file, or ``None`` on failure (the caller
    reports rather than raises).
    """
    target.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if url.lower().split("?")[0].endswith(".zip") else ".dat"
    dest = target / f"_nasa_download{suffix}"

    try:
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "btms-nasa-loader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                dest.open("wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            read = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100.0 * read / total
                    print(f"\r  {read/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)",
                          end="", flush=True)
        print()
    except Exception as exc:
        print(f"\ndownload failed: {exc!r}", file=sys.stderr)
        print("Falling back to manual instructions (below).", file=sys.stderr)
        return None

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(target)
            print(f"extracted to {target}")
        except zipfile.BadZipFile as exc:
            print(f"archive is not a valid zip: {exc}", file=sys.stderr)
            return None
        finally:
            dest.unlink(missing_ok=True)
    else:
        print(f"saved {dest}; rename it to .zip if it is an archive and re-run",
              file=sys.stderr)

    return dest


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; never raises, always reports."""
    p = argparse.ArgumentParser(
        prog="download_nasa.py",
        description="Stage the NASA Battery Data Set for the calibration loader.",
    )
    p.add_argument("--url", default=os.getenv("NASA_DATA_URL"),
                   help="direct download URL for the archive")
    p.add_argument("--target", default=str(DEFAULT_TARGET),
                   help="directory to stage the .mat files into")
    p.add_argument("--verify-only", action="store_true",
                   help="do not download; only report what is present")
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = p.parse_args(argv)

    target = Path(args.target).resolve()
    report: dict = {}

    if args.url and not args.verify_only:
        download(args.url, target)
        report = verify(target)
    else:
        report = verify(target)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 68)
        print("NASA Battery Data Set -- staging report")
        print("=" * 68)
        print(f"target        : {report['target']}")
        print(f"exists        : {report['exists']}")
        print(f".mat files    : {report['n_mat_files']}")
        if report.get("cell_ids"):
            print(f"cell ids      : {', '.join(report['cell_ids'])}")
        if report.get("missing_cells"):
            print(f"missing       : {', '.join(report['missing_cells'])}")
        if "decodable" in report:
            print(f"decodable     : {report['decodable']}"
                  + ("" if report["decodable"] else f" ({report.get('decode_error')})"))
        print(f"complete      : {report['complete']}")
        print()

        if report["complete"]:
            print("The real dataset is staged. Refit the calibration with:")
            print("    curl -X POST http://127.0.0.1:8000/api/nasa/calibrate")
        else:
            print(INSTRUCTIONS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())