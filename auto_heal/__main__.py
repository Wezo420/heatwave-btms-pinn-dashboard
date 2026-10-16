r"""
auto_heal.__main__
=================

``python -m auto_heal`` runs exactly the same monitor as ``python auto_heal.py``.

Why this shim exists
--------------------

The repository deliberately carries *both* a package (``auto_heal/``) and a
script (``auto_heal.py``) with the same name. That is a collision waiting to
happen, and it is worth being explicit about how it resolves:

* Running ``python auto_heal.py`` executes the script directly. The script never
  has to import itself, so nothing collides and this is the documented entry
  point.
* ``import auto_heal`` binds to the **package**, because a directory with an
  ``__init__.py`` takes precedence over a same-named module on the same path.
  The script is therefore unreachable by name, which is why every intra-package
  import here uses a relative path (``from .graph import ...``) rather than an
  absolute one that would be misresolved.

``python -m auto_heal`` needs the *script*, not the package, so this module
loads it by file path through :func:`runpy.run_path`. Importing it normally
would recurse into the package's own ``__init__``, which imports the graph,
which needs langgraph installed -- an unnecessary precondition for merely asking
for ``--help``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "auto_heal.py"

if not _SCRIPT.is_file():  # pragma: no cover - a broken checkout
    sys.stderr.write(
        f"error: expected the CLI script at {_SCRIPT} but it does not exist.\n"
        f"       Run `python {_SCRIPT.name}` from the repository root instead.\n"
    )
    raise SystemExit(2)

if __name__ == "__main__":
    # run_name="__main__" so the script's own `if __name__ == "__main__"` guard
    # fires and the exit code propagates.
    runpy.run_path(str(_SCRIPT), run_name="__main__")
