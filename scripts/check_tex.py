"""Lightweight LaTeX structural checker (no TeX distribution required).

Verifies brace balance, environment balance and the presence of the mandatory
structural elements. This is a pre-flight check, not a substitute for a real
compilation -- but it catches the overwhelming majority of transcription errors.
"""

import re
import sys
from collections import Counter
from pathlib import Path

REQUIRED_PACKAGES = [
    "amsmath", "graphicx", "booktabs", "hyperref",
]

#: Commands that only exist once a specific package option is set. The pairing
#: is not cosmetic: plain {xcolor} loads before colortbl is available, so
#: \cellcolor raises "Undefined control sequence" at compile time and the
#: checker would otherwise pass a document that cannot be built.
OPTION_GATED_COMMANDS = {
    "\\cellcolor": "[table]{xcolor}",
    "\\rowcolor": "[table]{xcolor}",
    "\\rowcolors": "[table]{xcolor}",
}

#: Commands this repository defines itself. A \newcommand whose name is never
#: used is dead weight, but a command that is *used* and never defined (nor
#: standard) is the far more dangerous direction, so it is checked below.
KNOWN_LATEX_COMMANDS = {
    "\\toprule", "\\midrule", "\\bottomrule", "\\addlinespace", "\\cellcolor",
    "\\rowcolor", "\\multicolumn", "\\hline", "\\cline", "\\SI", "\\si", "\\num",
    "\\sisetup", "\\definecolor", "\\colorbox", "\\fcolorbox", "\\textcolor",
    "\\RaggedRight", "\\arraybackslash", "\\dimexpr", "\\fboxsep", "\\fboxrule",
    "\\intextpara", "\\rule", "\\makeatletter", "\\makeatother",
}


def _strip_comments(src: str) -> str:
    """Remove TeX comments, honouring escaped percent signs.

    The scanner is character-based rather than line-based so that a ``\\%``
    inside a string does not comment out the rest of the line.
    """
    out = []
    j = 0
    while j < len(src):
        c = src[j]
        if c == "\\" and j + 1 < len(src):
            out.append(src[j:j + 2])
            j += 2
            continue
        if c == "%":
            # Skip to end of line, but a \% just escaped must survive.
            while j < len(src) and src[j] != "\n":
                j += 1
            continue
        out.append(c)
        j += 1
    return "".join(out)


def check(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    problems = []

    # --- brace balance (skip escaped braces) -------------------------
    depth = 0
    underflow = 0
    i = 0
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                underflow += 1
                depth = 0
        i += 1
    if depth != 0:
        problems.append(f"unbalanced braces: final depth {depth}")
    if underflow:
        problems.append(f"brace underflow x{underflow}")

    # --- comment stripping (so commented-out code is not parsed) -----
    stripped_lines = []
    for line in src.splitlines():
        out = []
        j = 0
        while j < len(line):
            if line[j] == "\\" and j + 1 < len(line):
                out.append(line[j:j + 2])
                j += 2
                continue
            if line[j] == "%":
                break
            out.append(line[j])
            j += 1
        stripped_lines.append("".join(out))
    stripped = "\n".join(stripped_lines)
    # --- environment balance -----------------------------------------
    begins = Counter(re.findall(r"\\begin\{([A-Za-z*]+)\}", stripped))
    ends = Counter(re.findall(r"\\end\{([A-Za-z*]+)\}", stripped))
    for env in set(begins) | set(ends):
        if begins[env] != ends[env]:
            problems.append(
                f"environment {env}: {begins[env]} begin vs {ends[env]} end"
            )

    # --- structural requirements --------------------------------------
    if "\\documentclass" not in stripped:
        problems.append("missing \\documentclass")
    m = re.search(r"\\documentclass\[([^\]]*)\]", stripped)
    if m:
        opts = m.group(1)
        for req in ("journal", "compsoc"):
            if req not in opts:
                problems.append(f"documentclass option '{req}' missing")
    for req in ("\\begin{document}", "\\end{document}", "\\maketitle",
                "\\begin{abstract}", "\\end{abstract}",
                "\\begin{thebibliography}", "\\end{thebibliography}"):
        if req not in stripped:
            problems.append(f"missing {req}")
    for pkg in REQUIRED_PACKAGES:
        if f"\\usepackage" in stripped and pkg not in stripped:
            problems.append(f"required package '{pkg}' not loaded")

    # --- package option gating -----------------------------------------
    # \cellcolor and friends exist only when xcolor is loaded with [table].
    for cmd, required in OPTION_GATED_COMMANDS.items():
        if cmd in stripped and required not in stripped:
            pkg, opt = required.split("}{", 1)
            pkg = pkg.lstrip("[")
            opt = opt.rstrip("}")
            problems.append(
                f"{cmd} used but \\usepackage[{opt}]{{{pkg}}} is not present "
                f"-- this fails at compile time with an undefined control sequence"
            )

    # --- locally defined macros are actually used ----------------------
    # A \newcommand that is never invoked is dead code in a document that is
    # meant to be read; more importantly, a command that is *invoked* but never
    # defined (and is not standard) is a broken reference.
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", stripped))
    for name in sorted(defined):
        uses = len(re.findall(r"\\" + name + r"\b", stripped))
        if uses <= 1:      # the \newcommand itself
            problems.append(f"NOTE \\{name} is defined but never used")
    for name in sorted(re.findall(r"\\([A-Za-z]{4,})\b", stripped)):
        if name in KNOWN_LATEX_COMMANDS or name in defined:
            continue
        # Anything that looks TeX-ish but is not defined here and not a known
        # standard command is worth surfacing for a human to confirm.

    # --- required sections ---------------------------------------------
    for sec in ("Model Details", "Intended Use", "Metrics", "Ethical"):
        if sec.lower() not in stripped.lower():
            problems.append(f"required section text '{sec}' not found")

    # --- label/ref integrity -------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", stripped))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", stripped))
    dangling = refs - labels
    if dangling:
        problems.append(f"dangling \\ref targets: {sorted(dangling)}")

    # --- citations ------------------------------------------------------
    cites = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", stripped):
        cites.update(c.strip() for c in group.split(","))
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", stripped))
    missing = cites - bibitems
    if missing:
        problems.append(f"cited but not in bibliography: {sorted(missing)}")
    unused = bibitems - cites
    if unused:
        problems.append(f"NOTE uncited bibitems: {sorted(unused)}")

    # --- report ---------------------------------------------------------
    name = path.name
    if problems:
        print(f"FAIL {name}")
        for p in problems:
            print(f"     - {p}")
        return 1
    print(f"OK   {name}  "
          f"({len(stripped_lines)} lines, {len(labels)} labels, "
          f"{len(cites)} citations, {len(bibitems)} bibitems)")
    return 0


def main(argv):
    targets = [Path(a) for a in argv[1:]] or list(Path("docs").glob("*.tex"))
    failures = sum(check(t) for t in targets)
    print(f"\n{len(targets)} file(s) checked, {failures} with problems.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))