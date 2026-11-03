"""Fail if `oztest` results differ from the known failures.

Cases in `known_lax_failures.txt` fail by default, but pass with `--strict`.  Cases in
`known_failures.txt` fail in both modes.  Each case not listed is expected to pass, so
this exits nonzero for new failures, and for listed cases that now pass or fail
differently (update the list).  `oztest run` itself exits 0 regardless of the results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

HERE = Path(__file__).parent
DINGUS = HERE / "dingus.py"


def _known_failures(strict: bool) -> dict[str, str]:
    """Return `{slug: status}` for the cases expected to fail in this mode."""
    names = ["known_failures.txt"]
    if not strict:
        names.append("known_lax_failures.txt")
    known = {}
    for name in names:
        for line in (HERE / name).read_text().splitlines():
            if entry := line.partition("#")[0].split():
                status, slug = entry
                known[slug] = status
    return known


def _reason(slug: str, strict: bool) -> str:
    """Re-run the dingus on a case, as `oztest` drops its output from the results."""
    path = Path(str(files("oztest.cases"))) / slug
    case = next(path.parent.glob(f"{path.name}.*"))
    cmd = [sys.executable, str(DINGUS), *(["--strict"] if strict else []), str(case)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        return proc.stderr.strip().splitlines()[-1]
    return json.loads(proc.stdout).get("message", "accepted as valid")[:500]


def main(result_files: list[str], strict: bool = False) -> int:
    known = _known_failures(strict)

    results = {}
    for file in result_files:
        data = json.loads(Path(file).read_text())
        results.update({r["slug"]: r["result"] for r in data["results"]})

    passed = sum(result == "pass" for result in results.values())
    mode = "strict" if strict else "default"
    print(f"[{mode}] {passed}/{len(results)} passed, {len(known)} known failures")

    unexpected = 0
    for slug in sorted(known | results):
        expected = known.get(slug, "pass")
        actual = results.get(slug, "missing")
        if actual != expected:
            print(f"UNEXPECTED: {slug}: expected {expected}, got {actual}")
            if actual in ("fail", "error"):
                print(f"    {_reason(slug, strict)}")
            unexpected += 1
    return 1 if unexpected else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.results, args.strict))
