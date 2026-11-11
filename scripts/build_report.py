"""Build reports/baseline.html from reports/baseline.template.html.

    uv run python scripts/build_report.py

The template holds the text and chart code; results/report_data.json holds the
data the charts read (D: baseline replies and error analysis, R: per-round
results for every arm and seed, CMP: paired comparisons). They replace the
/*DATA*/, /*RES*/ and /*CMP*/ placeholders.

/*BUILD*/ in the footer is replaced by the current git commit (short hash) and
its date. If tracked or untracked files other than the built page differ from
that commit, it says so, so the footer never claims a commit the page wasn't
built from. To stamp a final hash: commit everything, run this script, and
commit the rebuilt page on its own.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "reports/baseline.template.html"
OUT = ROOT / "reports/baseline.html"
DATA = ROOT / "results/report_data.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build_stamp() -> str:
    try:
        commit = git("rev-parse", "--short", "HEAD")
        date = git("log", "-1", "--format=%cd", "--date=format:%-d %b %Y")
        dirty = [l for l in git("status", "--porcelain").splitlines() if not l.endswith(str(OUT.relative_to(ROOT)))]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Built outside a git checkout."
    note = " plus uncommitted changes" if dirty else ""
    return f"Built from commit <code>{commit}</code> ({date}){note}."


def main() -> None:
    data = json.loads(DATA.read_text())
    html = TEMPLATE.read_text()
    for key, mark in (("D", "/*DATA*/null"), ("R", "/*RES*/null"), ("CMP", "/*CMP*/null")):
        assert html.count(mark) == 1, mark
        html = html.replace(mark, json.dumps(data[key], separators=(",", ":")))
    assert html.count("/*BUILD*/") == 1
    html = html.replace("/*BUILD*/", build_stamp())
    OUT.write_text(html)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
