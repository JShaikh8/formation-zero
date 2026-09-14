"""Log a baby step to the tracker and push it, so the site moves while work happens.

    python3 scripts/step.py "Ported the NFL API token client"      # start a step (previous active -> done)
    python3 scripts/step.py --done                                  # finish the active step
    python3 scripts/step.py --focus "Milestone 1: ingestion"        # change the headline
    python3 scripts/step.py --clear                                 # drop done steps (start of a new day)
    add --no-push to any of these to skip the commit and push.

Stdlib only. Edits docs/tracker/now.json, rebuilds site/data/content.js, commits, pushes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOW = ROOT / "docs" / "tracker" / "now.json"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("text", nargs="?", help="start this step (marks the current active step done)")
    ap.add_argument("--done", action="store_true", help="mark the active step done")
    ap.add_argument("--focus", help="set the headline")
    ap.add_argument("--clear", action="store_true", help="drop done steps")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args(argv)

    data = json.loads(NOW.read_text())
    steps = data.setdefault("steps", [])
    if a.clear:
        data["steps"] = steps = [s for s in steps if s.get("status") != "done"]
        data["since"] = stamp()[:10]
    if a.focus:
        data["focus"] = a.focus
    if a.done or a.text:
        for s in steps:
            if s.get("status") == "active":
                s["status"] = "done"
                s["at"] = stamp()
    if a.text:
        # If the text matches a queued todo, activate that one instead of adding a duplicate.
        match = next((s for s in steps if s.get("status") == "todo" and s["text"].lower() == a.text.lower()), None)
        if match:
            match["status"] = "active"
            match["at"] = stamp()
        else:
            steps.append({"text": a.text, "status": "active", "at": stamp()})
    NOW.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    subprocess.run([sys.executable, str(ROOT / "site" / "build.py")], check=True, cwd=ROOT)
    if a.no_push:
        print("updated now.json and rebuilt (not pushed)")
        return 0
    msg = a.text or ("Step done" if a.done else "Tracker: update the Right now block")
    subprocess.run(["git", "add", "docs/tracker/now.json", "site/data/content.js"], check=True, cwd=ROOT)
    subprocess.run(["git", "commit", "-q", "-m", f"Step: {msg}"], check=True, cwd=ROOT)
    subprocess.run(["git", "push", "-q", "origin", "main"], check=True, cwd=ROOT)
    print(f"pushed: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
