"""`fz-chat` — ask the play database questions in natural language (hosted Claude).

    fz-chat "how often did KC run 11 personnel on third down?"
    fz-chat            # interactive REPL

Requires ANTHROPIC_API_KEY (the only hosted dependency; the CV pipeline stays self-hosted).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chat with the play database (hosted Claude).")
    p.add_argument("question", nargs="*", help="your question; omit for an interactive REPL")
    p.add_argument("--data", default="data", help="data root containing pbp/*.parquet")
    args = p.parse_args(argv)

    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("Chat needs the chat extra:  pip install -e '.[chat]'", file=sys.stderr)
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to use chat. Only the chat brain is hosted; the CV stays "
              "self-hosted.", file=sys.stderr)
        return 1

    from formation_zero.chat.agent import chat
    from formation_zero.data.store import load_plays

    con = load_plays(args.data)
    n = con.execute("SELECT COUNT(*) FROM plays").fetchone()[0]

    def ask(q: str) -> None:
        try:
            res = chat(q, con)
        except anthropic.APIError as e:  # noqa: F821 (imported above)
            print(f"[API error: {e}]", file=sys.stderr)
            return
        print("\n" + (res["answer"] or "(no answer)") + "\n")

    if args.question:
        ask(" ".join(args.question))
        return 0

    print(f"formation_zero chat — {n} plays loaded. Ask about formations / personnel / situations. "
          "Ctrl-D to exit.")
    while True:
        try:
            q = input("» ").strip()
        except EOFError:
            print()
            break
        if q:
            ask(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
