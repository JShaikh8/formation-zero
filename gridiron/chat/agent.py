"""The chat brain: hosted Claude translates football questions into structured tool calls.

This is the only hosted piece of the system (the CV stays self-hosted). It uses the official
Anthropic SDK with a manual tool-use loop and two constrained tools (filter + aggregate), so
Claude never writes raw SQL — it picks structured filters/dimensions and we run them deterministically.
"""

from __future__ import annotations

import json

from gridiron.chat.query import (
    AGG_ARGS,
    AGGREGATE_PLAYS_TOOL,
    QUERY_ARGS,
    QUERY_PLAYS_TOOL,
    run_aggregate_plays,
    run_query_plays,
)

MODEL = "claude-opus-4-8"

# name -> (function, allowed-arg whitelist)
_TOOLS = {
    "query_plays": (run_query_plays, QUERY_ARGS),
    "aggregate_plays": (run_aggregate_plays, AGG_ARGS),
}

SYSTEM = """You are a film-room assistant for NFL coaches-film analysis. You answer questions about
plays in a single-game database by calling tools, then summarizing what they return. Never invent
numbers — report only what the tools return.

Tools:
- query_plays: filter plays; return rows or a count (count_only=true for "how many / how often").
- aggregate_plays: group plays and compute efficiency — n, epa_per_play, success_rate (share with
  EPA > 0), yards_per_play, explosive_rate (share gaining 15+ yds). Use this for "which X was most
  efficient / most productive / best", trends, and comparisons. Pick group_by (e.g. ["off_grouping"],
  ["off_formation"], ["play_type","down"]) and sort_by (default epa_per_play). EPA per play is the
  standard efficiency metric.

Translate football language into filters:
- "11 personnel" -> off_grouping "11"; "12"/"21" similarly (1st digit = #backs, 2nd = #TEs).
- formations -> off_formation UPPERCASE (SHOTGUN, UNDER CENTER, PISTOL, EMPTY, I_FORM, JUMBO, WILDCAT).
- "nickel/dime/base/quarter" -> def_package.
- "third and long" -> down 3 + distance_min 7; "third and short" -> down 3 + distance_max 2.
- "red zone" -> red_zone true; "goal to go" -> goal_to_go true.
- "runs/rushing" -> play_type "run"; "passes/dropbacks" -> play_type "pass".
- "shotgun" / "no huddle" -> shotgun / no_huddle true.
- "explosive plays" -> explosive_rate (aggregate) or count plays with first_down for conversions.
- "to the left/middle/right" -> pass_location / run_location; "deep/short" -> pass_length;
  "off guard/tackle/end" -> run_gap. sacks/INTs/TDs/first downs are boolean filters.
- teams look like KC, DET, BUF, NYJ; posteam = offense, defteam = defense.

Data note: this database is play-by-play, so PRE-SNAP MOTION and PLAY-ACTION are NOT available yet
(they need the computer-vision layer). If asked, say so plainly rather than guessing.

Keep answers concise; cite a few example play_uids or the top group(s)."""


def chat(question: str, con, *, client=None, model: str = MODEL, max_steps: int = 5) -> dict:
    """Answer one question against the loaded `plays` table. Returns {answer, plays}."""
    import anthropic

    client = client or anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]
    found: list[dict] = []

    for _ in range(max_steps):
        resp = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM,
            tools=[QUERY_PLAYS_TOOL, AGGREGATE_PLAYS_TOOL], messages=messages,
        )
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text.strip(), "plays": found}

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name in _TOOLS:
                fn, allowed = _TOOLS[block.name]
                kwargs = {k: v for k, v in dict(block.input).items() if k in allowed}
                out = fn(con, **kwargs)
                found.extend(out.get("plays", []))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, default=str),
                })
        messages.append({"role": "user", "content": results})

    return {"answer": "(stopped after the tool-call limit; try a narrower question)", "plays": found}
