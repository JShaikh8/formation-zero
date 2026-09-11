# Formation Zero (package: gridiron)

NFL **All-22** coaches-film breakdown: computer-vision perception + rule-based football
intelligence + a chat interface ("show me all the Cover 3 defenses").

- **Blueprint:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Progress tracker (public site):** https://formation-zero.onrender.com — generated from
  [docs/](docs/) by `make site`; see [docs/TRACKING.md](docs/TRACKING.md) for how to update it.
- **Decision log:** [docs/DECISIONS.md](docs/DECISIONS.md) · **Journal:** [docs/journal/](docs/journal/)

This package holds all the **football logic**. It *imports* [`roboflow/sports`](https://github.com/roboflow/sports)
as CV scaffolding (detection, tracking, team-split, the `ViewTransformer` homography engine);
everything football-specific lives here.

See the full build plan at `~/.claude/plans/hello-i-want-to-spicy-quiche.md`.

## Architecture (5 layers)

1. **Perception** (`gridiron/perception/`) — video → tracking table (player x/y in field yards).
2. **Intelligence** (`gridiron/intelligence/`) — tracking → roles, formation, personnel, technique, coverage.
3. **Storage** (`gridiron/data/`) — DuckDB + Parquet, `plays.parquet` master index.
4. **Chat** (`gridiron/chat/`) — text-to-SQL over a constrained semantic layer (hosted Claude).
5. **UX** — Next.js film room (separate app, built last).

## Strategy

The intelligence + chat layers are built and validated on **Big Data Bowl** tracking data first
(same schema as our CV output), so we get a working demo before the hard CV is done. **Play-by-play
data** (`nfl_data_py`) supplies down/distance/line-of-scrimmage and personnel — which also anchors
field registration (the hardest CV piece) to a known yard line.

## Quickstart (Week 0)

```bash
cd ~/projects/gridiron
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # core (PBP + field model + storage)
# pip install -e '.[cv,dev]'  # add the CV stack when you start perception

# Pull play-by-play for one game (canonical play list + labels/alignment backbone):
gridiron-pull-pbp --season 2024 --week 5 --away BUF --home NYJ --out data
```

That writes `data/pbp/2024_wk05_BUF-NYJ.parquet` — down, distance, line-of-scrimmage,
personnel, and a `play_index` you align your segmented film clips against.

```bash
# Chat with the data (hosted Claude; the CV stays self-hosted):
pip install -e '.[chat]'
export ANTHROPIC_API_KEY=sk-ant-...
gridiron-chat "how often did KC run 11 personnel on third down?"
gridiron-chat "which personnel grouping was most efficient?"   # EPA/play, success, explosive
gridiron-chat            # interactive REPL
```

The chat brain (`claude-opus-4-8`) translates football questions into two constrained tools —
`query_plays` (filter/retrieve) and `aggregate_plays` (efficiency: EPA/play, success rate,
explosive rate) — and never writes raw SQL; everything runs deterministically over DuckDB.
Note: pre-snap motion and play-action aren't in the play-by-play data — they arrive with the CV layer.

## Data layout

```
data/
  raw/<season>/wk<NN>/<AWAY>_at_<HOME>/
    source/            # continuous coaches-film file(s), immutable
    sideline/  pNNN.mp4
    endzone/   pNNN.mp4
  pbp/<game_key>.parquet
  derived/
    tracking/<play_uid>.parquet
    labels/<play_uid>.json
  plays.parquet        # master index linking both angles + PBP per play_uid
```

`play_uid` = `2024_wk05_BUF-NYJ_p017` — deterministic, sortable, joins everywhere.
