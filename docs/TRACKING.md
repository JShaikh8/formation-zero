# How the tracker works

The public progress site at https://formation-zero.onrender.com is generated from files in this
repository. There is no database and no CMS. Edit a file, rebuild, commit, push; Render redeploys.

## Sources (edit these)

| File | What it drives | Format |
|---|---|---|
| `docs/tracker/status.json` | The "Now" page: phase, plain-English summary, next up, key numbers, project name and links | JSON |
| `docs/tracker/milestones.json` | The "Plan" page and the Drive graphic. Epics (F0, M0 to M8) with `progress` 0 to 100 and stories with Jira-style keys `GRD-###` | JSON |
| `docs/tracker/tech.json` | The "Tech" page. Groups of tools with a status from the legend | JSON |
| `docs/tracker/costs.json` | The "Costs" page: cap, scenarios, model prices, fixed costs, ledger, future options | JSON |
| `docs/tracker/gallery.json` | The "Gallery" page. `kind` is `image` (with `src` under `site/`) or `placeholder` | JSON |
| `docs/DECISIONS.md` | The "Decisions" page. One `## D-NNN — Title` per decision with `- key: value` metadata lines | Markdown |
| `docs/GLOSSARY.md` | The "Glossary" page. One `### Term` per entry | Markdown |
| `docs/journal/YYYY-MM-DD.md` | The "Journey" page and calendar. One file per work day with front matter | Markdown |

## Build and deploy

```bash
make site          # = python3 site/build.py  (stdlib only, any Python 3)
make assets        # regenerate the gallery SVGs from real data (needs the project venv)
```

`site/build.py` writes `site/data/content.js`. That file is committed, so Render needs no build
step: the static site is the `site/` folder as-is. `render.yaml` at the repo root declares it.

Rhythm at the end of a work session:

1. Add or extend today's `docs/journal/YYYY-MM-DD.md` (one `## Session N — title` per session).
2. Update `progress` and story statuses in `milestones.json`; update `next_up`, `numbers` and
   `updated` in `status.json`.
3. Add a decision to `DECISIONS.md` if one was made. Add a ledger row to `costs.json` if money
   was spent.
4. `make site`, commit everything (sources and the generated `content.js`), push.

## Conventions

- **Milestone ids** are stable. Ten of them make the hundred-yard Drive; each is ten yards. The
  ball sits at the sum of `progress` across milestones.
- **Story keys** `GRD-###` never change or get reused. They are the join to Jira: create the
  Jira issue with the same key in its summary or as a label.
- **Statuses**: stories are `done`, `active`, `todo`, `parked`. Tech is `in_use`, `planned`,
  `evaluated`, `rejected`, `parked`. Decisions are `accepted`, `superseded`, `revisit`.
- **Dates** are ISO `YYYY-MM-DD` everywhere.
- **Costs**: estimates live in `scenarios`; real spend goes only in `ledger`, from logged usage.
  If a cap changes, that is a decision with the `cost` tag, so it shows on the Costs page.
- **Journal front matter** keys: `date`, `title`, `milestone`, `hours`, `tags` (comma-separated).

## Markdown subset

`build.py` converts headings, paragraphs, bullet and numbered lists (one level, continuation
lines indented), blockquotes, fenced code, pipe tables, horizontal rules, and inline code, bold,
italic and links. Keep the docs to that subset.

## Preview without deploying

```bash
python3 -m http.server 8000 --directory site
```

Then open http://localhost:8000.
