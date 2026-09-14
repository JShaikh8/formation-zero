"""Compile the tracker's sources under docs/ into site/data/content.js.

Standard library only, so it runs anywhere Python 3 exists (the laptop, Render's build step,
a CI runner). Nothing here needs the project's own package.

Sources
-------
    docs/tracker/status.json       where we are, plain-English summary, next up, key numbers
    docs/tracker/milestones.json   epics and stories, mirrored to Jira keys
    docs/tracker/tech.json         the technology list with statuses
    docs/tracker/costs.json        cost model, fixed costs, ledger, policy
    docs/tracker/gallery.json      demo images and placeholders
    docs/DECISIONS.md              the decision log (parsed into entries)
    docs/GLOSSARY.md               the glossary (parsed into terms)
    docs/journal/YYYY-MM-DD.md     one entry per work day, with front matter

Output
------
    site/data/content.js           `window.FZ = {...}` consumed by site/app.js
    --preview PATH                 optionally, a body-only copy of index.html for the
                                   Artifact preview (the page shell minus the document wrapper)

Usage:  python3 site/build.py [--preview /path/to/preview.html]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE = ROOT / "site"


# ------------------------------------------------------------------ markdown --------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\*)(.+?)(?<!\*)\*(?![*\w])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def inline(s: str) -> str:
    """Escape HTML, then apply inline markdown: code, bold, italic, links."""
    s = html.escape(s, quote=False)
    s = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITALIC.sub(r"<em>\1</em>", s)
    s = _LINK.sub(r'<a href="\2">\1</a>', s)
    return s


def markdown(text: str) -> str:
    """A small, predictable markdown subset -> HTML.

    Supports: headings, paragraphs, bullet and numbered lists (one level, with continuation
    lines), blockquotes, fenced code, tables, horizontal rules, and the inline set above. It is
    deliberately not a full parser: the docs are written to this subset.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    def flush_para(buf: list[str]) -> None:
        if buf:
            out.append(f"<p>{inline(' '.join(x.strip() for x in buf))}</p>")
            buf.clear()

    para: list[str] = []
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para(para)
            lang = stripped[3:].strip()
            code: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        if not stripped:
            flush_para(para)
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para(para)
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,})$", stripped):
            flush_para(para)
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para(para)
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        if stripped.startswith("|"):
            flush_para(para)
            rows: list[str] = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            body_start = 1
            if len(cells) > 1 and all(re.match(r"^:?-{2,}:?$", c) for c in cells[1]):
                body_start = 2
            head = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in cells[body_start:]
            )
            out.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                       f"<tbody>{body}</tbody></table></div>")
            continue

        m_ul = re.match(r"^[-*]\s+(.*)$", stripped)
        m_ol = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m_ul or m_ol:
            flush_para(para)
            tag = "ul" if m_ul else "ol"
            pattern = r"^[-*]\s+(.*)$" if m_ul else r"^\d+[.)]\s+(.*)$"
            items: list[str] = []
            while i < n:
                cur = lines[i].strip()
                mm = re.match(pattern, cur)
                if mm:
                    items.append(mm.group(1))
                    i += 1
                elif cur and lines[i].startswith((" ", "\t")) and items:
                    items[-1] += " " + cur          # continuation line
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        para.append(line)
        i += 1

    flush_para(para)
    return "\n".join(out)


# ------------------------------------------------------------------ sources ---------------

def read_json(name: str) -> dict:
    return json.loads((DOCS / "tracker" / name).read_text())


def parse_front_matter(text: str) -> tuple[dict, str]:
    """`---` fenced `key: value` lines at the top of a file. No YAML library needed."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta: dict = {}
    for raw in text[3:end].strip().split("\n"):
        if ":" in raw:
            k, v = raw.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[end + 4:].lstrip("\n")


def load_journal() -> list[dict]:
    entries = []
    for path in sorted((DOCS / "journal").glob("*.md")):
        meta, body = parse_front_matter(path.read_text())
        date = meta.get("date") or path.stem
        sessions = len(re.findall(r"^##\s+Session", body, flags=re.M))
        entries.append({
            "date": date,
            "title": meta.get("title", path.stem),
            "milestone": meta.get("milestone", ""),
            "hours": float(meta["hours"]) if meta.get("hours") else None,
            "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
            "sessions": sessions,
            "html": markdown(body),
            "file": f"docs/journal/{path.name}",
        })
    entries.sort(key=lambda e: e["date"])
    return entries


def load_decisions() -> list[dict]:
    text = (DOCS / "DECISIONS.md").read_text()
    chunks = re.split(r"^##\s+", text, flags=re.M)[1:]
    decisions = []
    for chunk in chunks:
        head, _, rest = chunk.partition("\n")
        m = re.match(r"^(D-\d+)\s+[—-]\s+(.*)$", head.strip())
        if not m:
            continue
        meta: dict = {}
        body_lines: list[str] = []
        for raw in rest.split("\n"):
            mm = re.match(r"^-\s+([a-z_]+):\s*(.*)$", raw.strip())
            if mm and not body_lines:
                meta[mm.group(1)] = mm.group(2).strip()
            else:
                body_lines.append(raw)
        decisions.append({
            "id": m.group(1),
            "title": m.group(2).strip(),
            "date": meta.get("date", ""),
            "status": meta.get("status", "accepted"),
            "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
            "supersedes": meta.get("supersedes", ""),
            "html": markdown("\n".join(body_lines).strip()),
        })
    return decisions


def load_glossary() -> list[dict]:
    text = (DOCS / "GLOSSARY.md").read_text()
    chunks = re.split(r"^###\s+", text, flags=re.M)[1:]
    terms = []
    for chunk in chunks:
        term, _, rest = chunk.partition("\n")
        terms.append({"term": term.strip(), "html": markdown(rest.strip())})
    return terms


# ------------------------------------------------------------------ output ----------------

def build_content() -> dict:
    return {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "status": read_json("status.json"),
        "now": read_json("now.json") if (DOCS / "tracker" / "now.json").exists() else None,
        "milestones": read_json("milestones.json"),
        "tech": read_json("tech.json"),
        "costs": read_json("costs.json"),
        "gallery": read_json("gallery.json"),
        "decisions": load_decisions(),
        "glossary": load_glossary(),
        "journal": load_journal(),
    }


def write_content_js(content: dict) -> Path:
    out = SITE / "data" / "content.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(content, ensure_ascii=False, indent=1)
    # A "</script>" inside a string would end the script tag early. Neutralise it.
    payload = payload.replace("</", "<\\/")
    out.write_text("// Generated by site/build.py from docs/. Do not edit by hand.\n"
                   f"window.FZ = {payload};\n")
    return out


def write_preview(path: Path) -> None:
    """Body-only copy of index.html for the Artifact preview: title + head links + body."""
    src = (SITE / "index.html").read_text()
    head = re.search(r"<head>(.*?)</head>", src, flags=re.S).group(1)
    body = re.search(r"<body[^>]*>(.*?)</body>", src, flags=re.S).group(1)
    keep = "\n".join(
        line for line in head.split("\n")
        if "<title>" in line or "fonts.googleapis" in line or "fonts.gstatic" in line
        or 'rel="stylesheet"' in line
    )
    path.write_text(keep + "\n" + body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preview", help="also write a body-only preview page to this path")
    args = parser.parse_args(argv)

    content = build_content()
    out = write_content_js(content)
    j, d, g = content["journal"], content["decisions"], content["glossary"]
    print(f"{out.relative_to(ROOT)}: {len(j)} journal days, {len(d)} decisions, "
          f"{len(g)} glossary terms, "
          f"{sum(len(m['stories']) for m in content['milestones']['milestones'])} stories")
    if args.preview:
        write_preview(Path(args.preview))
        print(f"preview -> {args.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
