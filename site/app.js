/* Formation Zero tracker. Renders window.FZ (built from docs/ by site/build.py). */
(function () {
  'use strict';
  const FZ = window.FZ;
  if (!FZ) { document.getElementById('main').textContent = 'Content not built. Run: python3 site/build.py'; return; }

  const $ = (sel, root) => (root || document).querySelector(sel);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const STATUS_WORD = { done: 'Done', active: 'In progress', todo: 'To do', parked: 'Parked', in_use: 'In use', planned: 'Planned', evaluated: 'Evaluated', rejected: 'Rejected', accepted: 'Accepted', superseded: 'Superseded', revisit: 'Revisit' };
  const pill = (s) => `<span class="pill ${esc(s)}">${esc(STATUS_WORD[s] || s)}</span>`;
  const fmtDate = (iso) => { const d = new Date(iso + 'T12:00:00'); return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' }); };
  const money = (n) => (n == null ? '—' : '$' + Number(n).toLocaleString('en-US', { maximumFractionDigits: 2 }));
  const msById = Object.fromEntries(FZ.milestones.milestones.map((m) => [m.id, m]));

  /* ---------------- the drive: progress as a football field ---------------- */
  function driveStats() {
    const ms = FZ.milestones.milestones;
    const per = 100 / ms.length;
    let yards = 0;
    ms.forEach((m) => { yards += per * (Number(m.progress) || 0) / 100; });
    yards = Math.min(100, Math.round(yards * 10) / 10);
    const idx = Math.min(ms.length - 1, Math.floor(yards / per));
    const firstDown = Math.min(100, (idx + 1) * per);
    return { yards, firstDown, per, current: ms[idx] };
  }

  function driveSVG() {
    const ms = FZ.milestones.milestones;
    const { yards, firstDown, per } = driveStats();
    const W = 1000, H = 268, padL = 60, padR = 60, top = 24, fieldH = 130;
    const x = (yd) => padL + (yd / 100) * (W - padL - padR);
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Progress drawn as a football field: ${yards} yards gained of 100">`;
    s += `<rect class="turf-ez" x="${padL - 44}" y="${top}" width="44" height="${fieldH}"/>`;
    s += `<rect class="turf-ez" x="${x(100)}" y="${top}" width="44" height="${fieldH}"/>`;
    s += `<rect class="turf" x="${padL}" y="${top}" width="${x(100) - padL}" height="${fieldH}"/>`;
    for (let yd = 0; yd <= 100; yd += 5) {
      const heavy = yd === 0 || yd === 100;
      s += `<line class="chalk" x1="${x(yd)}" y1="${top}" x2="${x(yd)}" y2="${top + fieldH}" stroke-width="${heavy ? 2.5 : 1}"/>`;
    }
    for (let yd = 10; yd <= 90; yd += 10) {
      const n = yd <= 50 ? yd : 100 - yd;
      s += `<text class="num" x="${x(yd)}" y="${top + fieldH - 14}" text-anchor="middle">${n}</text>`;
    }
    // Milestone markers at the end of each ten-yard chunk.
    ms.forEach((m, i) => {
      const xx = x((i + 1) * per);
      const cls = m.status === 'done' ? 'ms-done' : m.status === 'active' ? 'ms-active' : 'ms-todo';
      const y0 = top + fieldH + 18;
      // Labels alternate between two rows so ten of them fit without colliding.
      const row = i % 2 === 0 ? 0 : 1;
      const ly = y0 + 22 + row * 30;
      s += `<circle class="${cls}" cx="${xx}" cy="${y0}" r="6" stroke-width="1.5"/>`;
      if (row === 1) s += `<line class="chalk" x1="${xx}" y1="${y0 + 8}" x2="${xx}" y2="${ly - 12}" stroke-width="1" opacity="0.5"/>`;
      s += `<text class="ms-id" x="${xx}" y="${ly}" text-anchor="middle">${esc(m.id)}</text>`;
      const label = m.title.length > 20 ? m.title.slice(0, 19).replace(/\s+\S*$/, '') + '…' : m.title;
      s += `<text class="ms-label" x="${xx}" y="${ly + 15}" text-anchor="middle">${esc(label)}</text>`;
    });
    // First-down line (next milestone boundary), then the line of scrimmage (where we are).
    s += `<line class="first-line" x1="${x(firstDown)}" y1="${top - 6}" x2="${x(firstDown)}" y2="${top + fieldH + 6}" stroke-width="4"/>`;
    s += `<line class="los-line" x1="${x(yards)}" y1="${top - 6}" x2="${x(yards)}" y2="${top + fieldH + 6}" stroke-width="4"/>`;
    // The ball.
    s += `<ellipse class="ball" cx="${x(yards)}" cy="${top + fieldH / 2}" rx="11" ry="7" stroke-width="1.5"/>`;
    s += `<line class="chalk" x1="${x(yards) - 4}" y1="${top + fieldH / 2}" x2="${x(yards) + 4}" y2="${top + fieldH / 2}" stroke-width="1.5"/>`;
    // Callouts above the field.
    const losLabel = `${yards} yd · where we are`;
    const fdLabel = `1st down at ${firstDown}`;
    const losW = losLabel.length * 6.6 + 14, fdW = fdLabel.length * 6.6 + 14;
    let losX = x(yards) - losW / 2, fdX = x(firstDown) - fdW / 2;
    if (fdX < losX + losW + 8) fdX = losX + losW + 8;
    if (fdX + fdW > W - 8) { fdX = W - 8 - fdW; losX = Math.min(losX, fdX - losW - 8); }
    s += `<rect class="callout-bg los" x="${losX}" y="2" width="${losW}" height="17" rx="3"/>`;
    s += `<text class="callout los-text" x="${losX + losW / 2}" y="14" text-anchor="middle">${esc(losLabel)}</text>`;
    s += `<rect class="callout-bg first" x="${fdX}" y="2" width="${fdW}" height="17" rx="3"/>`;
    s += `<text class="callout first" x="${fdX + fdW / 2}" y="14" text-anchor="middle">${esc(fdLabel)}</text>`;
    s += '</svg>';
    return s;
  }

  /* ---------------- pages ---------------- */
  function renderNow() {
    const st = FZ.status;
    const latest = FZ.journal[FZ.journal.length - 1];
    const { yards, current } = driveStats();
    const firstPara = (html) => {
      const m = /<p>([\s\S]*?)<\/p>/.exec(html);
      if (m) return m[1];
      const text = html.replace(/<h\d>[\s\S]*?<\/h\d>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
      return text.length > 220 ? text.slice(0, 219).replace(/\s+\S*$/, '') + '…' : text;
    };
    return `
      <div class="hero">
        <div class="eyebrow">Where we are · updated ${esc(st.updated)}</div>
        <h1>${esc(st.phase.name)}</h1>
        <p class="lede">${esc(st.phase.plain)}</p>
      </div>
      <div class="drive">
        <div class="drive-head">
          <div><span class="eyebrow">The drive</span> &nbsp;<span class="mono">${yards} of 100 yards · ${esc(current.id)} ${esc(current.title)}</span></div>
          <div class="drive-legend"><span class="l-los">line of scrimmage: where we are</span><span class="l-first">first down: next milestone</span></div>
        </div>
        <div class="drive-scroll">${driveSVG()}</div>
      </div>
      <div class="now-grid">
        <div class="stack">
          <div>
            <h2>What this project is</h2>
            <div class="prose measure" style="margin-top:10px">${st.what_it_is.map((p) => `<p>${esc(p)}</p>`).join('')}</div>
          </div>
          ${st.end_state ? `<div>
            <h2>${esc(st.end_state.title)}</h2>
            <ol class="steps" style="margin-top:12px">${st.end_state.steps.map((t) => `<li>${esc(t)}</li>`).join('')}</ol>
          </div>` : ''}
          <div class="latest">
            <div class="eyebrow">Latest from the journal · ${esc(latest.date)}</div>
            <h3 style="margin-top:6px">${esc(latest.title)}</h3>
            <p class="muted" style="margin-top:6px">${firstPara(latest.html)}</p>
            <p style="margin-top:8px"><a href="#journey">Read the whole journey →</a></p>
          </div>
        </div>
        <div class="stack">
          <div>
            <h2>Next up</h2>
            <ul class="next-list" style="margin-top:12px">${st.next_up.map((t) => `<li><span class="dot"></span><span>${esc(t)}</span></li>`).join('')}</ul>
          </div>
          <div>
            <h2>By the numbers</h2>
            <div class="tiles" style="margin-top:12px">${st.numbers.map((n) => `<div class="tile"><div class="val">${esc(n.value)}</div><div class="lab">${esc(n.label)}</div><div class="note">${esc(n.note || '')}</div></div>`).join('')}</div>
          </div>
        </div>
      </div>`;
  }

  function renderJourney() {
    const entries = FZ.journal.slice().reverse();
    const days = new Set(FZ.journal.map((e) => e.date));
    const hours = FZ.journal.reduce((a, e) => a + (e.hours || 0), 0);
    const sessions = FZ.journal.reduce((a, e) => a + (e.sessions || 0), 0);
    const first = FZ.journal[0].date, last = FZ.status.updated > FZ.journal[FZ.journal.length - 1].date ? FZ.status.updated : FZ.journal[FZ.journal.length - 1].date;
    const months = [];
    for (let d = new Date(first.slice(0, 7) + '-01T12:00:00'); d.toISOString().slice(0, 7) <= last.slice(0, 7); d.setMonth(d.getMonth() + 1)) months.push(new Date(d));
    const today = new Date().toISOString().slice(0, 10);
    const cal = months.map((m) => {
      const y = m.getFullYear(), mo = m.getMonth();
      const firstDow = new Date(y, mo, 1).getDay();
      const nDays = new Date(y, mo + 1, 0).getDate();
      let cells = '<span class="wd">S</span><span class="wd">M</span><span class="wd">T</span><span class="wd">W</span><span class="wd">T</span><span class="wd">F</span><span class="wd">S</span>';
      for (let i = 0; i < firstDow; i++) cells += '<span class="day empty"></span>';
      for (let dd = 1; dd <= nDays; dd++) {
        const iso = `${y}-${String(mo + 1).padStart(2, '0')}-${String(dd).padStart(2, '0')}`;
        const has = days.has(iso);
        const cls = 'day' + (has ? ' has' : '') + (iso === today ? ' today' : '');
        cells += has ? `<button class="${cls}" type="button" data-goto="entry-${iso}" title="${esc(fmtDate(iso))}">${dd}</button>` : `<span class="${cls}">${dd}</span>`;
      }
      return `<div class="month"><h4>${m.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</h4><div class="month-grid">${cells}</div></div>`;
    }).join('');
    const list = entries.map((e) => `
      <article class="entry" id="entry-${esc(e.date)}">
        <div class="entry-head">
          <div class="entry-date">${esc(fmtDate(e.date))}</div>
          <div class="entry-meta">
            ${e.milestone ? `<span class="chip">${esc(e.milestone)} · ${esc((msById[e.milestone] || {}).title || '')}</span>` : ''}
            ${e.hours ? `<span>${e.hours} h</span>` : ''}
            ${e.sessions ? `<span>${e.sessions} session${e.sessions === 1 ? '' : 's'}</span>` : ''}
            ${e.tags.map((t) => `<span class="chip">${esc(t)}</span>`).join('')}
          </div>
        </div>
        <h3 class="entry-title">${esc(e.title)}</h3>
        <div class="prose measure">${e.html}</div>
      </article>`).join('');
    return `
      <div class="section-head"><h2>The journey, day by day</h2><p>Every day work happened, what was done in each session, and what it bought. Green days on the calendar are work days; click one to jump to it.</p></div>
      <div class="journey-grid">
        <aside class="calendar">
          <div class="cal-stats">${days.size} work days · ${sessions} sessions · ${hours} hours logged</div>
          ${cal}
        </aside>
        <div class="entries">${list}</div>
      </div>`;
  }

  function renderPlan() {
    const ms = FZ.milestones.milestones;
    const all = ms.flatMap((m) => m.stories);
    const count = (s) => all.filter((x) => x.status === s).length;
    const blocks = ms.map((m) => `
      <section class="milestone ${esc(m.status)}" id="ms-${esc(m.id)}">
        <div class="ms-head">
          <div class="ms-key">${esc(m.id)}</div>
          <div><div class="ms-title">${esc(m.title)}</div><div class="bar" aria-label="${m.progress}% complete"><i style="width:${Number(m.progress) || 0}%"></i></div></div>
          ${pill(m.status)}
        </div>
        <div class="ms-body">
          <p class="measure">${esc(m.plain)}</p>
          <p class="proof"><strong>Proof it's done:</strong> ${esc(m.proof)}</p>
          <details ${m.status === 'active' ? 'open' : ''}>
            <summary>${m.stories.length} stories</summary>
            <div class="table-wrap"><table><thead><tr><th>Key</th><th>Story</th><th>Status</th></tr></thead><tbody>
              ${m.stories.map((s) => `<tr><td>${esc(s.key)}</td><td>${esc(s.title)}${s.note ? ` <span class="muted">— ${esc(s.note)}</span>` : ''}</td><td>${pill(s.status)}</td></tr>`).join('')}
            </tbody></table></div>
          </details>
        </div>
      </section>`).join('');
    return `
      <div class="section-head"><h2>The plan</h2><p>${esc(FZ.milestones.key_scheme)}</p></div>
      <div class="summary-strip"><span><b>${count('done')}</b> done</span><span><b>${count('active')}</b> in progress</span><span><b>${count('todo')}</b> to do</span><span><b>${count('parked')}</b> parked</span><span><b>${all.length}</b> stories in ${ms.length} milestones</span></div>
      <div class="stack" style="margin-top:20px">${blocks}</div>`;
  }

  function renderTech() {
    const t = FZ.tech;
    const legend = Object.entries(t.legend).map(([k, v]) => `<span>${pill(k)} ${esc(v)}</span>`).join('');
    const groups = t.groups.map((g) => `
      <div class="tech-group"><h3>${esc(g.name)}</h3>
        <div class="cards">${g.items.map((it) => `
          <div class="card ${esc(it.status)}">
            <div class="card-head"><span class="name">${esc(it.name)}</span>${pill(it.status)}</div>
            <div class="plain">${esc(it.plain)}</div>
            <div class="why"><strong>Why:</strong> ${esc(it.why)}</div>
            <div class="lic">${esc(it.license)}</div>
          </div>`).join('')}</div>
      </div>`).join('');
    return `<div class="section-head"><h2>Technology</h2><p>What each tool is, in one sentence, and why it was chosen over the alternatives.</p></div><div class="legend">${legend}</div>${groups}`;
  }

  function renderDecisions() {
    const ds = FZ.decisions.slice().reverse();
    const byId = Object.fromEntries(FZ.decisions.map((d) => [d.id, d]));
    const supersededBy = {};
    FZ.decisions.forEach((d) => { if (d.supersedes) supersededBy[d.supersedes] = d.id; });
    return `
      <div class="section-head"><h2>Decision log</h2><p>Every choice that shaped the project: what was decided, why, what else was considered, what it cost. Newest first. Superseded decisions stay, greyed, with a pointer to what replaced them.</p></div>
      <div class="stack">${ds.map((d) => `
        <article class="decision ${esc(d.status)}" id="${esc(d.id)}">
          <div class="decision-head"><span class="decision-id">${esc(d.id)}</span><h3>${esc(d.title)}</h3></div>
          <div class="meta">${pill(d.status)}<span>${esc(d.date)}</span>${d.tags.map((t) => `<span class="chip">${esc(t)}</span>`).join('')}
            ${d.supersedes ? `<span>supersedes <a href="#${esc(d.supersedes)}">${esc(d.supersedes)}</a></span>` : ''}
            ${supersededBy[d.id] ? `<span>superseded by <a href="#${esc(supersededBy[d.id])}">${esc(supersededBy[d.id])}</a></span>` : ''}
          </div>
          <div class="prose measure">${d.html}</div>
        </article>`).join('')}</div>`;
  }

  function renderCosts() {
    const c = FZ.costs;
    const costDecisions = FZ.decisions.filter((d) => d.tags.includes('cost'));
    const spent = c.ledger.reduce((a, r) => a + (Number(r.amount) || 0), 0);
    return `
      <div class="section-head"><h2>Costs</h2><p>${esc(c.policy.headline)}</p></div>
      <div class="cost-grid">
        <div class="cap"><span class="eyebrow">Hosted model cap per game</span><div class="big">${money(c.policy.cap_per_game)}<small> / game</small></div><p class="measure">${esc(c.policy.starting_policy)}</p><p class="muted measure" style="font-size:0.9rem">${esc(c.policy.rule)}</p></div>
        <div class="cap"><span class="eyebrow">Spent so far</span><div class="big">${money(spent)}</div><p class="measure">${c.ledger.length ? `${c.ledger.length} entries in the ledger.` : 'Nothing has been billed yet. The ledger below fills in as real usage is logged.'}</p></div>
      </div>
      <div class="stack" style="margin-top:28px">
        <div><h3>What a game could cost</h3><p class="muted measure" style="margin:6px 0 10px;font-size:0.9rem">${esc(c.assumptions)}</p>
          <div class="table-wrap"><table><thead><tr><th>Routing policy</th><th class="num">Requests / game</th><th class="num">Claude Opus 5</th><th class="num">Claude Sonnet 5</th></tr></thead><tbody>
            ${c.scenarios.map((s) => `<tr class="${s.recommended ? 'rec' : ''}"><td>${esc(s.name)}${s.recommended ? ' <span class="chip">starting policy</span>' : ''}</td><td class="num">~${s.requests}</td><td class="num">about ${money(s.opus5)}</td><td class="num">about ${money(s.sonnet5)}</td></tr>`).join('')}
          </tbody></table></div></div>
        <div><h3>Model prices used</h3>
          <div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Input $ / 1M tokens</th><th class="num">Output $ / 1M tokens</th><th class="num">Per request (est.)</th></tr></thead><tbody>
            ${c.models.map((m) => `<tr><td>${esc(m.name)}</td><td class="num">${money(m.input_per_m)}</td><td class="num">${money(m.output_per_m)}</td><td class="num">${money(m.per_request)}</td></tr>`).join('')}
          </tbody></table></div></div>
        <div><h3>Fixed costs</h3>
          <div class="table-wrap"><table><thead><tr><th>Item</th><th class="num">Cost</th><th>Note</th></tr></thead><tbody>
            ${c.fixed.map((f) => `<tr><td>${esc(f.item)}</td><td class="num">${f.amount == null ? '—' : money(f.amount)}</td><td class="muted">${esc(f.note)}</td></tr>`).join('')}
          </tbody></table></div></div>
        <div><h3>Ledger</h3>${c.ledger.length ? `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Item</th><th class="num">Amount</th><th>Note</th></tr></thead><tbody>${c.ledger.map((r) => `<tr><td class="mono">${esc(r.date)}</td><td>${esc(r.item)}</td><td class="num">${money(r.amount)}</td><td class="muted">${esc(r.note || '')}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty" style="margin-top:10px">No spend recorded yet.</div>'}</div>
        <div><h3>Options we have not paid for</h3>
          <div class="table-wrap"><table><thead><tr><th>Option</th><th>Estimate</th><th>When we would</th></tr></thead><tbody>
            ${c.future_options.map((o) => `<tr><td>${esc(o.item)}</td><td class="mono">${esc(o.estimate)}</td><td class="muted">${esc(o.when)}</td></tr>`).join('')}
          </tbody></table></div></div>
        <div><h3>Decisions that touched money</h3><p class="muted" style="margin:6px 0 10px;font-size:0.9rem">If a cap is ever raised or broken, the decision log records the new number, what it bought, and what it cost.</p>
          <div class="stack">${costDecisions.map((d) => `<article class="decision ${esc(d.status)}"><div class="decision-head"><span class="decision-id">${esc(d.id)}</span><h3>${esc(d.title)}</h3></div><div class="meta">${pill(d.status)}<span>${esc(d.date)}</span></div><div class="prose measure">${d.html}</div></article>`).join('') || '<div class="empty">None yet.</div>'}</div></div>
      </div>`;
  }

  function renderGallery() {
    const items = FZ.gallery.items.map((g) => g.kind === 'placeholder'
      ? `<figure class="figure placeholder"><div class="ph">${esc(g.milestone)} · coming</div><figcaption class="figcap"><div class="t">${esc(g.title)}</div><div class="c">${esc(g.caption)}</div></figcaption></figure>`
      : `<figure class="figure">${g.link ? `<a href="${esc(g.link)}">` : ''}<img src="${esc(g.src)}" alt="${esc(g.title)}" loading="lazy">${g.link ? '</a>' : ''}<figcaption class="figcap"><div class="t">${g.link ? `<a href="${esc(g.link)}">${esc(g.title)} →</a>` : esc(g.title)}</div><div class="c">${esc(g.caption)}</div><div class="chips"><span class="chip">${esc(g.milestone)}</span><span class="chip">${esc(g.date)}</span></div></figcaption></figure>`);
    return `<div class="section-head"><h2>Gallery</h2><p>Real outputs so far, and the demos each milestone will add. Everything here was generated from the project's own data.</p></div><div class="gallery">${items.join('')}</div>`;
  }

  function renderGlossary() {
    return `
      <div class="section-head"><h2>Glossary</h2><p>The words this project uses, in plain English.</p></div>
      <label class="eyebrow" for="glossary-search">Search</label>
      <input class="search" id="glossary-search" type="search" placeholder="Try: homography, 3-technique, teacher" autocomplete="off">
      <dl class="terms" id="terms">${FZ.glossary.map((t) => `<div class="term" data-term="${esc(t.term.toLowerCase())}"><dt>${esc(t.term)}</dt><dd>${t.html.replace(/^<p>|<\/p>$/g, '')}</dd></div>`).join('')}</dl>`;
  }

  /* ---------------- routing ---------------- */
  const renderers = { now: renderNow, journey: renderJourney, plan: renderPlan, tech: renderTech, decisions: renderDecisions, costs: renderCosts, gallery: renderGallery, glossary: renderGlossary };
  const rendered = {};
  const titles = { now: 'Now', journey: 'Journey', plan: 'Plan', tech: 'Tech', decisions: 'Decisions', costs: 'Costs', gallery: 'Gallery', glossary: 'Glossary' };

  function tabForHash(hash) {
    const h = (hash || '').replace(/^#/, '');
    if (renderers[h]) return { tab: h, anchor: null };
    if (/^D-\d+$/.test(h)) return { tab: 'decisions', anchor: h };
    if (/^entry-/.test(h)) return { tab: 'journey', anchor: h };
    if (/^ms-/.test(h)) return { tab: 'plan', anchor: h };
    return { tab: 'now', anchor: null };
  }

  function show(hash) {
    const { tab, anchor } = tabForHash(hash);
    document.querySelectorAll('.tab').forEach((el) => { el.hidden = true; });
    const el = document.getElementById('tab-' + tab);
    if (!rendered[tab]) { el.innerHTML = renderers[tab](); rendered[tab] = true; wire(tab, el); }
    el.hidden = false;
    document.querySelectorAll('.tabs a').forEach((a) => { if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    document.title = tab === 'now' ? 'Formation Zero' : `${titles[tab]} · Formation Zero`;
    if (anchor) { const target = document.getElementById(anchor); if (target) target.scrollIntoView({ block: 'start' }); }
    else if (hash) window.scrollTo({ top: 0 });
  }

  function wire(tab, el) {
    if (tab === 'journey') {
      el.addEventListener('click', (ev) => {
        const b = ev.target.closest('[data-goto]');
        if (b) { const t = document.getElementById(b.dataset.goto); if (t) { history.replaceState(null, '', '#' + b.dataset.goto); t.scrollIntoView({ block: 'start', behavior: 'smooth' }); } }
      });
    }
    if (tab === 'glossary') {
      const input = $('#glossary-search', el);
      input.addEventListener('input', () => {
        const q = input.value.trim().toLowerCase();
        el.querySelectorAll('.term').forEach((t) => { t.hidden = q !== '' && !t.dataset.term.includes(q) && !t.textContent.toLowerCase().includes(q); });
      });
    }
  }

  $('#brand-tag').textContent = FZ.status.project.tagline;
  $('#foot').innerHTML = `<span>Built ${esc(FZ.built)} from the repository's docs. <a href="${esc(FZ.status.project.repo)}">Source on GitHub</a> · <a href="${esc(FZ.status.project.repo)}/blob/main/docs/TRACKING.md">How this site works</a></span><span>${esc(FZ.status.project.name)} · ${esc(FZ.status.project.owner)}</span>`;
  window.addEventListener('hashchange', () => show(location.hash));
  show(location.hash);
})();
