/* Formation Zero film room (hand-made example). Reads window.FZ_DEMO (games + play lists) and
   window.FZ_DEMO_RECORDS (fully authored play records with tracking). */
(function () {
  'use strict';
  const GAMES = (window.FZ_DEMO && window.FZ_DEMO.games) || [];
  const RECORDS = window.FZ_DEMO_RECORDS || {};
  const $ = (s, r) => (r || document).querySelector(s);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const FIELD_L = 120, FIELD_W = 53.333, S = 16;            // yards, px per yard
  const HASH = 23.583;
  const ORD = { 1: '1st', 2: '2nd', 3: '3rd', 4: '4th' };
  const QTR = { 1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'OT' };
  const ROUTE_WORDS = { flat: 'flat', slant: 'slant', comeback: 'comeback', curl: 'curl', out: 'out', dig: 'dig', corner: 'corner', post: 'post', go: 'go', seam: 'seam', wheel: 'wheel', screen: 'screen', swing: 'swing', cross: 'shallow cross', stick: 'stick', hitch: 'hitch' };
  const COVER = { cover_0: 'Cover 0', cover_1: 'Cover 1', cover_2: 'Cover 2', cover_2_man: 'Cover 2 man', cover_3: 'Cover 3', cover_4: 'Cover 4', cover_6: 'Cover 6', unknown: 'unknown' };
  const CAMERAS = {
    top: { tilt: 0, spin: 0, zoom: 1.0, follow: false, label: "Bird's-eye" },
    side: { tilt: 56, spin: 0, zoom: 1.1, follow: true, label: 'Sideline' },
    end: { tilt: 58, spin: -90, zoom: 1.0, follow: true, label: 'End zone' },
    free: { tilt: 45, spin: -35, zoom: 1.05, follow: true, label: 'Free camera' },
    film: { tilt: 0, spin: 0, zoom: 1.0, follow: false, label: 'Film', video: true },
  };
  const OL = ['LT', 'LG', 'C', 'RG', 'RT'];
  const teamOf = (game, abbr) => (game.away.abbr === abbr ? game.away : game.home);
  const fmtDD = (p) => (p.down ? `${ORD[p.down]} & ${p.ydstogo === 0 ? 'Goal' : p.ydstogo}` : '—');
  const yd = (v) => (v == null ? '—' : `${Number(v).toFixed(1)} yd`);

  const state = { view: 'home', game: null, play: null, cam: 'top', lastCam: 'top', split: false, filmAngle: 'sideline',
    frame: 0, playing: false, speed: 1, labels: 'jersey', trails: true, routes: false, follow: 'ball', tab: 'play',
    filters: { qtr: 'all', type: 'all', tags: new Set(), q: '' }, selected: null, free: { tilt: 45, spin: -35, zoom: 1.05 }, focus: null };
  let raf = null, lastT = 0, rendered = { game: null };
  const dom = {};

  /* ------------------------------------------------------------ routing --------- */
  function parseHash() {
    const h = location.hash.replace(/^#\/?/, '');
    const [path, query] = h.split('?');
    const m = /^g\/([^/]+)(?:\/p\/([^/]+))?/.exec(path);
    if (!m) return { view: 'home' };
    const q = Object.fromEntries((query || '').split('&').filter(Boolean).map((kv) => kv.split('=')));
    return { view: 'game', game: m[1], play: m[2] || null, frame: q.f != null ? Number(q.f) : null, cam: CAMERAS[q.cam] ? q.cam : null, split: q.split === '1' };
  }
  function go(gameKey, playUid) { location.hash = playUid ? `#/g/${gameKey}/p/${playUid}` : gameKey ? `#/g/${gameKey}` : '#/'; }
  function momentLink() { const r = record(); return `${location.origin}${location.pathname}#/g/${state.game.game_key}/p/${state.play.uid}${r ? `?f=${Math.floor(state.frame)}&cam=${state.cam}${state.split ? '&split=1' : ''}` : ''}`; }

  function route() {
    const r = parseHash();
    stop();
    if (r.view === 'home') { state.view = 'home'; state.game = null; state.play = null; rendered.game = null; renderHome(); return; }
    const game = GAMES.find((g) => g.game_key === r.game);
    if (!game) { location.hash = '#/'; return; }
    state.view = 'game'; state.game = game;
    const play = game.plays.find((p) => p.uid === r.play) || game.plays.find((p) => p.featured) || game.plays[0];
    state.play = play; state.frame = 0; state.selected = null; state.focus = null; state.follow = 'ball';
    if (rendered.game !== game.game_key) { renderRoom(); rendered.game = game.game_key; }
    if (r.play !== play.uid) { history.replaceState(null, '', `#/g/${game.game_key}/p/${play.uid}`); }
    if (r.cam) setCamera(r.cam, true);
    if (r.split) { state.split = true; $('#t-split').setAttribute('aria-pressed', 'true'); }
    loadPlay();
    if (r.frame != null && record()) { state.frame = r.frame; state.focus = null; for (let i = 0; i < 40; i++) drawFrame(); }
  }

  /* ------------------------------------------------------------ home ------------ */
  function crumbs(parts) {
    $('#crumbs').innerHTML = parts.map((p, i) => (i ? '<span class="sep">/</span>' : '') + (p.href ? `<a href="${p.href}">${esc(p.text)}</a>` : `<span class="cur">${esc(p.text)}</span>`)).join('');
  }
  function renderHome() {
    crumbs([{ text: 'Games' }]);
    document.title = 'Formation Zero Film Room';
    $('#main').innerHTML = `
      <div class="home">
        <div class="eyebrow">Film room · hand-made example of the finished product</div>
        <h1>Open a game. Pick a play. Watch it from anywhere.</h1>
        <p class="lede">This is what Formation Zero will hand you when the pipeline is done: every play as a record you can replay from above or from any angle, check against the film, read as a formation, ask questions about, and export as JSON.</p>
        <p class="note">Three plays in the Rams–Bears game are fully brought to life. Their situation, personnel, formation label, passer, receiver, tackler and result are official. Every path between those facts was authored by hand to show the shape of the real output. Every other play shows exactly what the system knows before the vision work runs: the official data. The Film view is a placeholder: real footage is frame-locked to the recreation in the product, and none is published here.</p>
        <div class="games">${GAMES.map((g) => `
          <a class="game-card" href="#/g/${esc(g.game_key)}" style="--away-c:${esc(g.away.color)};--home-c:${esc(g.home.color)}">
            <div class="strip"></div>
            <div class="body">
              <div class="score">
                <div class="team"><div class="abbr" style="color:${esc(g.away.color)}">${esc(g.away.abbr)}</div><div class="name">${esc(g.away.name)}</div></div>
                <div><div class="pts">${g.score.away}<span class="muted"> – </span>${g.score.home}</div><div class="at">${g.overtime ? 'FINAL / OT' : 'FINAL'}</div></div>
                <div class="team home"><div class="abbr" style="color:${esc(g.home.color)}">${esc(g.home.abbr)}</div><div class="name">${esc(g.home.name)}</div></div>
              </div>
              <div class="meta"><span>${esc(g.label)}</span><span>·</span><span>${esc(g.date)}</span><span>·</span><span>${esc(g.stadium)}</span></div>
            </div>
            <div class="foot"><span>${g.play_count} plays · ${g.featured_count ? `<span class="chip first">${g.featured_count} fully broken down</span>` : '<span class="chip">official data only</span>'}</span><span class="cta">Open game →</span></div>
          </a>`).join('')}
        </div>
      </div>`;
  }

  /* ------------------------------------------------------------ room shell ------ */
  function renderRoom() {
    const g = state.game;
    $('#main').innerHTML = `
      <div class="room">
        <aside class="pane pane-left">
          <div class="pane-head">
            <div class="eyebrow">Play-by-play · ${g.play_count} plays</div>
            <input class="search" id="q" type="search" placeholder="Search: Nacua, 89, 3rd & long, red zone…" autocomplete="off" aria-label="Search plays">
            <div class="filters" id="filters">
              ${['all', 1, 2, 3, 4, 5].filter((q) => q === 'all' || g.plays.some((p) => p.qtr === q)).map((q) => `<button type="button" data-f="qtr" data-v="${q}" aria-pressed="${q === 'all'}">${q === 'all' ? 'All' : QTR[q]}</button>`).join('')}
              <button type="button" data-f="type" data-v="pass" aria-pressed="false">Pass</button>
              <button type="button" data-f="type" data-v="run" aria-pressed="false">Run</button>
              <button type="button" data-f="tag" data-v="featured" aria-pressed="false"><span class="dot-f"></span> Broken down</button>
              <button type="button" data-f="tag" data-v="td" aria-pressed="false">TD</button>
              <button type="button" data-f="tag" data-v="explosive" aria-pressed="false">15+ yds</button>
              <button type="button" data-f="tag" data-v="shotgun" aria-pressed="false">Shotgun</button>
              <button type="button" data-f="tag" data-v="motion" aria-pressed="false">Motion</button>
              <button type="button" data-f="tag" data-v="pa" aria-pressed="false">Play-action</button>
            </div>
          </div>
          <div class="pane-scroll" id="playlist"></div>
        </aside>
        <section class="viewer">
          <div class="view-bar">
            <div class="seg" id="cams">${Object.entries(CAMERAS).map(([k, c]) => `<button type="button" data-cam="${k}" aria-pressed="${k === state.cam}">${c.label}</button>`).join('')}</div>
            <div class="toggles">
              <button type="button" id="t-split" aria-pressed="false" title="Film beside the recreation">Split</button>
              <button type="button" id="t-follow" title="What the sideline, end zone and free cameras track">Follow: ball</button>
              <button type="button" id="t-labels">Labels: jersey</button>
              <button type="button" id="t-trails" aria-pressed="true">Trails</button>
              <button type="button" id="t-routes" aria-pressed="false">Routes</button>
              <button type="button" id="t-share" title="Copy a link to this exact moment">Share moment</button>
            </div>
          </div>
          <div class="stage-wrap" id="stage-wrap">
            <div class="film" id="film" hidden></div>
            <div class="stage-col" id="stage-col">
              <div class="stage" id="stage"><div class="cam" id="cam"><div class="field" id="field"></div></div></div>
              <div class="hud" id="hud"></div>
              <div class="free-cam" id="freecam" hidden>
                <label>tilt <input type="range" id="fc-tilt" min="0" max="72" value="45"></label>
                <label>spin <input type="range" id="fc-spin" min="-180" max="180" value="-35"></label>
                <label>zoom <input type="range" id="fc-zoom" min="8" max="30" value="10"></label>
              </div>
              <div class="player-card" id="pcard" hidden></div>
              <div class="stage-note" id="note" hidden></div>
            </div>
          </div>
          <div class="transport" id="transport">
            <div class="tl"><div class="bands" id="bands"></div><div id="evs"></div><input type="range" id="scrub" min="0" max="100" value="0" aria-label="Frame"></div>
            <div class="tr-row">
              <button class="btn primary" id="b-play" type="button">Play</button>
              <button class="btn" id="b-snap" type="button" title="Jump to the snap">Snap</button>
              <button class="btn" id="b-speed" type="button">1×</button>
              <span class="frame" id="frame-lbl"></span>
            </div>
          </div>
        </section>
        <aside class="pane pane-right">
          <div class="pane-head"><div class="ptabs" id="ptabs" role="tablist">
            ${['play', 'players', 'chat', 'json'].map((t) => `<button type="button" role="tab" data-tab="${t}" aria-selected="${t === state.tab}">${t === 'json' ? 'JSON' : t[0].toUpperCase() + t.slice(1)}</button>`).join('')}
          </div></div>
          <div class="pane-scroll" id="panel"></div>
        </aside>
      </div>`;
    buildField();
    wireRoom();
    renderPlaylist();
  }

  function wireRoom() {
    $('#filters').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      const f = b.dataset.f, v = b.dataset.v;
      if (f === 'qtr') { state.filters.qtr = v === 'all' ? 'all' : Number(v); $('#filters').querySelectorAll('[data-f="qtr"]').forEach((x) => x.setAttribute('aria-pressed', x === b)); }
      if (f === 'type') { state.filters.type = state.filters.type === v ? 'all' : v; $('#filters').querySelectorAll('[data-f="type"]').forEach((x) => x.setAttribute('aria-pressed', x.dataset.v === state.filters.type)); }
      if (f === 'tag') { if (state.filters.tags.has(v)) state.filters.tags.delete(v); else state.filters.tags.add(v); b.setAttribute('aria-pressed', state.filters.tags.has(v)); }
      renderPlaylist();
    });
    $('#q').addEventListener('input', (e) => { state.filters.q = e.target.value; renderPlaylist(); });
    $('#playlist').addEventListener('click', (e) => { const r = e.target.closest('.play-row'); if (r) go(state.game.game_key, r.dataset.uid); });
    $('#cams').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setCamera(b.dataset.cam); });
    $('#t-split').addEventListener('click', (e) => { state.split = !state.split; e.currentTarget.setAttribute('aria-pressed', state.split); layoutStage(); drawFrame(); });
    $('#t-follow').addEventListener('click', () => { state.follow = state.follow === 'ball' ? 'player' : 'ball'; state.focus = null; updateFollowLabel(); drawFrame(); });
    $('#t-labels').addEventListener('click', () => { state.labels = state.labels === 'jersey' ? 'name' : state.labels === 'name' ? 'role' : 'jersey'; $('#t-labels').textContent = 'Labels: ' + state.labels; drawFrame(); });
    $('#t-trails').addEventListener('click', (e) => { state.trails = !state.trails; e.currentTarget.setAttribute('aria-pressed', state.trails); drawFrame(); });
    $('#t-routes').addEventListener('click', (e) => { state.routes = !state.routes; e.currentTarget.setAttribute('aria-pressed', state.routes); drawFrame(); });
    $('#t-share').addEventListener('click', (e) => { const b = e.currentTarget; navigator.clipboard.writeText(momentLink()).then(() => { b.textContent = 'Link copied'; setTimeout(() => { b.textContent = 'Share moment'; }, 1400); }).catch(() => { prompt('Copy this link', momentLink()); }); });
    ['tilt', 'spin', 'zoom'].forEach((k) => $('#fc-' + k).addEventListener('input', (e) => { state.free[k] = k === 'zoom' ? Number(e.target.value) / 10 : Number(e.target.value); applyCamera(); }));
    $('#scrub').addEventListener('input', (e) => { stop(); state.frame = Number(e.target.value); drawFrame(); });
    $('#b-play').addEventListener('click', () => (state.playing ? stop() : play()));
    $('#b-snap').addEventListener('click', () => { const r = record(); if (r) { stop(); state.frame = r.film.snap_frame; drawFrame(); } });
    $('#b-speed').addEventListener('click', (e) => { state.speed = state.speed === 1 ? 0.5 : state.speed === 0.5 ? 0.25 : state.speed === 0.25 ? 2 : 1; e.currentTarget.textContent = state.speed + '×'; });
    $('#ptabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setTab(b.dataset.tab); });
    $('#field').addEventListener('click', (e) => { const m = e.target.closest('.marker'); if (m) selectPlayer(m.dataset.id === state.selected ? null : m.dataset.id); });
    $('#film').addEventListener('click', (e) => { const b = e.target.closest('[data-angle]'); if (b) { state.filmAngle = b.dataset.angle; drawFrame(); } });
    document.addEventListener('keydown', (e) => {
      if (state.view !== 'game' || e.target.matches('input, textarea')) return;
      if (e.key === ' ') { e.preventDefault(); state.playing ? stop() : play(); }
      if (e.key === 'ArrowRight') { stop(); step(e.shiftKey ? 10 : 1); }
      if (e.key === 'ArrowLeft') { stop(); step(e.shiftKey ? -10 : -1); }
      if (e.key === 'j' || e.key === 'J') { const rows = visiblePlays(); const i = rows.findIndex((p) => p.uid === state.play.uid); if (rows[i + 1]) go(state.game.game_key, rows[i + 1].uid); }
      if (e.key === 'k' || e.key === 'K') { const rows = visiblePlays(); const i = rows.findIndex((p) => p.uid === state.play.uid); if (rows[i - 1]) go(state.game.game_key, rows[i - 1].uid); }
      if (e.key === 'Escape') selectPlayer(null);
    });
  }
  function updateFollowLabel() {
    const r = record(); const pl = r && state.selected ? playerById(r, state.selected) : null;
    $('#t-follow').textContent = state.follow === 'player' && pl ? `Follow: #${pl.jersey}` : 'Follow: ball';
  }

  /* ------------------------------------------------------------ play list ------- */
  function matchesQuery(p, q) {
    if (!q) return true;
    const s = q.trim().toLowerCase();
    const dd = /(\d)(?:st|nd|rd|th)?\s*(?:&|and)\s*(long|short|goal|\d+)/.exec(s);
    if (dd) {
      if (p.down !== Number(dd[1])) return false;
      const d = dd[2];
      if (d === 'long') return p.ydstogo >= 7;
      if (d === 'short') return p.ydstogo <= 3;
      if (d === 'goal') return p.yardline_100 <= p.ydstogo;
      return p.ydstogo === Number(d);
    }
    if (/red ?zone/.test(s)) return p.yardline_100 != null && p.yardline_100 <= 20;
    if (/two.?minute|2.?minute/.test(s)) return p.clock && (p.qtr === 2 || p.qtr === 4) && Number(p.clock.split(':')[0]) < 2;
    const hay = `${p.desc} ${p.yrdln || ''} ${p.posteam || ''} ${p.formation || ''} ${QTR[p.qtr] || ''}`.toLowerCase();
    return s.split(/\s+/).every((w) => hay.includes(w));
  }
  function hasTag(p, t) {
    const r = RECORDS[p.uid];
    if (t === 'featured') return p.featured;
    if (t === 'td') return p.td;
    if (t === 'explosive') return p.yards != null && p.yards >= 15;
    if (t === 'shotgun') return p.formation === 'SHOTGUN' || /\(shotgun/i.test(p.desc);
    if (t === 'motion') return !!(r && (r.pre_snap.motions.length || r.pre_snap.shifts.length));
    if (t === 'pa') return !!(r && r.live.play_action);
    return true;
  }
  function visiblePlays() {
    const f = state.filters;
    return state.game.plays.filter((p) => (f.qtr === 'all' || p.qtr === f.qtr) && (f.type === 'all' || p.play_type === f.type) && [...f.tags].every((t) => hasTag(p, t)) && matchesQuery(p, f.q));
  }
  function renderPlaylist() {
    const g = state.game; const rows = visiblePlays();
    let html = '', lastDrive = null;
    rows.forEach((p) => {
      if (p.drive !== lastDrive) { lastDrive = p.drive; html += `<div class="drive-head">Drive ${p.drive} · ${esc(p.posteam)} ball · ${QTR[p.qtr] || ''}</div>`; }
      const t = teamOf(g, p.posteam); const r = RECORDS[p.uid];
      const tags = [];
      if (p.featured) tags.push('<span class="chip first">broken down</span>');
      if (p.td) tags.push('<span class="chip good">TD</span>');
      if (p.first_down && !p.td) tags.push('<span class="chip">1st down</span>');
      if (p.yards != null && p.yards >= 15) tags.push(`<span class="chip">+${p.yards}</span>`);
      if (r && r.pre_snap.motions.length) tags.push('<span class="chip">motion</span>');
      if (r && r.live.play_action) tags.push('<span class="chip">play-action</span>');
      html += `<button class="play-row" type="button" data-uid="${esc(p.uid)}" aria-current="${state.play && p.uid === state.play.uid}" style="--team-c:${esc(t.color)}">
        <span class="stripe"></span>
        <span><span class="l1"><span>${QTR[p.qtr] || ''} ${esc(p.clock || '')}</span><b>${esc(fmtDD(p))}</b><span>${esc(p.yrdln || '')}</span></span>
        <span class="l2">${esc(p.desc)}</span>${tags.length ? `<span class="tags">${tags.join('')}</span>` : ''}</span></button>`;
    });
    $('#playlist').innerHTML = html || '<div class="pending" style="margin:14px">No plays match. Try a name, a number, "3rd & long", "red zone", or clear the chips.</div>';
    const cur = $('#playlist [aria-current="true"]'); if (cur) cur.scrollIntoView({ block: 'nearest' });
  }

  /* ------------------------------------------------------------ field ----------- */
  function fieldSVG() {
    let s = `<svg class="lines" viewBox="0 0 ${FIELD_L} ${FIELD_W}" width="${FIELD_L * S}" height="${FIELD_W * S}" xmlns="http://www.w3.org/2000/svg">`;
    s += `<rect x="0" y="0" width="${FIELD_L}" height="${FIELD_W}" fill="#2C6B45"/>`;
    for (let x = 10; x < 110; x += 10) s += `<rect x="${x}" y="0" width="5" height="${FIELD_W}" fill="#2A6742"/>`;
    s += `<rect x="0" y="0" width="10" height="${FIELD_W}" fill="#1F4D33"/><rect x="110" y="0" width="10" height="${FIELD_W}" fill="#1F4D33"/>`;
    for (let x = 10; x <= 110; x += 5) s += `<line x1="${x}" y1="0" x2="${x}" y2="${FIELD_W}" stroke="#E9EEE8" stroke-width="${x === 10 || x === 110 ? 0.35 : 0.14}" opacity="0.9"/>`;
    for (let x = 11; x < 110; x++) { if (x % 5 === 0) continue; for (const y of [HASH, FIELD_W - HASH, 0.6, FIELD_W - 0.6]) s += `<line x1="${x}" y1="${y - 0.35}" x2="${x}" y2="${y + 0.35}" stroke="#E9EEE8" stroke-width="0.1" opacity="0.8"/>`; }
    s += `<rect x="0.2" y="0.2" width="${FIELD_L - 0.4}" height="${FIELD_W - 0.4}" fill="none" stroke="#E9EEE8" stroke-width="0.4"/>`;
    for (let x = 20; x <= 100; x += 10) { const n = x <= 60 ? x - 10 : 110 - x; const label = String(n).padStart(2, '0');
      s += `<text x="${x}" y="${FIELD_W - 6}" font-family="Barlow Condensed, Arial Narrow, sans-serif" font-weight="700" font-size="4.2" fill="#E9EEE8" text-anchor="middle" opacity="0.85">${label}</text>`;
      s += `<text x="${x}" y="6" font-family="Barlow Condensed, Arial Narrow, sans-serif" font-weight="700" font-size="4.2" fill="#E9EEE8" text-anchor="middle" opacity="0.85" transform="rotate(180 ${x} 4.6)">${label}</text>`; }
    s += `<line id="los-line" x1="0" y1="0" x2="0" y2="${FIELD_W}" stroke="#5B86F2" stroke-width="0.3" opacity="0.95"/>`;
    s += `<line id="fd-line" x1="0" y1="0" x2="0" y2="${FIELD_W}" stroke="#F2C94C" stroke-width="0.3" opacity="0.95"/>`;
    s += '</svg>';
    return s;
  }
  function buildField() {
    const f = $('#field');
    f.style.width = FIELD_L * S + 'px'; f.style.height = FIELD_W * S + 'px';
    f.innerHTML = fieldSVG() + `<svg class="trails" viewBox="0 0 ${FIELD_L * S} ${FIELD_W * S}" width="${FIELD_L * S}" height="${FIELD_W * S}" xmlns="http://www.w3.org/2000/svg" id="trails"></svg><div id="markers"></div>`;
    dom.field = f; dom.cam = $('#cam'); dom.markers = $('#markers'); dom.trails = $('#trails');
    applyCamera(); layoutStage();
  }
  const px = (x) => x * S, py = (y) => (FIELD_W - y) * S;   // field y up -> screen y down

  function camParams() {
    const key = state.cam === 'film' ? state.lastCam : state.cam;
    const c = CAMERAS[key];
    return key === 'free' ? { ...c, ...state.free } : c;
  }
  function applyCamera() {
    const c = camParams();
    const r = record();
    let fx, fy;
    if (r && c.follow && state.focus) { fx = state.focus[0]; fy = state.focus[1]; }
    else if (r) { fx = Math.min(112, r.situation.los_x + 12); fy = FIELD_W / 2; }
    else if (state.play && state.play.yardline_100 != null) { fx = Math.min(112, 10 + (100 - state.play.yardline_100) + 10); fy = FIELD_W / 2; }
    else { fx = 60; fy = FIELD_W / 2; }
    dom.cam.style.transform = `rotateX(${c.tilt}deg) rotateZ(${c.spin}deg) scale(${c.zoom}) translate(${-px(fx)}px, ${-py(fy)}px)`;
    dom.field.dataset.bill = `rotateZ(${-c.spin}deg) rotateX(${-c.tilt}deg)`;
    $('#stage').style.perspective = c.tilt > 0 ? '1500px' : 'none';
    dom.markers.querySelectorAll('.bill').forEach((b) => { if (!b.closest('.ballm')) b.style.transform = dom.field.dataset.bill; });
  }
  function layoutStage() {
    const wrap = $('#stage-wrap'); const filmOnly = state.cam === 'film' && !state.split;
    wrap.classList.toggle('split', state.split && !!record());
    $('#film').hidden = !(state.cam === 'film' || state.split) || !record();
    $('#stage-col').hidden = filmOnly && !!record();
  }
  function setCamera(k, silent) {
    if (k !== 'film') state.lastCam = k;
    state.cam = k; $('#cams').querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.cam === k));
    $('#freecam').hidden = k !== 'free';
    if (k === 'free') { $('#fc-tilt').value = state.free.tilt; $('#fc-spin').value = state.free.spin; $('#fc-zoom').value = Math.round(state.free.zoom * 10); }
    layoutStage(); applyCamera(); if (!silent) drawFrame();
  }

  /* ------------------------------------------------------------ play loading ---- */
  const record = () => (state.play ? RECORDS[state.play.uid] : null);
  const playerById = (r, id) => r.players.find((p) => p.track_id === id);
  const lastName = (n) => (n ? n.split(' ').slice(-1)[0] : null);

  function loadPlay() {
    const p = state.play, r = record(), g = state.game;
    crumbs([{ text: 'Games', href: '#/' }, { text: `${g.away.abbr} at ${g.home.abbr} · ${g.label}`, href: `#/g/${g.game_key}` }, { text: `${QTR[p.qtr] || ''} ${p.clock || ''} · ${fmtDD(p)} at ${p.yrdln || ''}` }]);
    document.title = `${fmtDD(p)} at ${p.yrdln || ''} · Formation Zero Film Room`;
    $('#playlist').querySelectorAll('.play-row').forEach((row) => row.setAttribute('aria-current', row.dataset.uid === p.uid));
    const cur = $('#playlist [aria-current="true"]'); if (cur) cur.scrollIntoView({ block: 'nearest' });
    const losX = r ? r.situation.los_x : 10 + (100 - p.yardline_100);
    const fdX = r ? r.situation.first_down_x : Math.min(110, losX + (p.ydstogo || 0));
    $('#los-line').setAttribute('x1', losX); $('#los-line').setAttribute('x2', losX);
    $('#fd-line').setAttribute('x1', fdX); $('#fd-line').setAttribute('x2', fdX);
    dom.markers.innerHTML = ''; dom.trails.innerHTML = '';
    state.frame = 0; state.selected = null; state.focus = null; $('#pcard').hidden = true; updateFollowLabel();
    if (r) {
      buildMarkers(r); buildTimeline(r); buildFilm(r);
      $('#note').hidden = true; $('#transport').style.opacity = '1';
      state.frame = Math.max(0, r.film.snap_frame - 15);
    } else {
      const ballM = document.createElement('div'); ballM.className = 'ball'; ballM.style.left = px(losX) + 'px'; ballM.style.top = py(FIELD_W / 2) + 'px'; dom.markers.appendChild(ballM);
      $('#bands').innerHTML = ''; $('#evs').innerHTML = ''; $('#scrub').max = 0; $('#scrub').value = 0;
      const next = g.plays.find((x) => x.featured);
      $('#note').innerHTML = `<h3>Official data only</h3><p>This play has its situation, result and labels from the play-by-play. Player positions arrive when the vision pipeline runs. This is exactly what a client sees for a play before analysis.</p>${next ? `<button type="button" id="jump">Open a fully broken-down play</button>` : ''}`;
      $('#note').hidden = false;
      if (next) $('#jump').addEventListener('click', () => go(g.game_key, next.uid));
      $('#transport').style.opacity = '0.5';
    }
    layoutStage(); applyCamera(); drawFrame(); renderPanel();
  }

  function buildMarkers(r) {
    const g = state.game;
    const frag = document.createDocumentFragment();
    r.players.forEach((pl) => {
      const t = pl.side === 'official' ? null : teamOf(g, pl.team);
      const m = document.createElement('div'); m.className = 'marker' + (pl.confidence != null && pl.confidence < 0.6 ? ' lowconf' : ''); m.dataset.id = pl.track_id;
      m.innerHTML = `<div class="shadow"></div><div class="bill"><div class="tag${pl.side === 'official' ? ' official' : ''}" style="${t ? `--c:${t.color};--i:${t.ink}` : ''}">${esc(pl.jersey)}</div></div>`;
      m.querySelector('.bill').style.transform = dom.field.dataset.bill;
      frag.appendChild(m);
    });
    const ball = document.createElement('div'); ball.className = 'marker ballm'; ball.innerHTML = '<div class="shadow" style="width:12px;height:6px"></div><div class="bill" style="transform-origin:50% 50%"><div class="ball"></div></div>';
    frag.appendChild(ball);
    dom.markers.appendChild(frag);
    dom.mk = {}; dom.markers.querySelectorAll('.marker').forEach((m) => { if (m.dataset.id) dom.mk[m.dataset.id] = m; });
    dom.ballM = ball;
  }
  function buildTimeline(r) {
    const n = r.tracking.frames; const ph = r.film.phases;
    $('#bands').innerHTML = ['pre_snap', 'live', 'post_snap'].map((k) => `<div class="band ${k}" style="width:${((ph[k][1] - ph[k][0] + 1) / n) * 100}%"></div>`).join('');
    const short = { snap: 'snap', play_fake: 'fake', handoff: 'handoff', pass_forward: 'throw', pass_outcome_caught: 'catch', first_contact: 'contact', tackle: 'tackle', touchdown: 'TD', whistle: 'whistle' };
    $('#evs').innerHTML = r.live.events.map((e, i) => `<div class="ev ${i % 2 ? 'lo' : 'hi'}" style="left:${(e.frame / (n - 1)) * 100}%" data-l="${esc(short[e.type] || e.type)}" title="${esc(e.type)} · frame ${e.frame}"></div>`).join('');
    $('#scrub').max = n - 1;
  }
  function buildFilm(r) {
    $('#film').innerHTML = `
      <div class="film-head"><span class="eyebrow">Film · frame-locked to the recreation</span>
        <div class="seg small">${['sideline', 'endzone'].map((a) => `<button type="button" data-angle="${a}" aria-pressed="${state.filmAngle === a}">${a === 'sideline' ? 'Sideline' : 'End zone'}</button>`).join('')}</div></div>
      <div class="monitor"><div class="monitor-glass"><div class="tc" id="film-tc">00:00:00:00</div><div class="film-note">Real footage plays here in the product, scrubbing in lockstep with the recreation: the recreation's frame numbers <em>are</em> the film's. None is published in this public demo.</div><div class="film-prog"><i id="film-prog"></i></div></div></div>
      <div class="film-meta kv"><b>clip</b><span class="mono" id="film-clip"></span><b>frame</b><span class="mono" id="film-frame"></span><b>phase</b><span id="film-phase"></span></div>`;
  }

  /* ------------------------------------------------------------ drawing --------- */
  function phaseAt(r, f) { const ph = r.film.phases; return f < ph.live[0] ? 'pre_snap' : f <= ph.live[1] ? 'live' : 'post_snap'; }
  function carrierAt(r, f) {
    const b = r.tracking.ball[f]; let best = null, bd = 1.2;
    for (const id of r.tracking.ids) { const p = r.tracking.xy[id][f]; const d = Math.hypot(p[0] - b[0], p[1] - b[1]); if (d < bd && b[2] < 2.2) { bd = d; best = id; } }
    return best;
  }
  function drawFrame() {
    const r = record(); const p = state.play;
    $('#scrub').value = state.frame;
    if (!r) { $('#hud').innerHTML = `<div class="dd">${esc(fmtDD(p))} · ${esc(p.yrdln || '')}</div><div class="sub">${esc(QTR[p.qtr] || '')} ${esc(p.clock || '')} · ${esc(p.posteam)} ball</div>`; $('#frame-lbl').textContent = 'no tracking for this play yet'; return; }
    const f = Math.max(0, Math.min(r.tracking.frames - 1, Math.floor(state.frame)));
    const phase = phaseAt(r, f); const carrier = carrierAt(r, f);
    const b = r.tracking.ball[f];
    if (camParams().follow) {
      const tgt = state.follow === 'player' && state.selected && r.tracking.xy[state.selected] ? r.tracking.xy[state.selected][f] : b;
      state.focus = state.focus ? [state.focus[0] + (tgt[0] - state.focus[0]) * 0.12, state.focus[1] + (tgt[1] - state.focus[1]) * 0.12] : [tgt[0], tgt[1]];
      applyCamera();
    }
    r.players.forEach((pl) => {
      const m = dom.mk[pl.track_id]; const t = r.tracking.xy[pl.track_id][f];
      m.style.transform = `translate(${px(t[0])}px, ${py(t[1])}px)`;
      const tag = m.querySelector('.tag');
      let label = pl.jersey;
      if (state.labels === 'name') label = lastName(pl.name) || pl.jersey;
      if (state.labels === 'role') label = pl.role;
      if (tag.textContent !== label) { tag.textContent = label; tag.classList.toggle('wide', label.length > 2); }
      m.classList.toggle('carrier', carrier === pl.track_id);
      m.classList.toggle('sel', state.selected === pl.track_id);
      m.classList.toggle('dim', !!state.selected && state.selected !== pl.track_id && !isRelated(r, pl.track_id));
    });
    dom.ballM.style.transform = `translate(${px(b[0])}px, ${py(b[1])}px)`;
    dom.ballM.querySelector('.bill').style.transform = `translateZ(${b[2] * S}px) ${dom.field.dataset.bill}`;
    dom.ballM.querySelector('.shadow').style.opacity = String(Math.max(0.15, 0.4 - b[2] * 0.03));
    drawTrails(r, f);
    const sit = r.situation;
    $('#hud').innerHTML = `<div class="dd">${esc(fmtDD(p))} · ${esc(sit.yrdln)}</div><div class="sub">${esc(QTR[sit.quarter])} ${esc(sit.clock)} · ${esc(sit.possession)} ball · ${esc(state.game.away.abbr)} ${sit.score.away}–${sit.score.home} ${esc(state.game.home.abbr)}</div><span class="phase ${phase}">${phase.replace('_', ' ')}</span>`;
    $('#frame-lbl').textContent = `frame ${f} / ${r.tracking.frames - 1} · ${(f / r.tracking.fps).toFixed(2)} s · ${r.tracking.fps} fps`;
    $('#b-play').textContent = state.playing ? 'Pause' : 'Play';
    if (!$('#film').hidden) {
      const fps = r.tracking.fps, sec = f / fps;
      $('#film-tc').textContent = `00:${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(Math.floor(sec % 60)).padStart(2, '0')}:${String(f % fps).padStart(2, '0')}`;
      $('#film-prog').style.width = `${(f / (r.tracking.frames - 1)) * 100}%`;
      $('#film-clip').textContent = r.film[state.filmAngle].clip; $('#film-frame').textContent = `${f} of ${r.film[state.filmAngle].frames}`; $('#film-phase').textContent = phase.replace('_', ' ');
      $('#film').querySelectorAll('[data-angle]').forEach((x) => x.setAttribute('aria-pressed', x.dataset.angle === state.filmAngle));
    }
    if (state.selected) renderPlayerCard(r, state.selected, f);
  }
  function isRelated(r, id) {
    const sel = playerById(r, state.selected); if (!sel) return false;
    const rel = [(sel.block || {}).engaged_with, (sel.coverage || {}).on, (sel.metrics || {}).nearest_at_catch, (sel.metrics || {}).nearest_at_throw, (sel.metrics || {}).assignment];
    return rel.includes(id);
  }
  function drawTrails(r, f) {
    const g = state.game; let s = '';
    const snap = r.film.snap_frame;
    if (state.routes && f >= snap) {
      r.players.filter((pl) => pl.route_detail).forEach((pl) => {
        const pts = r.tracking.xy[pl.track_id].slice(snap, Math.min(f, r.film.whistle_frame) + 1).map((t) => `${px(t[0])},${py(t[1])}`).join(' ');
        s += `<polyline points="${pts}" stroke="${teamOf(g, pl.team).color}" stroke-width="3" opacity="0.55"/>`;
      });
    }
    if (state.trails) {
      const from = Math.max(0, f - 30);
      r.players.forEach((pl) => {
        if (pl.side === 'official' || OL.includes(pl.role)) return;
        const pts = r.tracking.xy[pl.track_id].slice(from, f + 1).map((t) => `${px(t[0])},${py(t[1])}`).join(' ');
        s += `<polyline points="${pts}" stroke="${teamOf(g, pl.team).color}" stroke-width="2" opacity="${state.selected && state.selected !== pl.track_id ? 0.15 : 0.35}"/>`;
      });
      const bp = r.tracking.ball.slice(from, f + 1).map((t) => `${px(t[0])},${py(t[1])}`).join(' ');
      s += `<polyline points="${bp}" stroke="#E9EEE8" stroke-width="1.5" opacity="0.6" stroke-dasharray="4 3"/>`;
    }
    // Separation line: the selected player to the defender or receiver they are measured against.
    if (state.selected) {
      const other = counterpartAt(r, state.selected, f);
      if (other) { const a = r.tracking.xy[state.selected][f], bq = r.tracking.xy[other][f]; s += `<line x1="${px(a[0])}" y1="${py(a[1])}" x2="${px(bq[0])}" y2="${py(bq[1])}" stroke="#F2C94C" stroke-width="2" stroke-dasharray="6 4" opacity="0.9"/>`; }
    }
    dom.trails.innerHTML = s;
  }

  /* ---- distances, read live from the tracking table ---- */
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function nearestAt(r, id, pool, f) { let best = null, bd = 1e9; pool.forEach((o) => { const d = dist(r.tracking.xy[id][f], r.tracking.xy[o][f]); if (d < bd) { bd = d; best = o; } }); return [best, bd]; }
  function counterpartAt(r, id, f) {
    const pl = playerById(r, id); if (!pl || pl.side === 'official') return null;
    if (pl.side === 'offense') { if (!pl.route_detail && !pl.ball_carrier && !pl.qb) return null; const pool = r.players.filter((x) => x.side === 'defense' && (pl.qb ? x.pass_rush && x.pass_rush.rushed : true)).map((x) => x.track_id); return nearestAt(r, id, pool, f)[0]; }
    if (pl.coverage && pl.coverage.on) return pl.coverage.on;
    return carrierAt(r, f);
  }
  function seriesFor(r, id) {
    const pl = playerById(r, id); const ph = r.film.phases.live; const out = [];
    for (let f = ph[0]; f <= ph[1]; f++) { const o = counterpartAt(r, id, f); out.push(o ? dist(r.tracking.xy[id][f], r.tracking.xy[o][f]) : null); }
    return { from: ph[0], values: out, label: pl.side === 'offense' ? (pl.qb ? 'nearest rusher' : 'separation') : pl.coverage && pl.coverage.on ? 'to assignment' : 'to ball carrier' };
  }
  function sparkline(series, f) {
    const W = 220, H = 40, vals = series.values.map((v) => (v == null ? 0 : Math.min(v, 15)));
    const n = vals.length; if (!n) return '';
    const x = (i) => (i / (n - 1)) * W, y = (v) => H - 4 - (v / 15) * (H - 8);
    const pts = vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    const i = Math.max(0, Math.min(n - 1, f - series.from));
    return `<svg class="spark" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><line x1="0" y1="${y(5)}" x2="${W}" y2="${y(5)}" stroke="#27332B" stroke-width="1"/><text x="${W}" y="${y(5) - 2}" font-size="8" fill="#7F8C84" text-anchor="end" font-family="IBM Plex Mono, monospace">5 yd</text><polyline points="${pts}" fill="none" stroke="#F2C94C" stroke-width="1.5"/><circle cx="${x(i).toFixed(1)}" cy="${y(vals[i]).toFixed(1)}" r="3" fill="#F2C94C"/></svg>`;
  }
  function selectPlayer(id) {
    state.selected = id; state.focus = null; updateFollowLabel(); drawFrame();
    document.querySelectorAll('table.roster tr').forEach((tr) => tr.setAttribute('aria-selected', tr.dataset.id === id));
    if (!id) $('#pcard').hidden = true;
  }
  function renderPlayerCard(r, id, f) {
    const pl = playerById(r, id); if (!pl) return;
    const t = r.tracking.xy[id][f]; const sp = f > 0 ? dist(t, r.tracking.xy[id][f - 1]) * r.tracking.fps : 0;
    const bits = [];
    if (pl.route) bits.push(['route', ROUTE_WORDS[pl.route] || pl.route]);
    if (pl.technique) bits.push(['technique', pl.technique + (pl.stance ? ` · ${pl.stance}` : '')]);
    if (pl.motion) bits.push(['motion', `${pl.motion.type} ${pl.motion.direction}, ${pl.motion.at_snap.replace('_', ' ')} at snap`]);
    if (pl.coverage) bits.push(['coverage', pl.coverage.assignment.replace(/_/g, ' ') + (pl.coverage.on ? ` (on #${(playerById(r, pl.coverage.on) || {}).jersey})` : '')]);
    if (pl.block) bits.push(['block', pl.block.assignment.replace(/_/g, ' ') + (pl.block.engaged_with ? ` vs #${(playerById(r, pl.block.engaged_with) || {}).jersey}` : '')]);
    if (pl.pass_rush) bits.push(['rush', pl.pass_rush.path.replace(/_/g, ' ') + (pl.pass_rush.pressure ? ', pressure' : '')]);
    const m = pl.metrics || {}; const other = counterpartAt(r, id, f); const now = other ? dist(t, r.tracking.xy[other][f]) : null;
    let metric = '';
    if (m.kind === 'separation') metric = `<b>separation</b><span>now ${yd(now)}${m.at_throw_yds != null ? ` · at throw ${m.at_throw_yds}` : ''}${m.at_catch_yds != null ? ` · at catch ${m.at_catch_yds}` : ''} · min ${m.min_live_yds}</span>`;
    else if (m.kind === 'coverage') metric = `<b>${m.assignment ? 'to his man' : 'to nearest WR'}</b><span>now ${yd(now)} · avg ${m.mean_to_assignment_yds} · min ${m.min_to_assignment_yds}${m.to_assignment_at_throw_yds != null ? ` · at throw ${m.to_assignment_at_throw_yds}` : ''}</span><b>pursuit</b><span>closest to ball ${m.pursuit.min_to_carrier_yds} yd${m.pursuit.first_within_2yds_frame != null ? ` · within 2 yd at f${m.pursuit.first_within_2yds_frame}` : ''} · closing ${m.pursuit.closing_speed_max_yds_s} yd/s</span>`;
    else if (m.kind === 'front' && m.pursuit) metric = `<b>pursuit</b><span>closest to ball ${m.pursuit.min_to_carrier_yds} yd · closing ${m.pursuit.closing_speed_max_yds_s} yd/s</span>`;
    else if (pl.qb && r.live.qb.pressure) { const q = r.live.qb.pressure; metric = `<b>pressure</b><span>nearest rusher now ${yd(now)} · min ${q.nearest_rusher_min_yds}${q.time_to_pressure_s != null ? ` · pressure in ${q.time_to_pressure_s} s` : ' · no pressure'}</span>`; }
    const series = pl.side !== 'official' ? seriesFor(r, id) : null;
    $('#pcard').innerHTML = `<div class="nm">#${esc(pl.jersey)} ${esc(pl.name || (pl.side === 'official' ? 'Official' : 'identity pending'))}${pl.confidence != null && pl.confidence < 0.6 ? ` <span class="conf lo">${Math.round(pl.confidence * 100)}%</span>` : ''}</div>
      <div class="kv"><b>role</b><span>${esc(pl.role)}${pl.position_roster && pl.position_roster !== pl.role ? ` (${esc(pl.position_roster)})` : ''}</span>
      <b>at snap</b><span>${pl.alignment_at_snap.lateral > 0 ? 'L' : 'R'} ${Math.abs(pl.alignment_at_snap.lateral)} yd · ${pl.alignment_at_snap.depth >= 0 ? 'depth' : 'off'} ${Math.abs(pl.alignment_at_snap.depth)} yd</span>
      <b>now</b><span>${sp.toFixed(1)} yd/s · dir ${t[2]}° · x ${t[0].toFixed(1)} y ${t[1].toFixed(1)}</span>
      ${bits.map(([k, v]) => `<b>${esc(k)}</b><span>${esc(v)}</span>`).join('')}${metric}
      <b>max speed</b><span>${pl.summary.max_speed_yds_s} yd/s · ${pl.summary.distance_yds} yd covered</span></div>
      ${series ? `<div class="spark-wrap"><span class="eyebrow">${esc(series.label)} through the play</span>${sparkline(series, f)}</div>` : ''}
      <div class="card-actions"><button type="button" id="pc-follow">${state.follow === 'player' ? 'Stop following' : 'Chase camera'}</button><button type="button" id="pc-close">Close</button></div>`;
    $('#pcard').hidden = false;
    $('#pc-follow').onclick = () => { state.follow = state.follow === 'player' ? 'ball' : 'player'; state.focus = null; if (state.follow === 'player' && !camParams().follow) setCamera('free'); updateFollowLabel(); drawFrame(); };
    $('#pc-close').onclick = () => selectPlayer(null);
  }

  /* ------------------------------------------------------------ transport ------- */
  function play() { const r = record(); if (!r) return; if (state.frame >= r.tracking.frames - 1) state.frame = 0; state.playing = true; lastT = performance.now(); raf = requestAnimationFrame(tick); $('#b-play').textContent = 'Pause'; }
  function stop() { state.playing = false; if (raf) cancelAnimationFrame(raf); raf = null; const b = $('#b-play'); if (b) b.textContent = 'Play'; }
  function step(n) { const r = record(); if (!r) return; state.frame = Math.max(0, Math.min(r.tracking.frames - 1, Math.floor(state.frame) + n)); drawFrame(); }
  function tick(now) {
    const r = record(); if (!r || !state.playing) return;
    const dt = (now - lastT) / 1000; lastT = now;
    state.frame += dt * r.tracking.fps * state.speed;
    if (state.frame >= r.tracking.frames - 1) { state.frame = r.tracking.frames - 1; drawFrame(); stop(); return; }
    drawFrame();
    raf = requestAnimationFrame(tick);
  }

  /* ------------------------------------------------------------ right panel ----- */
  function setTab(t) { state.tab = t; $('#ptabs').querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === t)); renderPanel(); }
  function renderPanel() {
    const el = $('#panel'); const r = record(); const p = state.play;
    if (state.tab === 'play') el.innerHTML = `<div class="panel-body">${panelPlay(r, p)}</div>`;
    else if (state.tab === 'players') { el.innerHTML = `<div class="panel-body">${panelPlayers(r, p)}</div>`; el.querySelectorAll('table.roster tr[data-id]').forEach((tr) => tr.addEventListener('click', () => selectPlayer(tr.dataset.id === state.selected ? null : tr.dataset.id))); }
    else if (state.tab === 'chat') { el.innerHTML = panelChat(); wireChat(); }
    else if (state.tab === 'json') { el.innerHTML = panelJson(r, p); wireJson(r, p); }
  }
  const jersey = (r, id) => { const pl = playerById(r, id); return pl ? `#${pl.jersey}${pl.name ? ' ' + lastName(pl.name) : ''}` : id; };
  const conf = (c) => (c == null ? '' : `<span class="conf ${c >= 0.85 ? 'hi' : c >= 0.6 ? 'mid' : 'lo'}" title="confidence">${Math.round(c * 100)}%</span>`);
  const src = (o) => (o && o.source ? `<span class="src">source: ${esc(o.source)} ${conf(o.confidence)}</span>` : '');

  function formationSVG(r) {
    const g = state.game; const W = 38, H = 21, cx = W / 2, losY = 7.5;   // yards: lateral -19..19, depth -13.5..7.5
    const X = (lat) => cx - lat, Y = (dep) => losY + dep;                // offense left (+lat) draws on the left
    let s = `<svg class="formation" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Alignment at the snap">`;
    s += `<rect x="0" y="0" width="${W}" height="${H}" fill="#182119"/>`;
    for (let d = -15; d <= 5; d += 5) if (d !== 0) s += `<line x1="0" y1="${Y(d)}" x2="${W}" y2="${Y(d)}" stroke="#27332B" stroke-width="0.12"/>`;
    s += `<line x1="0" y1="${Y(0)}" x2="${W}" y2="${Y(0)}" stroke="#5B86F2" stroke-width="0.25"/>`;
    const fd = r.situation.first_down_x - r.situation.los_x; if (fd > 0 && fd < 13) s += `<line x1="0" y1="${Y(-fd)}" x2="${W}" y2="${Y(-fd)}" stroke="#F2C94C" stroke-width="0.25"/>`;
    s += `<ellipse cx="${cx}" cy="${Y(0.4)}" rx="0.5" ry="0.3" fill="#8B4A1F" stroke="#E9EEE8" stroke-width="0.08"/>`;
    r.players.filter((pl) => pl.side !== 'official').forEach((pl) => {
      const a = pl.alignment_at_snap; const t = teamOf(g, pl.team); const x = X(a.lateral), y = Y(a.depth);
      if (x < 1 || x > W - 1 || y < 1 || y > H - 1) return;
      const label = pl.side === 'defense' ? (pl.technique || pl.role) : (OL.includes(pl.role) ? '' : pl.role);
      if (pl.side === 'offense') s += `<circle cx="${x}" cy="${y}" r="1.1" fill="${t.color}" stroke="#0B100D" stroke-width="0.14"/>`;
      else s += `<circle cx="${x}" cy="${y}" r="1.1" fill="#182119" stroke="${t.color}" stroke-width="0.3"/>`;
      s += `<text x="${x}" y="${y + 0.42}" font-size="1.25" font-family="Barlow Condensed, Arial Narrow, sans-serif" font-weight="700" fill="${pl.side === 'offense' ? t.ink : t.color}" text-anchor="middle">${esc(pl.jersey)}</text>`;
      if (label) s += `<text x="${x}" y="${y + (pl.side === 'offense' ? 2.3 : -1.7)}" font-size="0.95" font-family="IBM Plex Mono, monospace" fill="#B9C4BB" text-anchor="middle">${esc(label)}</text>`;
      if (pl.motion) s += `<path d="M ${x + (pl.motion.direction === 'right' ? 1.4 : -1.4)} ${y} l ${pl.motion.direction === 'right' ? 1.6 : -1.6} 0" stroke="#F2C94C" stroke-width="0.25" marker-end="url(#arr)"/>`;
    });
    s += `<defs><marker id="arr" viewBox="0 0 4 4" refX="3" refY="2" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L4,2 L0,4 z" fill="#F2C94C"/></marker></defs>`;
    s += `<text x="1" y="${H - 1}" font-size="0.9" font-family="IBM Plex Mono, monospace" fill="#7F8C84">offense L</text><text x="${W - 1}" y="${H - 1}" font-size="0.9" font-family="IBM Plex Mono, monospace" fill="#7F8C84" text-anchor="end">offense R</text>`;
    s += '</svg>';
    return s;
  }

  function panelPlay(r, p) {
    const g = state.game; const off = teamOf(g, p.posteam);
    const situ = `<div class="situ"><div><div class="v">${esc(fmtDD(p))}</div><div class="k">down & distance</div></div><div><div class="v">${esc(p.yrdln || '')}</div><div class="k">ball on</div></div><div><div class="v">${esc(QTR[p.qtr] || '')} ${esc(p.clock || '')}</div><div class="k">clock</div></div></div>`;
    const official = `<div class="block"><h3>Official result</h3><p style="margin:0">${esc(p.desc)}</p><div class="kv"><b>Type</b><span>${esc(p.play_type)}</span><b>Yards</b><span>${p.yards == null ? '—' : p.yards}</span><b>Formation</b><span>${esc(p.formation || '—')}</span><b>Personnel</b><span>${p.personnel ? esc(p.personnel) + ' personnel' : '—'}${p.defense ? ` · defense ${esc(p.defense)}` : ''}</span>${p.epa != null ? `<b>EPA</b><span>${p.epa > 0 ? '+' : ''}${p.epa}</span>` : ''}</div><span class="src">source: official play-by-play ${conf(1)}</span></div>`;
    if (!r) return `${situ}${official}<div class="pending"><strong>Vision analysis pending.</strong> Formation as lined up, motion, routes, coverage, separation, every player's position and the events timeline arrive when the pipeline runs on this play. The record already exists with the sections above filled.</div>`;
    const pre = r.pre_snap; const live = r.live; const post = r.post_snap;
    const motions = pre.motions.map((m) => `<li><span class="j" style="--c:${off.color}">${esc((playerById(r, m.track_id) || {}).jersey)}</span><span class="r"><em>${esc(m.type)} motion</em> to the ${esc(m.direction)}, ${esc(m.at_snap.replace('_', ' '))} at the snap${m.speed_at_snap_yds_s ? ` (${m.speed_at_snap_yds_s} yd/s)` : ''}</span></li>`);
    const shifts = pre.shifts.map((s) => `<li><span class="j" style="--c:${off.color}">${esc((playerById(r, s.track_id) || {}).jersey)}</span><span class="r"><em>shift</em> ${esc(s.from_slot)} <span class="arrow">→</span> ${esc(s.to_slot)}, set before the snap</span></li>`);
    const routes = live.routes.map((rt) => { const m = (playerById(r, rt.track_id) || {}).metrics || {}; const sep = rt.targeted && m.at_throw_yds != null ? ` · <em>${m.at_throw_yds} yd open at throw${m.at_catch_yds != null ? `, ${m.at_catch_yds} at catch` : ''}</em>` : m.min_live_yds != null ? ` · min sep ${m.min_live_yds} yd` : '';
      return `<li><span class="j" style="--c:${off.color}">${esc((playerById(r, rt.track_id) || {}).jersey)}</span><span class="r"><em>${esc(ROUTE_WORDS[rt.route] || rt.route)}</em>${rt.tree != null ? ` (tree ${rt.tree})` : ''}${rt.depth_at_break_yds != null ? ` · breaks at ${rt.depth_at_break_yds} yd ${esc(rt.break_direction || '')}` : ''}${rt.targeted ? ' · <strong>targeted</strong>' : ''}${sep}</span></li>`; });
    const run = live.run ? `<div class="kv"><b>Carrier</b><span>${esc(jersey(r, live.run.ball_carrier))}</span><b>Gap</b><span>${esc(live.run.gap)}</span><b>Scheme</b><span>${esc(live.run.scheme)}</span></div>` : '';
    const pressure = live.qb.pressure ? `<b>Pressure</b><span>nearest rusher ${live.qb.pressure.nearest_rusher_min_yds} yd${live.qb.pressure.time_to_pressure_s != null ? ` · first pressure ${live.qb.pressure.time_to_pressure_s} s after the snap` : ' · clean'}${live.qb.pressure.rushers_within_3yds_at_throw != null ? ` · ${live.qb.pressure.rushers_within_3yds_at_throw} within 3 yd at the throw` : ''}</span>` : '';
    const events = live.events.map((e) => `<li><span class="j mono" style="font-size:0.8rem;color:var(--muted)">${e.frame}</span><span class="r">${esc(e.type.replace(/_/g, ' '))}${e.track_id ? ` · ${esc(jersey(r, e.track_id))}` : ''}${e.by ? ` by ${e.by.map((b) => esc(jersey(r, b))).join(', ')}` : ''}</span></li>`);
    const tightest = r.players.filter((x) => x.metrics && x.metrics.kind === 'coverage' && x.metrics.assignment).sort((a, b) => a.metrics.mean_to_assignment_yds - b.metrics.mean_to_assignment_yds)[0];
    return `${situ}
      <div class="block"><h3>Alignment at the snap</h3>${formationSVG(r)}<span class="src">Filled circles: offense. Rings: defense, labelled by technique where the record has one. Blue: line of scrimmage. Yellow: first down.</span></div>
      <div class="block"><h3>Before the snap</h3>
        <div class="kv"><b>Lined up</b><span>${esc(pre.formation_initial.name.replace(/_/g, ' '))} · strength ${esc(pre.formation_initial.strength)} · ${esc(pre.formation_initial.qb.replace('_', ' '))}</span>
        ${pre.formation_at_snap.name !== pre.formation_initial.name ? `<b>At snap</b><span>${esc(pre.formation_at_snap.name.replace(/_/g, ' '))} · strength ${esc(pre.formation_at_snap.strength)}</span>` : ''}
        <b>Personnel</b><span>${esc(r.personnel.offense.grouping)} personnel (${r.personnel.offense.rb} RB, ${r.personnel.offense.te} TE, ${r.personnel.offense.wr} WR)</span>
        <b>Defense</b><span>${esc(r.personnel.defense.package)} · ${esc(pre.defense.front)} front · ${esc(pre.defense.shell)} · ${pre.defense.box_count} in the box</span></div>
        ${shifts.length || motions.length ? `<ul class="list">${shifts.join('')}${motions.join('')}</ul>` : '<span class="muted" style="font-size:0.85rem">No shift or motion.</span>'}
        ${src(pre.formation_at_snap)}</div>
      <div class="block"><h3>Live</h3>
        <div class="kv"><b>Family</b><span>${esc(live.play_family.replace('_', ' '))}${live.play_action ? ' · play-action' : ''}</span>
        ${live.qb.drop !== 'none' ? `<b>QB</b><span>${esc(live.qb.drop.replace('_', ' '))} · ${live.qb.time_to_throw_s} s to throw · pocket ${esc(live.qb.pocket)}</span>` : ''}${pressure}
        <b>Coverage</b><span>${esc(COVER[live.defense.coverage.family] || live.defense.coverage.family)} · ${esc(live.defense.coverage.man_zone)}${live.defense.coverage.rotation ? ` · rotation ${esc(live.defense.coverage.rotation)}` : ''}${live.defense.blitz ? ' · blitz' : ''} ${conf(live.defense.coverage.confidence)}</span>
        ${tightest ? `<b>Tightest</b><span>${esc(jersey(r, tightest.track_id))} stayed ${tightest.metrics.mean_to_assignment_yds} yd from ${esc(jersey(r, tightest.metrics.assignment))} on average</span>` : ''}
        ${live.ball.air_yards != null ? `<b>Ball</b><span>${live.ball.air_yards} air yards · ${esc(live.ball.outcome)}</span>` : ''}</div>
        ${run}${routes.length ? `<ul class="list">${routes.join('')}</ul>` : ''}${src(live.defense.coverage)}</div>
      <div class="block"><h3>Result</h3>
        <div class="kv"><b>Outcome</b><span>${esc(post.result)} · ${post.yards_gained} yards${post.first_down ? ' · first down' : ''}</span>
        ${post.yards_after_catch != null ? `<b>After catch</b><span>${post.yards_after_catch} yards</span>` : ''}
        ${post.tackle ? `<b>Tackle</b><span>by ${post.tackle.by.map((b) => esc(jersey(r, b))).join(', ')} at x ${post.tackle.x}, y ${post.tackle.y}</span>` : ''}
        <b>Ball spot</b><span>x ${post.ball_spot.x} · y ${post.ball_spot.y}</span></div></div>
      <div class="block"><h3>Events</h3><ul class="list">${events.join('')}</ul></div>
      <div class="pending" style="font-size:0.8rem">${esc(r.demo_note)}</div>`;
  }

  function panelPlayers(r, p) {
    const g = state.game;
    if (!r) return `<div class="pending"><strong>Who is on the field</strong> arrives with the vision pipeline: every player's jersey, role and alignment at the snap. The official play-by-play names only the players who touched the ball.</div><div class="block"><h3>Named in the play-by-play</h3><p style="margin:0">${esc(p.desc)}</p></div>`;
    const side = (label, list, team) => `<div class="block"><h3>${esc(label)} · ${esc(team.name)}</h3><table class="roster"><thead><tr><th>#</th><th>Player</th><th>Role</th><th>At snap</th><th>${list[0].side === 'offense' ? 'Route · sep' : 'Assignment · closeness'}</th></tr></thead><tbody>
      ${list.map((pl) => { const m = pl.metrics || {};
        const doing = pl.side === 'offense'
          ? (pl.route ? `${ROUTE_WORDS[pl.route] || pl.route}${m.min_live_yds != null ? ` · min ${m.min_live_yds}` : ''}` : pl.ball_carrier ? 'ball carrier' : pl.qb ? 'passer' : pl.block ? pl.block.assignment.replace(/_/g, ' ') : pl.motion ? `${pl.motion.type} motion` : '')
          : (pl.coverage ? `${pl.coverage.assignment.replace(/_/g, ' ')}${m.mean_to_assignment_yds != null ? ` · avg ${m.mean_to_assignment_yds}` : ''}` : pl.technique ? `${pl.technique}-tech${m.pursuit ? ` · ${m.pursuit.min_to_carrier_yds} to ball` : ''}` : '');
        const low = pl.confidence != null && pl.confidence < 0.6;
        return `<tr data-id="${esc(pl.track_id)}" aria-selected="${state.selected === pl.track_id}"><td class="j" style="--c:${team.color}">${esc(pl.jersey)}</td><td>${pl.name ? esc(pl.name) : `<span class="pend">identity pending</span>`}${low ? ` ${conf(pl.confidence)}` : ''}</td><td class="mono">${esc(pl.role)}</td><td class="mono">${pl.alignment_at_snap.lateral > 0 ? 'L' : 'R'}${Math.abs(pl.alignment_at_snap.lateral)} · ${pl.alignment_at_snap.depth}</td><td>${esc(doing)}</td></tr>`; }).join('')}</tbody></table></div>`;
    const off = r.players.filter((x) => x.side === 'offense'), def = r.players.filter((x) => x.side === 'defense');
    return side('Offense', off, teamOf(g, off[0].team)) + side('Defense', def, teamOf(g, def[0].team)) + '<span class="src">At snap: lateral yards left (L) or right (R) of the ball · depth in yards behind the line (negative = defense side). Distances in yards. Click a row to follow a player.</span>';
  }

  /* chat: scripted from the record, so it works for every featured play and says so */
  const chatLog = {};
  function panelChat() {
    return `<div class="chat"><div class="chat-log" id="chat-log"></div>
      <div class="chat-sugg" id="chat-sugg">${['What formation is this?', 'Who motioned?', 'What coverage is the defense in?', 'How open was the receiver?', 'Was the QB pressured?', 'Why did this play work?', 'Show me the JSON'].map((q) => `<button type="button">${esc(q)}</button>`).join('')}</div>
      <form class="chat-form" id="chat-form"><input id="chat-input" type="text" placeholder="Ask about this play…" autocomplete="off"><button type="submit">Ask</button></form>
      <div class="chat-note">Scripted for the demo from the play record. The real film room runs the same chat brain the repo already has, over the same records.</div></div>`;
  }
  function wireChat() {
    const log = $('#chat-log'); const uid = state.play.uid; chatLog[uid] = chatLog[uid] || [{ who: 'bot', html: `I have the record for <strong>${esc(fmtDD(state.play))} at ${esc(state.play.yrdln || '')}</strong>. Ask about the formation, motion, coverage, routes, separation, pressure, the result, or any player by number.` }];
    const draw = () => { log.innerHTML = chatLog[uid].map((m) => `<div class="msg ${m.who}">${m.html}</div>`).join(''); log.scrollTop = log.scrollHeight; };
    draw();
    const ask = (q) => { chatLog[uid].push({ who: 'user', html: esc(q) }); draw(); setTimeout(() => { chatLog[uid].push({ who: 'bot', html: answer(q) }); draw(); }, 350); };
    $('#chat-sugg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) ask(b.textContent); });
    $('#chat-form').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#chat-input').value.trim(); if (v) { $('#chat-input').value = ''; ask(v); } });
  }
  const INSIGHT = {
    '2025_wk20_LA-CHI_p001': 'The play fake pulled both inside linebackers and the strong safety a step toward the line. Moore ran his shallow cross into the space they vacated, caught it on the move, and with the corner on his side bailing to a deep third there was nobody underneath. The 34 yards were mostly after the catch.',
    '2025_wk20_LA-CHI_p019': 'Duo off the right side: the right tackle and tight end doubled the edge, the right guard sealed the 3-technique, and the back read the linebacker and pressed the C gap. Man coverage outside meant the corners were occupied by receivers; nobody was free to fill.',
    '2025_wk20_LA-CHI_p062': 'The jet motion took the nickel with it, which told Stafford it was man coverage before the snap. Higbee ran a corner against a corner trailing inside; with the single high safety held in the middle by the dig, the throw went to the sideline where only the receiver could get it.',
  };
  function answer(q) {
    const r = record(); const p = state.play; const s = q.toLowerCase();
    if (/json|export|record/.test(s)) { setTimeout(() => setTab('json'), 600); return 'Opening the JSON for this play. Every field carries its source and a confidence.'; }
    if (!r) return `For this play I only have the official play-by-play: <em>${esc(p.desc)}</em> ${p.formation ? `The official formation label is ${esc(p.formation)}${p.personnel ? ` with ${esc(p.personnel)} personnel` : ''}.` : ''} Player positions, motion, routes and separation arrive when the vision pipeline runs on it.`;
    const off = teamOf(state.game, p.posteam).color;
    const li = (id, text) => `<li><strong style="color:${off}">#${esc((playerById(r, id) || {}).jersey)}</strong> ${text}</li>`;
    const m = /#?\b(\d{1,2})\b/.exec(s);
    if (/who is|who's|number|#/.test(s) && m) { const pl = r.players.find((x) => x.jersey === m[1]); if (pl) return `<strong>#${esc(pl.jersey)} ${esc(pl.name || 'identity pending')}</strong>, ${esc(pl.role)}${pl.position_roster !== pl.role ? ` (${esc(pl.position_roster)})` : ''}, lined up ${Math.abs(pl.alignment_at_snap.lateral)} yards ${pl.alignment_at_snap.lateral > 0 ? 'left' : 'right'} of the ball${pl.route ? `, ran a ${esc(ROUTE_WORDS[pl.route] || pl.route)}` : ''}${pl.technique ? `, aligned as a ${esc(pl.technique)}-technique` : ''}${pl.coverage ? `, ${esc(pl.coverage.assignment.replace(/_/g, ' '))}` : ''}. Max speed ${pl.summary.max_speed_yds_s} yd/s.`; return `No #${m[1]} on the field for this play.`; }
    if (/open|separat|space|window/.test(s)) { const t = r.live.routes.find((x) => x.targeted); if (!t) return 'Run play, so no target to measure separation on. Ask about the gap or the pursuit instead.'; const pm = playerById(r, t.track_id).metrics; return `${esc(jersey(r, t.track_id))} had <strong>${pm.at_throw_yds} yards</strong> of separation when the ball left the hand (nearest: ${esc(jersey(r, pm.nearest_at_throw))}) and <strong>${pm.at_catch_yds} yards</strong> at the catch (nearest: ${esc(jersey(r, pm.nearest_at_catch))}). Least space all play: ${pm.min_live_yds} yards.`; }
    if (/pressur|rush|sack|pocket|hurr/.test(s)) { const q2 = r.live.qb.pressure; if (!q2) return 'Handoff, so no pass rush to measure.'; return `Nearest rusher got to <strong>${q2.nearest_rusher_min_yds} yards</strong>${q2.time_to_pressure_s != null ? `, first pressure ${q2.time_to_pressure_s} seconds after the snap` : ', never inside 2.5 yards'}. ${q2.rushers_within_3yds_at_throw != null ? `${q2.rushers_within_3yds_at_throw} rusher${q2.rushers_within_3yds_at_throw === 1 ? '' : 's'} within 3 yards at the throw.` : ''} Pocket read: ${esc(r.live.qb.pocket)}.`; }
    if (/close|tight|stick|stay|pursu|chase|closing/.test(s)) { const cov = r.players.filter((x) => x.metrics && x.metrics.kind === 'coverage' && x.metrics.assignment).sort((a, b) => a.metrics.mean_to_assignment_yds - b.metrics.mean_to_assignment_yds); const pur = r.players.filter((x) => x.metrics && x.metrics.pursuit).sort((a, b) => a.metrics.pursuit.min_to_carrier_yds - b.metrics.pursuit.min_to_carrier_yds).slice(0, 3); return `${cov.length ? `Tightest coverage: ${cov.map((x) => `${esc(jersey(r, x.track_id))} averaged ${x.metrics.mean_to_assignment_yds} yd from ${esc(jersey(r, x.metrics.assignment))}`).slice(0, 3).join('; ')}. ` : ''}Closest to the ball carrier: ${pur.map((x) => `${esc(jersey(r, x.track_id))} (${x.metrics.pursuit.min_to_carrier_yds} yd, closing up to ${x.metrics.pursuit.closing_speed_max_yds_s} yd/s)`).join(', ')}.`; }
    if (/formation|line ?up|personnel/.test(s)) { const pre = r.pre_snap; return `${esc(r.personnel.offense.grouping)} personnel, ${esc(pre.formation_initial.name.replace(/_/g, ' '))} with strength ${esc(pre.formation_initial.strength)}, quarterback ${esc(pre.formation_initial.qb.replace('_', ' '))}.${pre.formation_at_snap.name !== pre.formation_initial.name ? ` By the snap it became ${esc(pre.formation_at_snap.name.replace(/_/g, ' '))}.` : ''} The defense showed a ${esc(pre.defense.front)} front, ${esc(pre.defense.shell)}, ${pre.defense.box_count} in the box.`; }
    if (/motion|shift|move/.test(s)) { const pre = r.pre_snap; if (!pre.motions.length && !pre.shifts.length) return 'Nobody moved before the snap on this play.'; return `<ul>${pre.shifts.map((x) => li(x.track_id, `shifted from ${esc(x.from_slot)} to ${esc(x.to_slot)} and set`)).join('')}${pre.motions.map((x) => li(x.track_id, `${esc(x.type)} motion to the ${esc(x.direction)}, ${esc(x.at_snap.replace('_', ' '))} at the snap`)).join('')}</ul>`; }
    if (/cover|defen|blitz/.test(s)) { const d = r.live.defense; return `${esc(COVER[d.coverage.family] || d.coverage.family)}, ${esc(d.coverage.man_zone)}${d.coverage.rotation ? ` with a ${esc(d.coverage.rotation)} rotation` : ''} (confidence ${Math.round(d.coverage.confidence * 100)}%). ${d.pass_rush.length ? `${d.pass_rush.length} rushed${d.blitz ? ', including a blitz' : ', no blitz'}.` : 'Run play, so no pass rush to speak of.'} Pre-snap it was a ${esc(r.pre_snap.defense.shell)} shell with ${r.pre_snap.defense.box_count} in the box.`; }
    if (/route|receiver|target|run|gap|scheme/.test(s)) { if (r.live.run) return `Run: ${esc(jersey(r, r.live.run.ball_carrier))} through the ${esc(r.live.run.gap)} gap, ${esc(r.live.run.scheme)} scheme, for ${r.post_snap.yards_gained} yards${r.post_snap.result === 'touchdown' ? ' and the touchdown' : ''}.`; return `<ul>${r.live.routes.map((x) => li(x.track_id, `${esc(ROUTE_WORDS[x.route] || x.route)}${x.depth_at_break_yds != null ? `, breaking at ${x.depth_at_break_yds} yards` : ''}${x.targeted ? ' — targeted' : ''}`)).join('')}</ul>`; }
    if (/why|work|explain|happen|analy/.test(s)) return esc(INSIGHT[p.uid] || 'That reading is not in the record yet.');
    if (/result|yard|tackle|gain|outcome|score/.test(s)) { const po = r.post_snap; return `${esc(po.result)} for ${po.yards_gained} yards${po.first_down ? ', first down' : ''}${po.yards_after_catch != null ? `, ${po.yards_after_catch} after the catch` : ''}${po.tackle ? `. Tackled by ${po.tackle.by.map((b) => esc(jersey(r, b))).join(' and ')}` : ''}.`; }
    if (/qb|quarterback|throw|drop/.test(s)) { const qb = r.live.qb; return qb.drop === 'none' ? 'Handoff, no dropback.' : `${esc(qb.drop.replace('_', ' '))}, ${qb.time_to_throw_s} seconds from snap to throw, pocket ${esc(qb.pocket)}, ${r.live.ball.air_yards} air yards.`; }
    return 'I can answer about the formation, motion and shifts, coverage, routes or the run scheme, how open the receiver was, pressure on the quarterback, who stayed closest, the result, why the play worked, or any player by jersey number.';
  }

  /* json */
  function slimRecord(r) { const c = JSON.parse(JSON.stringify(r)); c.tracking = { fps: r.tracking.fps, frames: r.tracking.frames, snap_frame: r.tracking.snap_frame, ids: r.tracking.ids, columns: ['frame', 'track_id', 'x', 'y', 'dir'], note: `per-frame table omitted here: ${r.tracking.frames} frames × ${r.tracking.ids.length} players. Toggle "include tracking" to see it.` }; return c; }
  function officialOnlyRecord(p) {
    const g = state.game;
    return { schema_version: '1.0', play_uid: p.uid, game: { game_key: g.game_key, season: g.season, week: g.week, away: g.away.abbr, home: g.home.abbr },
      situation: { quarter: p.qtr, clock: p.clock, down: p.down, distance: p.ydstogo, possession: p.posteam, yardline_100: p.yardline_100, yrdln: p.yrdln, los_x: 10 + (100 - p.yardline_100), first_down_x: Math.min(110, 10 + (100 - p.yardline_100) + (p.ydstogo || 0)), direction_of_attack: 'right' },
      official: { play_type: p.play_type, description: p.desc, yards_gained: p.yards, first_down: p.first_down, touchdown: p.td, labels: { offense_formation: p.formation, offense_personnel: p.personnel, defense_personnel: p.defense, source: 'official' } },
      film: null, personnel: null, pre_snap: null, at_snap: null, live: null, post_snap: null, players: [], tracking: null,
      quality: { note: 'vision analysis pending; official sections populated' }, generated: { pipeline_version: 'official-only', at: '2026-09-11' } };
  }
  function highlight(json) { return esc(json).replace(/("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g, (m, str, colon, bool, num) => str ? (colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`) : bool ? `<span class="b">${bool}</span>` : `<span class="n">${num}</span>`); }
  function panelJson(r, p) {
    return `<div class="json-bar"><button type="button" id="j-copy">Copy</button>${r ? '<button type="button" id="j-track" aria-pressed="false">Include tracking</button>' : ''}<span class="st" id="j-st">${r ? 'play record v1.0' : 'official sections only'}</span></div><pre class="json" id="j-pre"></pre>`;
  }
  function wireJson(r, p) {
    let full = false;
    const text = () => JSON.stringify(r ? (full ? r : slimRecord(r)) : officialOnlyRecord(p), null, 2);
    const draw = () => { const t = text(); $('#j-pre').innerHTML = highlight(t); $('#j-st').textContent = `${r ? 'play record v1.0' : 'official sections only'} · ${(t.length / 1024).toFixed(0)} KB`; };
    draw();
    $('#j-copy').addEventListener('click', () => { navigator.clipboard.writeText(text()).then(() => { $('#j-copy').textContent = 'Copied'; setTimeout(() => { $('#j-copy').textContent = 'Copy'; }, 1200); }).catch(() => { $('#j-copy').textContent = 'Select and copy'; }); });
    const t = $('#j-track'); if (t) t.addEventListener('click', () => { full = !full; t.setAttribute('aria-pressed', full); draw(); });
  }

  window.addEventListener('hashchange', route);
  route();
})();
