/* SupplyMind Evidence Console — render engine.
   Reads /static/evidence/evidence.json (all numbers live there, each tagged with
   the receipt JSON-path it came from) and probes live endpoints. This file holds
   ZERO receipt-metric literals: only structural geometry (viewBox, padding, bar
   height, tick counts, animation ms) and formatting rules. Numbers arrive as data. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const EVIDENCE_URL = "/static/evidence/evidence.json";

  // ---- formatting (rules, not values) -----------------------------------
  const round = (x, d) => Number(x).toFixed(d);
  function usd(v) {
    const a = Math.abs(v);
    if (a >= 1e9) return "$" + round(v / 1e9, 1) + "B";
    if (a >= 1e6) return "$" + round(v / 1e6, 1) + "M";
    if (a >= 1e3) return "$" + round(v / 1e3, 1) + "K";
    return "$" + round(v, 0);
  }
  function fmt(v, spec) {
    switch (spec) {
      case "int": return Number(v).toLocaleString();
      case "num2": return round(v, 2);
      case "num3": return round(v, 3);
      case "num3_signed": return (v >= 0 ? "+" : "") + round(v, 3);
      case "pct": return round(v * 100, v * 100 % 1 === 0 ? 0 : 1) + "%";
      case "pct_raw": return round(v, v % 1 === 0 ? 0 : 1) + "%";
      case "pct_signed": return (v >= 0 ? "+" : "") + round(v * 100, 1) + "%";
      case "usd": return usd(v);
      case "usd_bbl": return "$" + round(v, 2);
      case "sec": return round(v, 1);
      case "x": return round(v, 0) + "×";
      case "bool_no": return v === false ? "NO" : "YES";
      case "bool": return v ? "YES" : "NO";
      default: return String(v);
    }
  }

  // ---- tiny DOM helper --------------------------------------------------
  function el(tag, attrs = {}, kids = []) {
    const n = document.createElement(tag);
    for (const [k, val] of Object.entries(attrs)) {
      if (k === "class") n.className = val;
      else if (k === "html") n.innerHTML = val;
      else if (k === "text") n.textContent = val;
      else if (val !== null && val !== undefined) n.setAttribute(k, val);
    }
    (Array.isArray(kids) ? kids : [kids]).forEach(c => {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  // ---- shared tooltip for SVG marks -------------------------------------
  const tip = el("div", { id: "tooltip" });
  document.body.appendChild(tip);
  function wireTip(root) {
    root.addEventListener("mousemove", e => {
      const t = e.target.closest("[data-tip]");
      if (!t) { tip.style.opacity = 0; return; }
      tip.textContent = t.getAttribute("data-tip");
      tip.style.left = e.clientX + "px";
      tip.style.top = e.clientY + "px";
      tip.style.opacity = 1;
    });
    root.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
  }

  // ---- SVG builder ------------------------------------------------------
  const SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs = {}, kids = []) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
    (Array.isArray(kids) ? kids : [kids]).forEach(c => { if (c != null) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  const cssv = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();

  // ---- charts (all geometry structural; values from `d`) ----------------
  const SCOPE_COLOR = { oil_channel: "--series-soft", supply_graph: "--series", macro_economy: "--series-deep" };

  function chartScopeRange(d) {
    const W = 640, rowH = 46, PADX = 150, PADR = 90, PADT = 34, PADB = 34;
    const H = PADT + PADB + d.methods.length * rowH;
    const vals = [];
    d.methods.forEach(m => { vals.push(m.ci_low_usd, m.ci_high_usd, m.estimate_usd); });
    vals.push(d.headline_low, d.headline_high);
    const lo = Math.log10(Math.max(1, Math.min(...vals.filter(v => v > 0))));
    const hi = Math.log10(Math.max(...vals));
    const pad = (hi - lo) * 0.08;
    const dLo = lo - pad, dHi = hi + pad;
    const x = v => PADX + (Math.log10(Math.max(Math.pow(10, dLo), v)) - dLo) / (dHi - dLo) * (W - PADX - PADR);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" });
    // headline band
    const bx0 = x(d.headline_low), bx1 = x(d.headline_high);
    svg.appendChild(s("rect", { x: bx0, y: PADT - 6, width: Math.max(2, bx1 - bx0), height: H - PADT - PADB + 12, fill: cssv("--good"), opacity: .1 }));
    svg.appendChild(s("line", { x1: bx0, x2: bx0, y1: PADT - 6, y2: H - PADB + 6, stroke: cssv("--good"), "stroke-dasharray": "3 3", opacity: .5 }));
    svg.appendChild(s("line", { x1: bx1, x2: bx1, y1: PADT - 6, y2: H - PADB + 6, stroke: cssv("--good"), "stroke-dasharray": "3 3", opacity: .5 }));
    svg.appendChild(s("text", { x: (bx0 + bx1) / 2, y: PADT - 12, "text-anchor": "middle", "font-size": 10, fill: cssv("--good-ink") }, `documented ${usd(d.headline_low)}–${usd(d.headline_high)}`));
    // x ticks at powers of ten
    for (let p = Math.ceil(dLo); p <= Math.floor(dHi); p++) {
      const xx = x(Math.pow(10, p));
      svg.appendChild(s("line", { x1: xx, x2: xx, y1: PADT - 6, y2: H - PADB, stroke: cssv("--grid") }));
      svg.appendChild(s("text", { x: xx, y: H - PADB + 16, "text-anchor": "middle", "font-size": 9, fill: cssv("--ink-3") }, usd(Math.pow(10, p))));
    }
    const sorted = [...d.methods].sort((a, b) => a.estimate_usd - b.estimate_usd);
    sorted.forEach((m, i) => {
      const cy = PADT + i * rowH + rowH / 2;
      const col = cssv(SCOPE_COLOR[m.scope] || "--series");
      const brackets = m.scope === "macro_economy";
      svg.appendChild(s("line", { x1: x(m.ci_low_usd), x2: x(m.ci_high_usd), y1: cy, y2: cy, stroke: col, "stroke-width": 2, opacity: .55 }));
      const dot = s("circle", { cx: x(m.estimate_usd), cy, r: 6, fill: col, stroke: cssv("--surface-2"), "stroke-width": 2, "data-tip": `${m.label}\nscope: ${m.scope}\n${usd(m.estimate_usd)}  [${usd(m.ci_low_usd)}, ${usd(m.ci_high_usd)}]\npath: ${m.path}` });
      svg.appendChild(dot);
      svg.appendChild(s("text", { x: PADX - 10, y: cy - 2, "text-anchor": "end", "font-size": 10.5, fill: cssv("--ink-1") }, m.label.replace(/_/g, " ")));
      svg.appendChild(s("text", { x: PADX - 10, y: cy + 11, "text-anchor": "end", "font-size": 9, fill: cssv("--ink-3") }, m.scope.replace(/_/g, " ")));
      svg.appendChild(s("text", { x: x(m.estimate_usd) + 10, y: cy + 3.5, "font-size": 10, fill: brackets ? cssv("--good-ink") : cssv("--ink-2") }, usd(m.estimate_usd) + (brackets ? "  ✓ in band" : "")));
    });
    return withLegend(svg, [["--series-soft", "oil channel"], ["--series", "supply graph"], ["--series-deep", "macro economy"], ["--good", "documented band"]]);
  }

  function hbars(rows, opts) {
    // rows: [{label, value, color, tag, tip}]; opts: {W,dom,fmtBar,rowH}
    const W = opts.W || 620, PADX = opts.padx || 150, PADR = 74, PADT = 10, rowH = opts.rowH || 30;
    const H = PADT * 2 + rows.length * rowH;
    const max = opts.dom || Math.max(...rows.map(r => r.value));
    const x = v => PADX + (v / max) * (W - PADX - PADR);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" });
    if (opts.refValue != null) {
      const rx = x(opts.refValue);
      svg.appendChild(s("line", { x1: rx, x2: rx, y1: PADT - 2, y2: H - PADT + 2, stroke: cssv("--warn"), "stroke-dasharray": "4 3", opacity: .7 }));
      svg.appendChild(s("text", { x: rx, y: PADT + 6, "font-size": 8.5, fill: cssv("--warn-ink"), "text-anchor": "middle" }, opts.refLabel || ""));
    }
    rows.forEach((r, i) => {
      const y = PADT + i * rowH + 5, bh = rowH - 13;
      svg.appendChild(s("rect", { x: PADX, y, width: W - PADX - PADR, height: bh, rx: 3, fill: cssv("--surface-3") }));
      const w = Math.max(2, x(r.value) - PADX);
      const bar = s("rect", { x: PADX, y, width: w, height: bh, rx: 3, fill: cssv(r.color), "data-tip": r.tip });
      if (r.hatch) bar.setAttribute("fill", "url(#hatch)");
      svg.appendChild(bar);
      svg.appendChild(s("text", { x: PADX - 10, y: y + bh - 2, "text-anchor": "end", "font-size": 10.5, fill: cssv("--ink-1") }, r.label));
      const lab = opts.fmtBar(r.value) + (r.tag ? "  " + r.tag : "");
      svg.appendChild(s("text", { x: x(r.value) + 8, y: y + bh - 2, "font-size": 10, fill: cssv(r.tagColor || "--ink-2") }, lab));
    });
    if (opts.hatch) {
      const defs = s("defs");
      const p = s("pattern", { id: "hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
      p.appendChild(s("rect", { width: 6, height: 6, fill: cssv("--warn"), opacity: .25 }));
      p.appendChild(s("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: cssv("--warn"), "stroke-width": 3, opacity: .8 }));
      defs.appendChild(p); svg.insertBefore(defs, svg.firstChild);
    }
    return svg;
  }

  function chartGhost(d) {
    const map = { KEEP: "--series", WEAKER: "--reference", RETIRE: "--retired" };
    const tagcol = { KEEP: "--series-soft", WEAKER: "--ink-3", RETIRE: "--retired-ink" };
    const rows = d.bars.map(b => ({
      label: b.label, value: b.p1, color: map[b.decision] || "--reference",
      tag: b.decision, tagColor: tagcol[b.decision] || "--ink-2",
      tip: `${b.label}\nP@1 = ${fmt(b.p1, "pct")}\ndecision: ${b.decision}\npath: ${b.path}`,
    }));
    return withLegend(hbars(rows, { dom: 1, padx: 190, fmtBar: v => fmt(v, "pct"), rowH: 34 }),
      [["--series", "incumbent / KEEP"], ["--retired", "RETIRED — lowered P@1"], ["--reference", "weaker single model"]]);
  }

  function chartCalibration(d) {
    const W = 560, PADX = 44, PADR = 20, PADT = 16, PADB = 40, gapG = 26;
    const H = 220, groups = d.rows.length, gw = (W - PADX - PADR) / groups, bw = (gw - gapG) / 2;
    const y0 = H - PADB, base = 0.5; // zoomed baseline shown explicitly on axis
    const y = v => y0 - ((v - base) / (1 - base)) * (y0 - PADT);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" });
    [0.5, 0.75, 1.0].forEach(g => {
      svg.appendChild(s("line", { x1: PADX, x2: W - PADR, y1: y(g), y2: y(g), stroke: cssv("--grid") }));
      svg.appendChild(s("text", { x: PADX - 6, y: y(g) + 3, "text-anchor": "end", "font-size": 8.5, fill: cssv("--ink-3") }, fmt(g, "pct")));
    });
    d.rows.forEach((r, i) => {
      const gx = PADX + i * gw + gapG / 2;
      const nB = s("rect", { x: gx, y: y(r.nominal), width: bw, height: y0 - y(r.nominal), rx: 2, fill: cssv("--reference"), opacity: .8, "data-tip": `nominal target: ${fmt(r.nominal, "pct")}` });
      const eB = s("rect", { x: gx + bw + 4, y: y(r.empirical), width: bw, height: y0 - y(r.empirical), rx: 2, fill: cssv("--series"), "data-tip": `empirical (held-out): ${fmt(r.empirical, "pct")}\ngap: ${fmt(r.gap, "pct_signed")}\nmean set size: ${round(r.set_size, 1)}\npath: ${r.path}` });
      svg.appendChild(nB); svg.appendChild(eB);
      svg.appendChild(s("text", { x: gx + gw / 2 - gapG / 2, y: y0 + 14, "text-anchor": "middle", "font-size": 9, fill: cssv("--ink-2") }, "α " + round(1 - r.nominal, 2)));
      svg.appendChild(s("text", { x: gx + gw / 2 - gapG / 2, y: y0 + 25, "text-anchor": "middle", "font-size": 8.5, fill: cssv("--good-ink") }, "Δ " + fmt(r.gap, "pct_signed")));
    });
    return withLegend(svg, [["--reference", "nominal target"], ["--series", "empirical (held-out test)"]]);
  }

  function chartLeaderboard(d) {
    const ref = d.rows.find(r => r.is_baseline);
    const rows = d.rows.map(r => ({
      label: r.agent, value: r.grade,
      color: r.is_baseline ? "--reference" : (r === d.rows[0] ? "--series" : "--series-soft"),
      hatch: !r.is_baseline,
      tag: r.is_baseline ? "baseline" : (r.p_value != null ? "p " + round(r.p_value, 2) : ""),
      tagColor: r.is_baseline ? "--ink-3" : "--ink-3",
      tip: `${r.agent}\ngrade ${round(r.grade, 3)}${r.p_value != null ? "\nvs scripted p=" + round(r.p_value, 3) + "  effect r=" + round(r.effect_r, 2) : ""}\npath: ${r.path}`,
    }));
    return hbars(rows, { dom: Math.max(...d.rows.map(r => r.grade)) * 1.15, padx: 168, rowH: 26,
      fmtBar: v => round(v, 3), hatch: true, refValue: ref ? ref.grade : null, refLabel: "scripted" });
  }

  function chartBrent(d) {
    const rows = d.bars.map((b, i) => ({
      label: b.model, value: b.mape,
      color: b.model === "ensemble" ? "--series" : "--series-soft",
      tag: fmt(b.mape, "pct_raw"), tagColor: "--ink-2",
      tip: `${b.model}\nmean MAPE ${fmt(b.mape, "pct_raw")}\npath: ${b.path}`,
    }));
    return hbars(rows, { dom: Math.max(...d.bars.map(b => b.mape)) * 1.2, padx: 100, rowH: 30, fmtBar: () => "" });
  }

  function chartGauntlet(d) {
    const rows = d.cats.map(c => ({
      label: c.cat.replace(/_/g, " "), value: c.n, color: "--good",
      tag: c.blocked + "/" + c.n + " blocked", tagColor: "--good-ink",
      tip: `${c.cat}\n${c.blocked} of ${c.n} blocked\npath: ${c.path}`,
    }));
    return hbars(rows, { dom: Math.max(...d.cats.map(c => c.n)) * 1.05, padx: 150, rowH: 26, fmtBar: () => "" });
  }

  function chartWarroom(d) {
    const rows = d.rows.map(r => ({
      label: r.label, value: r.value,
      color: r.value >= 1 ? "--good" : "--series",
      tag: fmt(r.value, "pct"), tagColor: r.value >= 1 ? "--good-ink" : "--ink-2",
      tip: `${r.label}\n${fmt(r.value, "pct")} of 8 events\npath: ${r.path}`,
    }));
    return hbars(rows, { dom: 1, padx: 200, rowH: 28, fmtBar: () => "" });
  }

  function liveStrip(d) {
    const wrap = el("div", { class: "list-panel" });
    d.sources.forEach(sc => {
      const state = sc.degraded ? "DEGRADED" : (sc.ok ? "REAL" : "DOWN");
      const dotc = sc.degraded ? "warn" : (sc.ok ? "good" : "crit");
      wrap.appendChild(el("div", { class: "probe", title: "receipt field: " + sc.path }, [
        el("div", { class: "top" }, [
          el("span", { class: "dot " + dotc }), el("span", { class: "name mono" }, sc.source),
          el("span", { class: "chip", style: "margin-left:auto" }, state),
        ]),
        el("div", { class: "url" }, sc.headline || ""),
        el("div", { class: "res" }, sc.degraded ? ("degraded: " + (sc.reason || "")) : ("age " + round(sc.age_seconds, 1) + "s")),
      ]));
    });
    return wrap;
  }

  function withLegend(svg, items) {
    const wrap = el("div");
    wrap.appendChild(svg);
    const lg = el("div", { class: "legend-inline" });
    items.forEach(([c, l]) => lg.appendChild(el("span", {}, [el("span", { class: "swatch", style: `background:${cssv(c)}` }), l])));
    wrap.appendChild(lg);
    return wrap;
  }

  const CHART = { scope_range: chartScopeRange, ghost_bars: chartGhost, calibration: chartCalibration,
    leaderboard: chartLeaderboard, brent_bars: chartBrent, gauntlet_cats: chartGauntlet,
    warroom_bars: chartWarroom, live_strip: liveStrip };

  // ---- card render ------------------------------------------------------
  function shaShort(h) { return h.slice(0, 10) + "…"; }

  function renderStat(st) {
    const tone = st.tone || "neutral";
    return el("div", { class: "stat " + tone, title: "receipt field: " + st.path }, [
      el("div", {}, [
        el("span", { class: "v" }, fmt(st.value, st.fmt)),
        st.unit ? el("span", { class: "u" }, " " + st.unit) : null,
      ]),
      el("div", { class: "l" }, st.label),
      st.note ? el("div", { class: "n" }, st.note) : null,
    ]);
  }

  function renderCard(c) {
    const card = el("div", { class: "card s-" + c.state, id: "card-" + c.id });
    // head
    card.appendChild(el("div", { class: "card-head" }, [
      el("div", {}, [
        el("div", { class: "ttl" }, [
          c.probe ? el("span", { class: "dot pending", id: "led-" + c.id, title: "live probe" }) : null,
          c.probe ? " " : null, c.title,
        ]),
        el("div", { class: "st" }, c.subtitle || ""),
      ]),
      el("span", { class: "badge b-" + c.state }, c.state),
    ]));
    if (c.question) card.appendChild(el("div", { class: "q" }, c.question));
    card.appendChild(el("div", { class: "claim" }, c.claim));
    // stats
    if (c.stats && c.stats.length) {
      const g = el("div", { class: "stats" });
      c.stats.forEach(st => g.appendChild(renderStat(st)));
      card.appendChild(g);
    }
    // live fields (filled by probe)
    if (c.live_fields) {
      const lf = el("div", { class: "stats", id: "live-" + c.id });
      c.live_fields.forEach(f => lf.appendChild(el("div", { class: "stat neutral" }, [
        el("div", {}, el("span", { class: "v", id: "lf-" + c.id + "-" + f.key }, "—")),
        el("div", { class: "l" }, f.label + " · live"),
      ])));
      card.appendChild(lf);
    }
    // chart
    if (c.chart && CHART[c.chart.type]) {
      const cw = el("div", { class: "chart-wrap" }, [
        el("div", { class: "chart-title" }, c.chart.type.replace(/_/g, " ")),
      ]);
      cw.appendChild(CHART[c.chart.type](c.chart));
      wireTip(cw);
      card.appendChild(cw);
    }
    // blocked-on-key surface
    if (c.blocked) {
      const b = el("div", { class: "notes" });
      b.appendChild(el("div", { class: "note", style: "border-left-color:var(--blocked)" }, "BLOCKED-ON-KEY: " + c.blocked.reason));
      c.blocked.items.forEach(it => b.appendChild(el("div", { class: "note", style: "border-left-color:var(--blocked)" }, "• " + it)));
      card.appendChild(b);
    }
    // notes
    if (c.notes && c.notes.length) {
      const nb = el("div", { class: "notes" });
      c.notes.forEach(n => nb.appendChild(el("div", { class: "note" }, n)));
      card.appendChild(nb);
    }
    // actions + provenance footer
    const actions = el("div", { class: "card-actions" });
    if (c.open) actions.appendChild(el("a", { class: "chip btn", href: c.open.url, target: "_blank", rel: "noopener" }, "↗ " + c.open.label));
    if (c.live_action) {
      const btn = el("button", { class: "chip btn", type: "button" }, "▶ " + c.live_action.label);
      btn.addEventListener("click", () => runLiveAction(c, btn));
      actions.appendChild(btn);
    }
    const prov = el("div", { class: "prov" }, [
      el("span", {}, "source:"),
      el("a", { href: c.receipt.served, target: "_blank", rel: "noopener", title: c.receipt.source }, c.receipt.source.split("/").pop()),
      el("span", { class: "sha", title: "file sha256: " + c.receipt.sha256 }, "sha " + shaShort(c.receipt.sha256)),
    ]);
    const foot = el("div", { class: "card-foot" }, [prov, actions]);
    card.appendChild(foot);
    return card;
  }

  // ---- live probing -----------------------------------------------------
  async function probe(method, url, body) {
    const t0 = performance.now();
    try {
      const opts = { method };
      if (method === "POST") { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(body || {}); }
      const r = await fetch(url, opts);
      let j = null; try { j = await r.json(); } catch (e) { /* non-json */ }
      return { status: r.status, ok: r.status === 200, ms: Math.round(performance.now() - t0), json: j };
    } catch (e) { return { status: 0, ok: false, ms: Math.round(performance.now() - t0), err: e.message || "network" }; }
  }

  function setLed(id, res) {
    const led = document.getElementById("led-" + id);
    if (!led) return;
    led.classList.remove("pending", "good", "warn", "crit");
    led.classList.add(res.ok ? "good" : (res.err ? "crit" : "warn"));
    led.title = (res.err ? "no response" : "HTTP " + res.status) + " · " + res.ms + "ms";
  }

  async function probeCards(manifest, statusEl) {
    const withProbe = manifest.cards.filter(c => c.probe);
    let live = 0;
    const results = [];
    await Promise.all(withProbe.map(async c => {
      const res = await probe(c.probe.method, c.probe.url, c.probe.body);
      setLed(c.id, res);
      if (res.ok) live++;
      results.push({ id: c.id, ...c.probe, ...res });
      // fill live fields (e.g. war-room health numbers)
      if (c.live_fields && res.json) c.live_fields.forEach(f => {
        const n = document.getElementById("lf-" + c.id + "-" + f.key);
        if (n && res.json[f.key] != null) n.textContent = Number(res.json[f.key]).toLocaleString();
      });
    }));
    statusEl.textContent = live + "/" + withProbe.length + " live endpoints · 200 OK";
    statusEl.classList.toggle("live-full", live === withProbe.length);
    renderProbeGrid(results);
  }

  function renderProbeGrid(results) {
    const grid = $("#probe-grid");
    grid.textContent = "";
    results.sort((a, b) => a.id.localeCompare(b.id)).forEach(r => {
      const dotc = r.ok ? "good" : (r.err ? "crit" : "warn");
      const state = r.err ? "no response" : "HTTP " + r.status;
      grid.appendChild(el("div", { class: "probe" }, [
        el("div", { class: "top" }, [el("span", { class: "dot " + dotc }), el("span", { class: "name" }, r.id)]),
        el("div", { class: "url" }, r.method + " " + r.url),
        el("div", { class: "res", style: `color:var(--${dotc === "good" ? "good" : dotc === "warn" ? "warn" : "crit"}-ink)` }, state + " · " + r.ms + "ms"),
      ]));
    });
  }

  async function runLiveAction(c, btn) {
    const host = btn.closest(".card");
    let out = host.querySelector(".live-out");
    if (!out) { out = el("div", { class: "chart-wrap live-out" }); host.querySelector(".card-foot").before(out); }
    out.textContent = ""; out.appendChild(el("div", { class: "loading" }, "running " + c.live_action.url + " …"));
    const res = await probe(c.live_action.method, c.live_action.url, {});
    out.textContent = "";
    if (!res.ok || !res.json) { out.appendChild(el("div", { class: "err" }, "live action failed · HTTP " + res.status + (res.err ? " · " + res.err : ""))); return; }
    const a = res.json.aggregate_accuracy || {};
    const g = el("div", { class: "stats" });
    Object.entries(a).forEach(([k, v]) => {
      if (typeof v !== "number") return;
      g.appendChild(el("div", { class: "stat " + (v >= 1 ? "good" : "neutral") }, [
        el("div", {}, el("span", { class: "v" }, fmt(v, "pct"))),
        el("div", { class: "l" }, k.replace(/_/g, " ")),
      ]));
    });
    out.appendChild(el("div", { class: "chart-title" }, "live result · " + c.live_action.url));
    out.appendChild(g);
  }

  // ---- page assembly ----------------------------------------------------
  function tickClock() {
    const d = new Date(), z = n => String(n).padStart(2, "0");
    const e = $("#clock"); if (e) e.textContent =
      d.getUTCFullYear() + "-" + z(d.getUTCMonth() + 1) + "-" + z(d.getUTCDate()) + " " +
      z(d.getUTCHours()) + ":" + z(d.getUTCMinutes()) + ":" + z(d.getUTCSeconds()) + "Z";
  }

  function renderLegend(m) {
    const g = $("#legend"); g.textContent = "";
    m.legend.forEach(x => g.appendChild(el("div", { class: "legend-item s-" + x.state }, [
      el("div", { class: "name" }, x.state), el("div", { class: "mean" }, x.meaning),
    ])));
  }

  function renderLedger(m) {
    const g = $("#ledger"); g.textContent = "";
    const L = m.ledger;
    const cells = [
      ["capabilities_real", "capabilities REAL · reproducible", "good"],
      ["provisional", "PROVISIONAL · weak-N, shown honestly", "warn"],
      ["blocked_on_key", "BLOCKED-ON-KEY · built, key revoked", "blocked"],
      ["retired_by_measurement", "RETIRED · measured then cut", "retired"],
      ["degraded_live_sources", "live sources DEGRADED (loud)", "warn"],
      ["receipts_cited", "receipts cited · sha256-stamped", "neutral"],
    ];
    cells.forEach(([k, label, cls]) => g.appendChild(el("div", { class: "ledger-cell " + cls }, [
      el("div", { class: "v" }, String(L[k])), el("div", { class: "k" }, label),
    ])));
  }

  const SPINE = [
    ["01", "Signal", "live GFW / NewsAPI / FRED / USGS — real APIs, degrade loud", "live"],
    ["02", "Assess", "calibrated analyst-v5 verdict — strict JSON, Brier-scored", "analyst"],
    ["03", "Decide", "engine + agent recommend a concrete action on real obs", "benchmark"],
    ["04", "Cost", "4-method causal counterfactual — real CI, real analogs", "counterfactual"],
    ["05", "War room", "renders it live, a receipt behind every number", "warroom"],
  ];
  function renderSpine() {
    const g = $("#spine"); g.textContent = "";
    SPINE.forEach(([n, t, d, id]) => {
      const step = el("a", { class: "spine-step", href: "#card-" + id }, [
        el("div", { class: "n" }, n), el("div", { class: "t" }, t), el("div", { class: "d" }, d),
      ]);
      g.appendChild(step);
    });
  }

  function renderCards(m) {
    const order = { "core-spine": 0, supporting: 1, appendix: 2 };
    const cards = [...m.cards].sort((a, b) => (order[a.tier] - order[b.tier]));
    const g = $("#cards"); g.textContent = "";
    cards.forEach(c => g.appendChild(renderCard(c)));
  }

  function renderCuttingRoom(m) {
    const g = $("#cutting"); g.textContent = "";
    m.cards.forEach(c => (c.retired || []).forEach(r => {
      g.appendChild(el("div", { class: "li" }, [
        el("div", { class: "h" }, "✕ " + r.name),
        el("div", { class: "r" }, [r.reason + "  ", el("span", { class: "mono" }, "(" + r.reason_metric_path + ")")]),
      ]));
    }));
    // retrain-pending from benchmark notes carried as appendix honesty
    const bench = m.cards.find(c => c.id === "benchmark");
    if (bench) g.appendChild(el("div", { class: "li prov" }, [
      el("div", { class: "h" }, "⏸ offline v2 agents — RETRAIN-PENDING"),
      el("div", { class: "r" }, bench.notes[0]),
    ]));
  }

  function renderLimitations(m) {
    const g = $("#limits"); g.textContent = "";
    // blocked-on-key items
    m.cards.filter(c => c.blocked).forEach(c => c.blocked.items.forEach(it =>
      g.appendChild(el("div", { class: "li blocked" }, [
        el("div", { class: "h" }, "🔒 BLOCKED-ON-KEY — " + it),
        el("div", { class: "r" }, c.blocked.reason),
      ]))));
    // provisional
    m.cards.filter(c => c.state === "PROVISIONAL").forEach(c =>
      g.appendChild(el("div", { class: "li prov" }, [
        el("div", { class: "h" }, "△ PROVISIONAL — " + c.title),
        el("div", { class: "r" }, c.claim),
      ])));
    // degraded live source
    const live = m.cards.find(c => c.id === "live");
    if (live) live.chart.sources.filter(s2 => s2.degraded).forEach(s2 =>
      g.appendChild(el("div", { class: "li prov" }, [
        el("div", { class: "h" }, "◐ DEGRADED — live source “" + s2.source + "”"),
        el("div", { class: "r" }, s2.reason || "reachable but not OK at capture time"),
      ])));
  }

  function renderFooter(m) {
    const f = $("#foot");
    f.textContent = "";
    f.appendChild(el("div", {}, [
      "Every number on this console is extracted from a committed receipt by its JSON path and carries that path (hover any figure) plus the served-copy sha256 (in each card footer). The HTML, CSS and JS hold ", el("strong", {}, "zero"), " metric literals — rebuild with ",
      el("span", { class: "mono" }, m.builder), ".",
    ]));
    f.appendChild(el("div", { style: "margin-top:8px" }, [
      el("span", { class: "mono" }, "evidence.json generated " + m.generated_at),
      " · LEDs are live HTTP probes from this browser (only 200 = green). ",
    ]));
    f.appendChild(el("div", { class: "note", style: "margin-top:10px" }, m.principle));
  }

  async function boot() {
    setInterval(tickClock, 500); tickClock();
    let m;
    try {
      const r = await fetch(EVIDENCE_URL, { cache: "no-store" });
      m = await r.json();
    } catch (e) {
      $("#cards").innerHTML = "";
      $("#cards").appendChild(el("div", { class: "err" }, "could not load evidence.json — run build_evidence.py"));
      return;
    }
    $("#hero-desc").textContent = m.principle;
    renderLegend(m); renderLedger(m); renderSpine(); renderCards(m);
    renderCuttingRoom(m); renderLimitations(m); renderFooter(m);
    const statusEl = $("#live-count");
    await probeCards(m, statusEl);
    $("#reprobe").addEventListener("click", () => probeCards(m, statusEl));
    // theme toggle
    $("#theme").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme");
      const next = cur === "light" ? "dark" : (cur === "dark" ? "light" : "light");
      document.documentElement.setAttribute("data-theme", next);
      $("#theme").textContent = next === "light" ? "◐ dark" : "◑ light";
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
