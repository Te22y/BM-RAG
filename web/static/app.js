"use strict";

const POSITIONS = [
  { x: 50, name: "Outside surface" },
  { x: 95, name: "Stucco / gypsum interface" },
  { x: 150, name: "Gypsum / insulation interface" },
  { x: 280, name: "Middle of insulation" },
  { x: 410, name: "Insulation / inner gypsum interface" },
];
const TM_GRID = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
const REPLAY_SECONDS = 11;
const TASK_LABELS = {
  climate: "Climate", building_type: "Building and HVAC", wall_construction: "Wall construction",
  pcm_parameters: "PCM parameters", goal: "Goal",
};

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

const fmt = {
  tokens: (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n ?? 0)),
  secs: (s) => (s >= 100 ? `${Math.round(s)} s` : `${(s ?? 0).toFixed(1)} s`),
  usd: (v) => `$${(v ?? 0).toFixed(v < 0.01 ? 4 : 3)}`,
  model: (m) => (m || "").replace(/-\d{4}-\d{2}-\d{2}$/, ""),
};

/* ── Wall section ─────────────────────────────────────────────────────── */
const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawWall(view) {
  const top = 58, bottom = 196;
  const svg = svgEl("svg", { viewBox: "0 0 515 262", role: "img",
    "aria-label": wallDescription(view) });
  const defs = svgEl("defs");
  const pattern = (id, body) => {
    const p = svgEl("pattern", { id, width: 8, height: 8, patternUnits: "userSpaceOnUse" });
    body.forEach((b) => p.append(b));
    defs.append(p);
  };
  pattern("hatch-stucco", [svgEl("rect", { width: 8, height: 8, fill: "var(--layer-a)" }),
    svgEl("circle", { cx: 2, cy: 2, r: 0.9, fill: "var(--hatch)" }),
    svgEl("circle", { cx: 6, cy: 6, r: 0.9, fill: "var(--hatch)" })]);
  pattern("hatch-gypsum", [svgEl("rect", { width: 8, height: 8, fill: "var(--layer-b)" }),
    svgEl("path", { d: "M0 8 L8 0", stroke: "var(--hatch)", "stroke-width": 0.6 })]);
  pattern("hatch-insulation", [svgEl("rect", { width: 8, height: 8, fill: "var(--layer-c)" }),
    svgEl("path", { d: "M0 4 Q2 0 4 4 T8 4", fill: "none", stroke: "var(--hatch)", "stroke-width": 0.7 })]);
  svg.append(defs);

  const layers = [
    { x: 50, w: 45, fill: "hatch-stucco", name: "Stucco" },
    { x: 95, w: 55, fill: "hatch-gypsum", name: "Gypsum" },
    { x: 150, w: 260, fill: "hatch-insulation", name: "Insulation" },
    { x: 410, w: 55, fill: "hatch-gypsum", name: "Gypsum" },
  ];
  svg.append(svgEl("text", { x: 0, y: 128, class: "side-label" }, "OUT"));
  svg.append(svgEl("text", { x: 515, y: 128, class: "side-label", "text-anchor": "end" }, "IN"));
  for (const layer of layers) {
    svg.append(svgEl("rect", { x: layer.x, y: top, width: layer.w, height: bottom - top, fill: `url(#${layer.fill})` }));
    svg.append(svgEl("rect", { x: layer.x, y: top, width: layer.w, height: bottom - top, class: "layer-edge" }));
    const cx = layer.x + layer.w / 2, cy = (top + bottom) / 2;
    const attrs = layer.w < 100
      ? { x: cx, y: cy, class: "layer-name", "text-anchor": "middle", "dominant-baseline": "middle", transform: `rotate(-90 ${cx} ${cy})` }
      : { x: cx, y: cy + (view.pos === 3 || view.optPos === 3 ? 44 : 0), class: "layer-name", "text-anchor": "middle", "dominant-baseline": "middle" };
    svg.append(svgEl("text", attrs, layer.name));
  }
  POSITIONS.forEach((p, i) => {
    svg.append(svgEl("line", { x1: p.x, x2: p.x, y1: bottom, y2: bottom + 7, class: "tick" }));
    svg.append(svgEl("text", { x: p.x, y: bottom + 20, class: "pos-label", "text-anchor": "middle" }, `pos${i}`));
  });

  if (view.optPos !== null && view.optPos !== undefined) {
    const x = POSITIONS[view.optPos].x;
    svg.append(svgEl("rect", { x: x - 12, y: top - 5, width: 24, height: bottom - top + 10, rx: 3, class: "opt-bar" }));
    svg.append(svgEl("text", { x, y: bottom + 40, class: "opt-label", "text-anchor": anchorFor(x) }, "Optimum"));
  }
  view.history.forEach((h, i) => {
    if (h === view.pos) return;
    const x = POSITIONS[h]?.x;
    if (x === undefined) return;
    svg.append(svgEl("circle", { cx: x, cy: top - 16, r: 8, class: "ghost-dot" }));
    svg.append(svgEl("text", { x, y: top - 12.5, class: "ghost-num" }, String(i + 1)));
  });
  if (view.pos !== null && view.pos !== undefined) {
    const x = POSITIONS[view.pos].x;
    svg.append(svgEl("rect", { x: x - 6, y: top - 2, width: 12, height: bottom - top + 4, rx: 2, class: "pred-bar" }));
    svg.append(svgEl("text", { x, y: top - 32, class: "pred-label", "text-anchor": anchorFor(x) }, "Agent"));
  }
  $("wall").replaceChildren(svg);
}

function anchorFor(x) {
  return x < 40 ? "start" : x > 475 ? "end" : "middle";
}

function wallDescription(view) {
  const parts = ["Wall section from outdoors to indoors: stucco, gypsum, insulation, gypsum."];
  if (view.pos !== null && view.pos !== undefined) parts.push(`Agent recommendation: ${POSITIONS[view.pos].name}.`);
  if (view.optPos !== null && view.optPos !== undefined) parts.push(`EnergyPlus optimum: ${POSITIONS[view.optPos].name}.`);
  return parts.join(" ");
}

function drawTmScale(tm, optSet) {
  $("tm-scale").replaceChildren(...TM_GRID.map((t) => el("div", {
    class: `tm-cell${t === tm ? " pred" : ""}${(optSet || []).includes(t) ? " opt" : ""}`,
  }, String(t))));
}

/* ── Trace reducer: one code path for replay and live ──────────────────── */
function newTrace() {
  return { analyst: { status: "pending" }, attempts: [], finished: false };
}

function attemptState(trace, n) {
  while (trace.attempts.length < n) {
    trace.attempts.push({
      n: trace.attempts.length + 1,
      cases: { status: "pending", n: 0, time: 0 }, rules: { status: "pending", n: 0, calls: 0, time: 0 },
      climate: { status: "pending" }, graphRules: { status: "pending" },
      papers: { status: "pending", time: 0, tokens: 0 },
      reasoner: { status: "pending" }, judge: { status: "pending" },
    });
  }
  return trace.attempts[n - 1];
}

function applyEvent(trace, e) {
  if (e.agent === "analyst") {
    if (e.kind === "agent_start") trace.analyst.status = "active";
    if (e.kind === "llm") Object.assign(trace.analyst, { model: e.model, in: e.input_tokens, out: e.output_tokens });
    if (e.kind === "agent_end") Object.assign(trace.analyst, { status: "done", time: e.elapsed_sec });
    return;
  }
  if (!e.attempt) return;
  const a = attemptState(trace, e.attempt);
  if (e.agent === "reasoner") {
    if (e.kind === "agent_start") a.cases.status = "active";
    if (e.kind === "llm" && e.caller === "case_query_rewrite") { a.cases.rewritten = true; a.cases.time += e.elapsed_sec || 0; }
    if (e.kind === "llm" && e.caller === "paper_rerank") { a.papers.tokens += e.input_tokens || 0; a.papers.model = e.model; }
    if (e.kind === "retrieval") applyRetrieval(a, e);
    if (e.kind === "llm" && e.caller === "reasoner") {
      if (a.graphRules.status !== "done") a.graphRules.status = "skipped";
      if (a.papers.status !== "done") a.papers.status = "skipped";
      Object.assign(a.reasoner, { model: e.model, in: e.input_tokens, out: e.output_tokens, time: e.elapsed_sec, chars: e.prompt_chars });
    }
    if (e.kind === "agent_end") {
      Object.assign(a.reasoner, { status: "done", pos: e.pos, tm: e.tm, confidence: e.confidence });
      a.judge.status = "active";
    }
  }
  if (e.agent === "judge" && e.kind === "agent_end") {
    Object.assign(a.judge, { status: e.verdict === "pass" ? "done" : "failed", verdict: e.verdict,
      fatal: e.n_fatal, soft: e.n_soft, time: e.elapsed_sec });
  }
}

function applyRetrieval(a, e) {
  if (e.fn === "retrieve_cases_with_fallback") {
    a.cases.n += e.n || 0; a.cases.time += e.elapsed_sec || 0;
    if (e.fallback_used) a.cases.rewritten = true;
  } else if (e.fn === "retrieve_cases") {
    a.cases.n += e.n || 0; a.cases.time += e.elapsed_sec || 0;
    a.cases.status = "done"; a.rules.status = "active";
  } else if (e.fn === "retrieve_rules") {
    a.rules.n += e.n || 0; a.rules.calls += 1; a.rules.time += e.elapsed_sec || 0;
    if (a.rules.calls === 3) { a.rules.status = "done"; a.climate.status = "active"; }
  } else if (e.template === "cases_by_climate") {
    Object.assign(a.climate, { status: "done", n: e.n || 0, time: e.elapsed_sec });
    a.reasoner.status = "active";
  } else if (e.template === "rules_by_climate_objective") {
    Object.assign(a.graphRules, { status: "done", n: e.n || 0, time: e.elapsed_sec });
  } else if (e.fn === "retrieve_paper_chunks") {
    Object.assign(a.papers, { status: "done", n: e.n || 0, time: e.elapsed_sec });
  }
}

function rowCount(n) {
  return `${n} ${n === 1 ? "row" : "rows"}`;
}

function stepRow(status, label, detail, time) {
  return el("div", { class: `step ${status}` },
    el("span", { class: "dot", "aria-hidden": "true" }),
    el("span", {}, label),
    el("span", { class: "detail" }, detail || ""),
    el("span", { class: "time" }, time ? fmt.secs(time) : ""));
}

function renderTrace(trace) {
  const nodes = [];
  const an = trace.analyst;
  nodes.push(el("div", { class: "attempt" }, stepRow(an.status, "Analyst",
    an.model ? `${fmt.model(an.model)}: task rewritten into retrieval queries` : "Rewrites the task into retrieval queries", an.time)));
  for (const a of trace.attempts) {
    const verdict = a.judge.verdict;
    const judgeDetail = !verdict ? "Checks citations, the position grid and PCM cycling" : verdict === "pass"
      ? "Pass: citations and constraints hold"
      : `Retry: ${a.judge.fatal || 0} fatal, ${a.judge.soft || 0} soft`;
    nodes.push(el("div", { class: "attempt" },
      el("div", { class: "attempt-title" }, `Attempt ${a.n}`),
      stepRow(a.cases.status, "Case vector search", a.cases.status === "pending" ? "" : `${a.cases.n} hits${a.cases.rewritten ? ", query rewritten" : ""}`, a.cases.time),
      stepRow(a.rules.status, "Rule vector search", a.rules.calls ? `${a.rules.n} hits across ${a.rules.calls} queries` : "", a.rules.time),
      stepRow(a.climate.status, "Neo4j climate cohort", a.climate.status === "done" ? rowCount(a.climate.n) : "", a.climate.time),
      stepRow(a.graphRules.status, "Neo4j rule query", a.graphRules.status === "skipped" ? "Skipped: enough rules found" : a.graphRules.status === "done" ? rowCount(a.graphRules.n) : ""),
      stepRow(a.papers.status, "Paper search + rerank", a.papers.status === "skipped" ? "Skipped: enough cases with a known optimum" : a.papers.status === "done" ? `${a.papers.n} chunks kept of 20, ${fmt.model(a.papers.model)}` : "", a.papers.time),
      stepRow(a.reasoner.status, "Reasoner", a.reasoner.pos !== undefined ? `pos${a.reasoner.pos}, Tm ${a.reasoner.tm} °C, ${fmt.model(a.reasoner.model)}, ${fmt.tokens(a.reasoner.in)} in / ${fmt.tokens(a.reasoner.out)} out` : "", a.reasoner.time),
      stepRow(a.judge.status, "Judge", judgeDetail)));
  }
  $("timeline").replaceChildren(...nodes);
}

function wallView(trace, report) {
  const done = trace.attempts.filter((a) => a.reasoner.pos !== undefined && a.reasoner.pos !== null);
  const latest = done[done.length - 1];
  return {
    pos: latest ? latest.reasoner.pos : null,
    tm: latest ? latest.reasoner.tm : null,
    history: done.map((a) => a.reasoner.pos),
    optPos: report?.truth ? report.truth.gt_pos : null,
    optSet: report?.truth ? report.truth.tied_tm_set : null,
  };
}

function refresh(trace, report, showOptimum) {
  const view = wallView(trace, showOptimum ? report : null);
  drawWall(view);
  drawTmScale(view.tm, view.optSet);
  renderTrace(trace);
}

/* ── Final report ─────────────────────────────────────────────────────── */
function stat(label, value, note, tone) {
  return el("div", {}, el("dt", {}, label),
    el("dd", { class: tone || "" }, value, note ? el("small", {}, note) : null));
}

function renderStats(report) {
  const s = report.score;
  const f = report.final;
  const items = [];
  if (s && s.status === "ok") {
    const loss = s.capture_loss;
    items.push(loss === null
      ? stat("Capture loss", "n/a", "No EnergyPlus result recorded for this cell")
      : stat("Capture loss", loss.toFixed(3), "0 is optimal", loss <= 0.15 ? "good" : loss <= 0.35 ? "warn" : "bad"));
    const posGap = Math.abs(s.pos - s.gt_pos);
    items.push(stat("Position", posGap === 0 ? "Match" : `Off by ${posGap}`, `Optimum pos${s.gt_pos}`, posGap === 0 ? "good" : posGap === 1 ? "warn" : "bad"));
    const tmGap = Math.min(...report.truth.tied_tm_set.map((t) => Math.abs(s.tm - t)));
    items.push(stat("Melting temp", tmGap === 0 ? "Match" : `${tmGap} °C off`, `Optimum ${report.truth.tied_tm_set.join(" or ")} °C`, tmGap <= 2 ? "good" : tmGap <= 4 ? "warn" : "bad"));
  } else {
    items.push(stat("Scoring", "Not scored", "Edited task, so no matching EnergyPlus optimum"));
  }
  const attempts = report.attempts.length;
  items.push(report.verdict === "pass"
    ? stat("Judge", "Pass", `Attempt ${attempts}`, "good")
    : stat("Judge", "Limit reached", `Last of ${attempts} attempts accepted`, "warn"));
  items.push(stat("Confidence", `${Math.round((f.confidence || 0) * 100)}%`, "Self-reported"));
  items.push(stat("Run", fmt.secs(report.elapsed_seconds), `About ${fmt.usd(report.usage.cost_usd_total)}`));
  $("stats").replaceChildren(...items);
}

function withCitations(text, report) {
  const ids = Object.keys(report.evidence).sort((a, b) => b.length - a.length);
  if (!ids.length) return [text];
  const pattern = new RegExp(`(?<![\\w-])(${ids.map((i) => i.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})(?![\\w-])`, "g");
  return String(text).split(pattern).map((part, i) => (i % 2
    ? el("button", { class: "cite", type: "button", onclick: () => openEvidence(part) }, part)
    : part));
}

let details = { report: null, active: "Reasoning" };
const TABS = {
  Reasoning: renderReasoning,
  Evidence: renderEvidence,
  Judge: renderJudge,
  Analyst: renderAnalyst,
  Usage: renderUsage,
};

function renderDetails(report) {
  details.report = report;
  $("details").hidden = false;
  selectDetail(details.active);
}

function selectDetail(name, focusId) {
  details.active = name;
  $("detail-tabs").replaceChildren(...Object.keys(TABS).map((tab) => el("button", {
    role: "tab", type: "button", "aria-selected": String(tab === name), onclick: () => selectDetail(tab),
  }, tab)));
  $("detail-body").replaceChildren(TABS[name](details.report));
  if (focusId) {
    const card = document.getElementById(`ev-${focusId}`);
    if (card) { card.classList.add("flash"); card.scrollIntoView({ block: "center", behavior: "smooth" }); }
  }
}

function openEvidence(id) { selectDetail("Evidence", id); }

function renderReasoning(report) {
  const f = report.final;
  const posName = POSITIONS[f.pcm_interface_pos]?.name || f.position;
  return el("div", { class: "prose" },
    el("p", {}, withCitations(f.joint_rationale || "", report)),
    el("h4", {}, "Reasoning chain"),
    el("ol", {}, (f.reasoning_chain || []).map((step) => el("li", {}, withCitations(step, report)))),
    (f.tm_reasoning || []).length ? [el("h4", {}, "Melting temperature"),
      el("ol", {}, f.tm_reasoning.map((step) => el("li", {}, withCitations(step, report))))] : null,
    (f.alternatives || []).length ? [el("h4", {}, `Alternatives to ${posName}`),
      el("ol", {}, f.alternatives.map((alt) => el("li", {},
        el("b", {}, `pos${alt.pcm_interface_pos ?? "?"}, Tm ${alt.pcm_tm_c ?? "?"} °C: `),
        withCitations(alt.why_not_preferred || alt.reason || "", report))))] : null);
}

function renderEvidence(report) {
  const cited = new Set(report.final.evidence_ids || []);
  const ids = Object.keys(report.evidence).sort((a, b) => cited.has(b) - cited.has(a));
  return el("div", {},
    el("p", { class: "hint", style: "margin-bottom:10px" }, `${cited.size} items cited in the final answer; earlier attempts' citations follow.`),
    el("div", { class: "evidence" }, ids.map((id) => {
      const ev = report.evidence[id];
      return el("article", { class: "ev", id: `ev-${id}` },
        el("div", { class: "ev-head" }, el("span", { class: "ev-id" }, id),
          el("span", { class: "ev-type" }, `${ev.type}${cited.has(id) ? ", final" : ""}`)),
        ev.climate ? el("p", {}, ev.climate) : null,
        ev.text ? el("p", {}, ev.text) : null,
        ev.favored.length ? el("p", {}, el("b", {}, "Favors: "), ev.favored.join("; ")) : null,
        ev.paper ? el("p", { class: "paper" }, ev.paper) : null,
        ev.verified ? null : el("p", { class: "warn" }, "Not verified by the Judge"));
    })));
}

function renderJudge(report) {
  return el("div", { class: "prose" }, report.attempts.map((a) => {
    const issues = [...a.fatal_issues.map((t) => ["Fatal", t]), ...a.soft_issues.map((t) => ["Soft", t])];
    return el("div", { class: "judge-item" },
      el("p", {}, el("b", {}, `Attempt ${a.attempt}: pos${a.pos}, Tm ${a.tm} °C. `),
        el("span", { class: a.verdict === "pass" ? "good" : "warn" }, a.verdict === "pass" ? "Pass" : "Retry")),
      issues.length ? issues.map(([kind, text]) => el("p", { class: "issue" }, `${kind}: ${text}`))
        : el("p", { class: "issue" }, "No issues."));
  }));
}

function renderAnalyst(report) {
  const a = report.analyst;
  return el("dl", { class: "kv" },
    el("dt", {}, "Climate keyword"), el("dd", {}, a.climate_kw || "none"),
    el("dt", {}, "Objective keyword"), el("dd", {}, a.objective_kw || "none"),
    el("dt", {}, "Materials"), el("dd", {}, (a.materials || []).join(", ")),
    el("dt", {}, "Composite query"), el("dd", {}, a.composite_query || ""),
    el("dt", {}, "Notes"), el("dd", {}, (a.notes || []).join(" ")));
}

const CALLER_NAMES = {
  analyst: "Analyst", reasoner: "Reasoner", paper_rerank: "Paper rerank",
  case_query_rewrite: "Query rewrite", stage5_query: "Paper query", judge: "Judge",
};

function renderUsage(report) {
  const rows = Object.entries(report.usage.by_caller).sort((a, b) => b[1].cost_usd - a[1].cost_usd);
  return el("div", { class: "scroll-x" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Call", "Calls", "Input", "Output", "Reasoning", "Time", "Cost"].map((h) => el("th", { scope: "col" }, h)))),
    el("tbody", {}, rows.map(([name, b]) => el("tr", {},
      el("td", {}, CALLER_NAMES[name] || name), el("td", {}, String(b.calls)),
      el("td", {}, fmt.tokens(b.input_tokens)), el("td", {}, fmt.tokens(b.output_tokens)),
      el("td", {}, fmt.tokens(b.reasoning_tokens)), el("td", {}, fmt.secs(b.elapsed_sec)), el("td", {}, fmt.usd(b.cost_usd))))),
    el("tfoot", {}, el("tr", {}, el("td", {}, "Total"), el("td", { colspan: 5 }, ""), el("td", {}, fmt.usd(report.usage.cost_usd_total))))));
}

function renderTask(input) {
  $("task-details").hidden = false;
  $("task-fields").replaceChildren(...Object.entries(TASK_LABELS).flatMap(([key, label]) => (
    input[key] ? [el("dt", {}, label), el("dd", {}, input[key])] : [])));
}

/* ── Playback ─────────────────────────────────────────────────────────── */
let playback = { token: 0, finish: null };

function finishView(report, trace) {
  refresh(trace, report, true);
  renderStats(report);
  renderDetails(report);
  const f = report.final;
  const name = POSITIONS[f.pcm_interface_pos]?.name || f.position;
  $("result-title").textContent = `${name}, Tm ${f.pcm_tm_c} °C`;
  $("run-status").textContent = "";
  $("skip-button").hidden = true;
}

function play(report) {
  const token = ++playback.token;
  const trace = newTrace();
  const events = report.events;
  const total = events.length ? events[events.length - 1].t : 0;
  const scale = Math.max(total / REPLAY_SECONDS, 1);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const finish = () => {
    if (token !== playback.token) return;
    playback.finish = null;
    events.slice(index).forEach((e) => applyEvent(trace, e));
    index = events.length;
    finishView(report, trace);
    $("replay-button").hidden = false;
  };
  let index = 0;
  playback.finish = finish;
  $("details").hidden = true;
  $("stats").replaceChildren();
  $("replay-button").hidden = true;
  $("skip-button").hidden = false;
  $("result-title").textContent = report.meta.title;
  $("run-status").textContent = report.meta.synthetic ? "Replaying a synthetic run" : `Replaying a ${fmt.secs(total)} run`;
  $("result-eyebrow").textContent = report.meta.synthetic
    ? "Synthetic example: invented data, not a real run"
    : `${report.meta.title}${report.meta.zone ? ` · ${report.meta.zone} climate` : ""}`;
  renderTask(report.input);
  refresh(trace, report, false);
  if (reduced) { finish(); return; }

  const step = () => {
    if (token !== playback.token) return;
    if (index >= events.length) { finish(); return; }
    const now = events[index].t;
    while (index < events.length && events[index].t === now) applyEvent(trace, events[index++]);
    refresh(trace, report, false);
    const next = index < events.length ? events[index].t : now;
    setTimeout(step, Math.min((next - now) / scale * 1000, 1800));
  };
  step();
}

async function selectSample(id) {
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === id)));
  history.replaceState(null, "", `#${id}`);
  const report = await getJson(`/api/samples/${encodeURIComponent(id)}`);
  details.active = "Reasoning";
  playback.report = report;
  play(report);
}

function sampleButton(sample) {
  const s = sample.score || {};
  const loss = s.capture_loss;
  const note = `pos${s.pos} / ${s.tm} °C · optimum pos${s.gt_pos} / ${s.gt_tm} °C`;
  return el("button", { class: "sample", type: "button", "data-id": sample.id, onclick: () => selectSample(sample.id) },
    el("b", {}, sample.title),
    el("span", { class: "loss", title: "Capture loss" }, loss === null || loss === undefined ? "n/a" : loss.toFixed(3)),
    el("small", {}, note));
}

/* ── Live mode ────────────────────────────────────────────────────────── */
let scenarios = [];

function fillScenario(city) {
  const scenario = scenarios.find((s) => s.city === city);
  if (!scenario) return;
  const form = $("live-form");
  for (const key of Object.keys(TASK_LABELS)) form.elements[key].value = scenario.input[key] || "";
  updateEditNote();
}

function isEdited() {
  const scenario = scenarios.find((s) => s.city === $("scenario").value);
  const form = $("live-form");
  return !scenario || Object.keys(TASK_LABELS).some((k) => form.elements[k].value.trim() !== (scenario.input[k] || "").trim());
}

function updateEditNote() {
  $("edit-note").textContent = isEdited()
    ? "Edited task: the result won't be scored, because the EnergyPlus optimum belongs to the original task."
    : "Unedited scenarios are scored against their EnergyPlus optimum.";
}

function liveFailure(message) {
  $("live-error").textContent = message;
  $("live-error").hidden = false;
  $("run-status").textContent = "Run failed";
  $("run-button").disabled = false;
}

async function startLive(event) {
  event.preventDefault();
  const form = $("live-form");
  const scenario = scenarios.find((s) => s.city === $("scenario").value);
  const task = { ...(scenario ? scenario.input : {}) };
  for (const key of Object.keys(TASK_LABELS)) task[key] = form.elements[key].value;
  $("live-error").hidden = true;
  $("run-button").disabled = true;

  const response = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ city: $("scenario").value, task }) });
  const body = await response.json();
  if (!response.ok) { liveFailure(body.error || "The run could not start."); return; }

  playback.token += 1;
  playback.finish = null;
  const trace = newTrace();
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-current", "false"));
  $("details").hidden = true;
  $("stats").replaceChildren();
  $("skip-button").hidden = true;
  $("replay-button").hidden = true;
  $("result-eyebrow").textContent = "Live run";
  $("result-title").textContent = body.scored ? $("scenario").selectedOptions[0].textContent : "Custom task";
  $("run-status").textContent = "Running: 2–5 minutes";
  renderTask(task);
  refresh(trace, null, false);

  const source = new EventSource(`/api/run/${body.id}/events`);
  source.addEventListener("trace", (msg) => {
    applyEvent(trace, JSON.parse(msg.data));
    refresh(trace, null, false);
  });
  source.addEventListener("report", (msg) => {
    source.close();
    finishView(JSON.parse(msg.data), trace);
    $("run-button").disabled = false;
  });
  source.addEventListener("failure", (msg) => {
    source.close();
    liveFailure(JSON.parse(msg.data).message);
  });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) liveFailure("Lost the connection to the demo server.");
  };
}

function selectMode(mode) {
  const live = mode === "live";
  $("tab-replay").setAttribute("aria-selected", String(!live));
  $("tab-live").setAttribute("aria-selected", String(live));
  $("panel-replay").hidden = live;
  $("panel-live").hidden = !live;
}

/* ── Start ────────────────────────────────────────────────────────────── */
async function init() {
  const [config, samples] = await Promise.all([getJson("/api/config"), getJson("/api/samples")]);
  $("mode-badge").textContent = config.live_enabled ? "Live mode on" : "Replay mode";
  $("tab-replay").addEventListener("click", () => selectMode("replay"));
  $("tab-live").addEventListener("click", () => selectMode("live"));
  $("skip-button").addEventListener("click", () => playback.finish && playback.finish());
  $("replay-button").addEventListener("click", () => playback.report && play(playback.report));

  $("sample-list").replaceChildren(...samples.map(sampleButton));
  if (config.live_enabled) {
    scenarios = await getJson("/api/scenarios");
    $("scenario").replaceChildren(...scenarios.map((s) => el("option", { value: s.city }, `${s.city} (${s.zone})`)));
    $("scenario").addEventListener("change", (e) => fillScenario(e.target.value));
    $("live-form").addEventListener("input", updateEditNote);
    $("live-form").addEventListener("submit", startLive);
    $("live-form").hidden = false;
    fillScenario(scenarios[0]?.city);
    $("run-button").disabled = config.running;
  } else {
    $("live-off").hidden = false;
  }

  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1);
    if (samples.some((s) => s.id === id) && playback.report?.meta.id !== id) selectSample(id);
  });

  const wanted = location.hash.slice(1);
  const first = samples.find((s) => s.id === wanted) || samples.find((s) => s.id === "chicago") || samples[0];
  if (first) await selectSample(first.id);
}

init().catch((error) => {
  $("result-title").textContent = "The demo could not load";
  $("run-status").textContent = error.message;
});
