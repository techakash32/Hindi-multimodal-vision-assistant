"use strict";
const $ = (id) => document.getElementById(id);
const T = JSON.parse($("strings").textContent);
const ACTIONS = { detect: T.act_detect, ocr: T.act_ocr, vlm: T.act_vlm };
const KIND = { detect: T.k_detect, speech: T.k_speech, command: T.k_command, ocr: T.k_ocr, vlm: T.k_vlm, system: T.k_system, error: T.k_error };
const ZONE_WORD = { L: T.zone_L, C: T.zone_C, R: T.zone_R };
let lastEventId = 0, lastSpoken = "", failures = 0;

const dev = new Intl.NumberFormat("en-IN");
function fmtUptime(s) { const m = Math.floor(s / 60); return m < 60 ? `${dev.format(m)}:${String(s % 60).padStart(2, "0")}` : `${dev.format(Math.floor(m / 60))}:${String(m % 60).padStart(2, "0")}`; }
function clock(ts) { return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false }); }
function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }

async function post(cmd) {
  try {
    await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd }) });
  } catch (_) { /* surfaced by the state poll */ }
}
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => post(b.dataset.cmd)));
const KEYS = { " ": "read", d: "describe", c: "changed", r: "repeat", m: "mute" };
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest("button")) return;
  const cmd = KEYS[e.key.toLowerCase()];
  if (cmd) { e.preventDefault(); post(cmd); }
});

function renderZones(dets) {
  for (const z of ["L", "C", "R"]) {
    const ul = $("zone-" + z).querySelector("ul"); ul.replaceChildren();
    const here = dets.filter((d) => d.zone === z);
    if (!here.length) ul.append(el("li", "none", T.clear));
    here.forEach((d) => { const li = el("li", "chip " + d.prox, d.label); li.title = d.name_hi; ul.append(li); });
  }
}
function renderLabels(dets) {
  const box = $("labels"); box.replaceChildren();
  dets.forEach((d) => {
    const t = el("span", "tag", `${d.label} · ${ZONE_WORD[d.zone]}`);
    t.style.left = d.box[0] * 100 + "%"; t.style.top = Math.max(d.box[1] * 100, 7) + "%";
    box.append(t);
  });
}
const SVG = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) { const n = document.createElementNS(SVG, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text) n.textContent = text; return n; }
function renderRadar(dets) {
  const g = $("radar-dots"); g.replaceChildren();
  $("radar-empty").hidden = dets.length > 0;
  dets.forEach((d) => {
    const cx = (d.box[0] + d.box[2]) / 2, area = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
    const theta = ((cx - 0.5) * 70 * Math.PI) / 180;                 // ~70 degree field of view
    const r = 140 - Math.min(Math.sqrt(area) / 0.6, 1) * 100;        // big in frame = close to you
    const x = 160 + r * Math.sin(theta), y = 170 - r * Math.cos(theta);
    g.append(svgEl("circle", { cx: x.toFixed(1), cy: y.toFixed(1), r: d.prox === "near" ? 8 : 7, class: "dot-" + d.prox }));
    const anchor = x > 240 ? "end" : x < 80 ? "start" : "middle";
    g.append(svgEl("text", { x: (x + (anchor === "end" ? -12 : anchor === "start" ? 12 : 0)).toFixed(1), y: (y - 13).toFixed(1), "text-anchor": anchor, class: "dl" }, d.label));
  });
}
function renderStats(m) {
  const tb = $("stats"); tb.replaceChildren();
  for (const [k, label] of Object.entries(ACTIONS)) {
    const s = m[k] || { count: 0 }, tr = el("tr");
    const ms = (v) => (v == null ? "–" : `${dev.format(v)} ${T.ms}`);
    [label, dev.format(s.count), ms(s.p50_ms), ms(s.p95_ms)].forEach((t, i) => tr.append(el(i ? "td" : "th", "", String(t))));
    tr.firstChild.setAttribute("scope", "row"); tb.append(tr);
  }
}
function renderHealth(state) {
  const ul = $("health"); ul.replaceChildren();
  const add = (status, name, detail) => {
    const li = el("li", status); li.append(el("span", "dot"));
    const box = el("div"); box.append(el("span", "", name + (status === "ok" ? "" : " " + (status === "warn" ? T.check : T.broken))));
    if (detail) box.append(el("small", "", detail)); li.append(box); ul.append(li);
  };
  add(state.camera_ok ? "ok" : "fail", T.h_camera, state.camera_ok ? T.h_camera_ok : T.h_camera_bad);
  const t = state.services.tts;
  add(t.hindi_piper || state.demo ? "ok" : "fail", T.h_voice, state.demo ? T.h_voice_demo : (t.hindi_piper ? T.h_voice_ok : T.h_voice_bad));
  add(state.voice ? "ok" : "warn", T.h_cmd, state.voice ? T.h_cmd_on : T.h_cmd_off);
  add(state.services.ocr_circuit === "closed" ? "ok" : "warn", T.h_ocr, state.services.ocr_circuit === "closed" ? T.h_ocr_ok : T.h_ocr_bad);
  const open = Object.values(state.services.vlm_circuits).filter((v) => v === "open").length, total = Object.keys(state.services.vlm_circuits).length;
  add(open < total ? "ok" : "warn", T.h_vlm, T.h_vlm_n.replace("{ok}", dev.format(total - open)).replace("{total}", dev.format(total)));
}

async function pollState() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" }); if (!r.ok) throw new Error(r.status);
    const s = await r.json(); failures = 0; $("offline").hidden = true;
    const pill = $("status"); pill.textContent = s.busy ? s.busy + "…" : (s.camera_ok ? T.ready : T.no_camera); pill.className = "pill" + (s.busy ? " busy" : s.camera_ok ? "" : " down");
    $("fps").textContent = dev.format(Math.round(s.fps)); $("infer").textContent = dev.format(Math.round(s.infer_ms)); $("uptime").textContent = fmtUptime(s.uptime_s);
    $("nocam").hidden = s.camera_ok;
    renderZones(s.detections); renderLabels(s.detections); renderRadar(s.detections); renderStats(s.metrics); renderHealth(s);
    const mute = $("mute"); mute.setAttribute("aria-pressed", String(s.muted)); mute.firstElementChild.textContent = s.muted ? T.unmute : T.mute;
    document.querySelectorAll('[data-cmd="read"],[data-cmd="describe"]').forEach((b) => (b.disabled = !!s.busy));
    $("progress").hidden = !s.busy;
    if (s.last_text && s.last_text !== lastSpoken) {
      lastSpoken = s.last_text;
      $("said").textContent = s.last_display || s.last_text;
      const hi = $("said-hi"); hi.hidden = !s.last_text; hi.textContent = `${T.spoken_label}: ${s.last_text}`;
      const card = $("speech"); card.classList.remove("fresh"); void card.offsetWidth; card.classList.add("fresh");
    }
  } catch (_) {
    if (++failures >= 3) { $("offline").hidden = false; const p = $("status"); p.textContent = T.disconnected; p.className = "pill down"; }
  }
}
async function pollEvents() {
  try {
    const r = await fetch("/api/events?since=" + lastEventId, { cache: "no-store" }); if (!r.ok) return;
    const { events } = await r.json(); if (!events.length) return;
    const ol = $("events"); ol.querySelector(".empty")?.remove();
    for (const e of events) {
      lastEventId = Math.max(lastEventId, e.id);
      if (e.kind === "speech") continue; // already shown in the speech card
      const li = el("li", e.level); li.append(el("time", "", clock(e.ts)), el("span", "k", KIND[e.kind] || e.kind), el("span", "", e.text));
      ol.prepend(li);
    }
    while (ol.children.length > 60) ol.lastChild.remove();
  } catch (_) { /* state poll reports connectivity */ }
}
pollState(); pollEvents();
setInterval(pollState, 250); setInterval(pollEvents, 800);
