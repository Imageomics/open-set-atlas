// ═══════════════════════════════════════════════════════════════════
//  Decision table: updateDecisionTable + renderMiniBar.
//  Renders OOD decisions for the active set of test points.
// ═══════════════════════════════════════════════════════════════════

import { softmax } from '../math/stats.js';
import { v3dot } from '../math/vec3.js';

let _d; // deps

export function initDecisionTable(deps) {
  _d = deps;
}

export function renderMiniBar(info) {
  if (!info) return '<span style="color:#9ca3af">\u2014</span>';
  const { type, items } = info;
  const sorted = [...items].sort((a, b) => type === "dist" ? a.v - b.v : b.v - a.v);
  const maxV = Math.max(...items.map(s => Math.abs(s.v)));
  return sorted.map(s => {
    let pct, valStr;
    if (type === "prob") {
      pct = s.v * 100;
      valStr = (s.v * 100).toFixed(1) + "%";
    } else if (type === "dist") {
      pct = maxV > 0 ? (1 - s.v / (maxV * 1.1)) * 100 : 0;
      valStr = s.v.toFixed(2);
    } else if (type === "ratio") {
      pct = Math.min(s.v, 1.5) / 1.5 * 100;
      valStr = s.v >= 1 ? s.v.toFixed(2) + "x \u2713" : s.v.toFixed(2) + "x";
    } else {
      pct = maxV > 0 ? (Math.max(0, s.v) / maxV) * 100 : 0;
      valStr = s.v.toFixed(2);
    }
    pct = Math.max(1, Math.min(100, pct));
    const opacity = type === "ratio" && s.v < 1 ? 0.3 : 0.7;
    return `<div class="mini-bar">` +
      `<span class="mini-bar-label" style="color:${s.c}">${s.l}</span>` +
      `<span class="mini-bar-track"><span class="mini-bar-fill" style="width:${pct.toFixed(0)}%;background:${s.c};opacity:${opacity}"></span></span>` +
      `<span class="mini-bar-val">${valStr}</span></div>`;
  }).join("");
}

export function updateDecisionTable(mode) {
  const oodDiv = _d.getOodDiv();
  const classes = _d.classes;
  const methodNames = _d.methodNames;
  const activeOodPoints = _d.getActiveOodPoints();
  const probeDir = _d.getProbeDir();
  const cpModes = _d.cpModes;

  if (mode === "none") { oodDiv.innerHTML = ""; return; }
  const isNear = activeOodPoints === _d.getNearOodPoints();
  const ptColor = "#E24B4A";
  const isRandom = activeOodPoints === _d.getRandomOodPoints();
  const setLabel = isRandom ? "Random" : (isNear ? "Boundary" : "Low-confidence");

  if (cpModes.has(mode)) {
    const fc = _d.getFC();
    const logitsFn = mode === "cp-fc" ? fc.logitsFn : _d.protoLogitsFn;
    const cpScoreType = _d.getCpScoreType();
    const cpAlpha = _d.getCpAlpha();
    const fcBalanced = _d.getFCBalanced();
    const qhat = _d.cpQhat(mode, cpScoreType, fcBalanced, cpAlpha);
    const isEmb = cpScoreType === "emb";
    const rows = activeOodPoints.map((x, k) => {
      const probs = softmax(logitsFn(x));
      const cosSims = classes.map(c => v3dot(x, c.muHat));
      const predSet = _d.conformalPredictScore(probs, cosSims, qhat, cpScoreType);
      const setHtml = predSet.length === 0
        ? '<span class="decision-rejected">&#8709; (OOD)</span>'
        : predSet.map(ci => `<span style="background:${classes[ci].color};color:#fff;padding:1px 5px;border-radius:3px;font-size:11px;font-weight:600;margin-right:2px">${classes[ci].label}</span>`).join("");
      const csInfo = isEmb
        ? { type: "cos", items: classes.map((c, i) => ({ l: c.label, c: c.color, v: cosSims[i] })) }
        : { type: "prob", items: classes.map((c, i) => ({ l: c.label, c: c.color, v: probs[i] })) };
      return `<tr>
        <td style="color:${ptColor}; font-weight:700; cursor:pointer;" class="pt-link" data-pt-idx="${k}">${k + 1}</td>
        <td>${setHtml}</td>
        <td><span class="formula">|C| = ${predSet.length}</span></td>
        <td>${predSet.length === 0 ? '<span class="decision-rejected">\u2717 OOD</span>' : '<span class="decision-accepted" style="color:' + classes[predSet[0]].color + '">\u2713 ' + predSet.map(ci => classes[ci].label).join(', ') + '</span>'}</td>
        <td>${renderMiniBar(csInfo)}</td>
      </tr>`;
    });
    if (probeDir) {
      const pProbs = softmax(logitsFn(probeDir));
      const pCos = classes.map(c => v3dot(probeDir, c.muHat));
      const pSet = _d.conformalPredictScore(pProbs, pCos, qhat, cpScoreType);
      const pSetHtml = pSet.length === 0
        ? '<span class="decision-rejected">&#8709; (OOD)</span>'
        : pSet.map(ci => `<span style="background:${classes[ci].color};color:#fff;padding:1px 5px;border-radius:3px;font-size:11px;font-weight:600;margin-right:2px">${classes[ci].label}</span>`).join("");
      const pCsInfo = isEmb
        ? { type: "cos", items: classes.map((c, i) => ({ l: c.label, c: c.color, v: pCos[i] })) }
        : { type: "prob", items: classes.map((c, i) => ({ l: c.label, c: c.color, v: pProbs[i] })) };
      rows.push(`<tr style="border-top:2px solid #00e5ff;">
        <td style="color:#00e5ff; font-weight:700">\u25C6</td>
        <td>${pSetHtml}</td>
        <td><span class="formula">|C| = ${pSet.length}</span></td>
        <td>${pSet.length === 0 ? '<span class="decision-rejected">\u2717 OOD</span>' : '<span class="decision-accepted" style="color:' + classes[pSet[0]].color + '">\u2713 ' + pSet.map(ci => classes[ci].label).join(', ') + '</span>'}</td>
        <td>${renderMiniBar(pCsInfo)}</td>
      </tr>`);
    }
    const colHeader = isEmb ? "cos(x, \u03BC\u0302\u1D62)" : "Class P(k|x)";
    oodDiv.innerHTML = `<table>
      <caption>${setLabel} decisions \u2014 ${methodNames[mode] || mode}</caption>
      <thead><tr><th>Pt</th><th>Prediction set</th><th>Size</th><th>Decision</th><th>${colHeader}</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
    return;
  }

  const oodDecision3d = _d.oodDecision3d;
  const classScores3d = _d.classScores3d;
  const rows = activeOodPoints.map((x, k) => {
    const d = oodDecision3d(x, mode);
    if (!d) return "";
    const accepted = d.accepted !== undefined ? d.accepted : (d.lo ? d.val <= d.thr : d.val >= d.thr);
    const cls = d.acceptCls !== undefined && d.acceptCls >= 0 ? d.acceptCls : d.cls;
    const label = classes[cls].label;
    const color = classes[cls].color;
    const pass = d.lo ? d.val <= d.thr : d.val >= d.thr;
    const comp = pass ? (d.lo ? "\u2264" : "\u2265") : (d.lo ? ">" : "<");
    const decisionHtml = accepted
      ? `<span class="decision-accepted" style="color:${color}">\u2713 accepted \u2192 ${label}</span>`
      : `<span class="decision-rejected">\u2717 rejected (OOD)</span>`;
    const cs = classScores3d(x, mode);
    return `<tr>
      <td style="color:${ptColor}; font-weight:700; cursor:pointer;" class="pt-link" data-pt-idx="${k}">${k + 1}</td>
      <td><span class="formula">${d.val.toFixed(3)}</span></td>
      <td><span class="formula">${d.val.toFixed(3)} ${comp} ${d.thr.toFixed(3)}</span></td>
      <td>${decisionHtml}</td>
      <td>${renderMiniBar(cs)}</td>
    </tr>`;
  });
  if (probeDir) {
    const pd = oodDecision3d(probeDir, mode);
    if (pd) {
      const pAccepted = pd.accepted !== undefined ? pd.accepted : (pd.lo ? pd.val <= pd.thr : pd.val >= pd.thr);
      const pCls = pd.acceptCls !== undefined && pd.acceptCls >= 0 ? pd.acceptCls : pd.cls;
      const pPass = pd.lo ? pd.val <= pd.thr : pd.val >= pd.thr;
      const pComp = pPass ? (pd.lo ? "\u2264" : "\u2265") : (pd.lo ? ">" : "<");
      const pDecHtml = pAccepted
        ? `<span class="decision-accepted" style="color:${classes[pCls].color}">\u2713 accepted \u2192 ${classes[pCls].label}</span>`
        : `<span class="decision-rejected">\u2717 rejected (OOD)</span>`;
      const pCs = classScores3d(probeDir, mode);
      rows.push(`<tr style="border-top:2px solid #00e5ff;">
        <td style="color:#00e5ff; font-weight:700">\u25C6</td>
        <td><span class="formula">${pd.val.toFixed(3)}</span></td>
        <td><span class="formula">${pd.val.toFixed(3)} ${pComp} ${pd.thr.toFixed(3)}</span></td>
        <td>${pDecHtml}</td>
        <td>${renderMiniBar(pCs)}</td>
      </tr>`);
    }
  }
  const scoreName = activeOodPoints.length > 0 ? (oodDecision3d(activeOodPoints[0], mode) || {}).score || "Score" : "Score";
  oodDiv.innerHTML = `<table>
    <caption>${setLabel} decisions \u2014 ${methodNames[mode] || mode}</caption>
    <thead><tr><th>Pt</th><th>${scoreName}</th><th>vs. threshold</th><th>Decision</th><th>Class scores</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}
