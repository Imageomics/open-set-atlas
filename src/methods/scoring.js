// ═══════════════════════════════════════════════════════════════════
//  CPU-side scoring functions for OOD detection methods.
//  Pure functions parameterized by dependencies — no DOM, no Three.js.
// ═══════════════════════════════════════════════════════════════════

import { v3dot, v3sub, v3add, v3scale, v3norm, v3len } from '../math/vec3.js';
import { softmax, logsumexp } from '../math/stats.js';
import { mahalDist2 } from '../math/linalg.js';

// ─── Core scoring helpers ────────────────────────────────────────

export function makeFCLogitsFn(W, B) {
  const nC = W.length;
  return function(x) {
    const l = [];
    for (let ci = 0; ci < nC; ci++) l.push(v3dot(W[ci], x) + B[ci]);
    return l;
  };
}

export function odinScore(x, getLogits, weights, T, eps) {
  const K = weights.length;
  const logits = getLogits(x);
  const probs = softmax(logits.map(l => l / T));
  let yhat = 0;
  for (let i = 1; i < K; i++) if (probs[i] > probs[yhat]) yhat = i;

  const grad = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    grad[d] = weights[yhat][d];
    for (let i = 0; i < K; i++) grad[d] -= probs[i] * weights[i][d];
    grad[d] /= T;
  }

  const xp = v3norm([
    x[0] + eps * (grad[0] >= 0 ? 1 : -1),
    x[1] + eps * (grad[1] >= 0 ? 1 : -1),
    x[2] + eps * (grad[2] >= 0 ? 1 : -1)
  ]);

  const logits2 = getLogits(xp);
  const probs2 = softmax(logits2.map(l => l / T));
  let argmax2 = 0;
  for (let i = 1; i < K; i++) if (probs2[i] > probs2[argmax2]) argmax2 = i;
  return { score: probs2[argmax2], argmax: argmax2 };
}

export function vimResidual(x, vimU, muAll) {
  const xc = v3sub(x, muAll);
  let proj = [0, 0, 0];
  for (const u of vimU) {
    const coeff = v3dot(xc, u);
    proj = v3add(proj, v3scale(u, coeff));
  }
  return v3sub(xc, proj);
}

export function vimScoreFn(x, getLogits, vimU, muAll, alpha) {
  const logits = getLogits(x);
  const res = vimResidual(x, vimU, muAll);
  const vLogit = alpha * v3len(res);
  const augLogits = [...logits, vLogit];
  let argmax = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
  return { energy: logsumexp(augLogits), argmax };
}

export function rmdsScore(x, classes, poolInvCov, muAll) {
  let minClassDist = Infinity;
  let bestClass = 0;
  for (let ci = 0; ci < classes.length; ci++) {
    const d2 = mahalDist2(x, classes[ci].muBar, poolInvCov);
    if (d2 < minClassDist) { minClassDist = d2; bestClass = ci; }
  }
  const bgDist = mahalDist2(x, muAll, poolInvCov);
  return { score: minClassDist - bgDist, bestClass };
}

export function knnResult(x, allTrainPts, allTrainLabels, kVal, skipIdx) {
  if (skipIdx === undefined) skipIdx = -1;
  const topK = new Array(kVal).fill(Infinity);
  const topKlabels = new Array(kVal).fill(0);
  let nearestDist = Infinity, nearestLabel = 0;
  for (let i = 0; i < allTrainPts.length; i++) {
    if (i === skipIdx) continue;
    const d = 1 - v3dot(x, allTrainPts[i]);
    if (d < nearestDist) { nearestDist = d; nearestLabel = allTrainLabels[i]; }
    if (d < topK[kVal - 1]) {
      topK[kVal - 1] = d;
      topKlabels[kVal - 1] = allTrainLabels[i];
      for (let j = kVal - 1; j > 0 && topK[j] < topK[j - 1]; j--) {
        const td = topK[j]; topK[j] = topK[j - 1]; topK[j - 1] = td;
        const tl = topKlabels[j]; topKlabels[j] = topKlabels[j - 1]; topKlabels[j - 1] = tl;
      }
    }
  }
  return { kthDist: topK[kVal - 1], nearestLabel };
}

export function kdeScore(x, classes, kdeBandwidth) {
  let bestLogD = -Infinity;
  let bestClass = 0;
  for (let ci = 0; ci < classes.length; ci++) {
    const pts = classes[ci].points;
    const n = classes[ci].n;
    const dots = pts.map(p => kdeBandwidth * v3dot(x, p));
    const mx = Math.max(...dots);
    const logD = mx + Math.log(dots.reduce((s, d) => s + Math.exp(d - mx), 0)) - Math.log(n);
    if (logD > bestLogD) {
      bestLogD = logD;
      bestClass = ci;
    }
  }
  return { density: bestLogD, bestClass };
}

// Per-class threshold decision (used by cos, vmf, kent variants).
// scoreFnPerClass(x, ci) returns the raw score for class ci.
export function perClassDecide(x, nClasses, scoreFnPerClass, thresholds, scoreLabel) {
  let best = -1, bestRatio = -Infinity;
  for (let ci = 0; ci < nClasses; ci++) {
    const s = scoreFnPerClass(x, ci);
    const ratio = thresholds[ci] !== 0 ? s / thresholds[ci] : s;
    if (s >= thresholds[ci] && ratio > bestRatio) { best = ci; bestRatio = ratio; }
  }
  let nearCls = 0, maxRatio = -Infinity;
  for (let ci = 0; ci < nClasses; ci++) {
    const s = scoreFnPerClass(x, ci);
    const ratio = thresholds[ci] !== 0 ? s / thresholds[ci] : s;
    if (ratio > maxRatio) { maxRatio = ratio; nearCls = ci; }
  }
  const nearS = scoreFnPerClass(x, nearCls);
  return {
    score: scoreLabel, val: nearS, thr: thresholds[nearCls],
    lo: false, cls: nearCls, accepted: best >= 0, acceptCls: best,
  };
}

// ═══════════════════════════════════════════════════════════════════
//  Per-method scoring: decideFn, classScoresFn, scoreFn
//
//  All take (x, ctx) where ctx carries runtime state:
//    { W, TAU, classes, poolInvCov, muAll, allTrainPts, allTrainLabels,
//      kVal, kdeBandwidth, bestT, vimU, vimAlpha, fc, fcVimAlpha,
//      thr, odinT, odinEps }
//
//  decideFn  → { score, val, thr, lo, cls [, accepted, acceptCls] }
//  classScoresFn → { type, items: [{l, c, v}] } | null
//  scoreFn   → { classIdx, score }
// ═══════════════════════════════════════════════════════════════════

// ─── Shared logit helpers ────────────────────────────────────────

function protoLogits(x, ctx) {
  return ctx.W.map(w => ctx.TAU * v3dot(x, w));
}

function protoArgmax(probs) {
  let a = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[a]) a = i;
  return a;
}

function argmax(arr) {
  let a = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[a]) a = i;
  return a;
}

// ─── Method scoring implementations ─────────────────────────────

export const methodScoring = {

  // ── MSP ──
  "msp": {
    decideFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const probs = softmax(logits);
      const a = protoArgmax(probs);
      return { score: "max P", val: probs[a], thr: ctx.thr.mspGamma, lo: false, cls: a };
    },
    classScoresFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const probs = softmax(logits);
      return { type: "prob", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: probs[i] })) };
    },
    scoreFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const probs = softmax(logits);
      const a = protoArgmax(probs);
      return { classIdx: a, score: probs[a] };
    },
  },
  "fc-msp": {
    decideFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x), fp = softmax(fl);
      const a = protoArgmax(fp);
      return { score: "max P", val: fp[a], thr: ctx.thr.fcMspGamma, lo: false, cls: a };
    },
    classScoresFn(x, ctx) {
      const fp = softmax(ctx.fc.logitsFn(x));
      return { type: "prob", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: fp[i] })) };
    },
    scoreFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x), fp = softmax(fl);
      const a = protoArgmax(fp);
      return { classIdx: a, score: fp[a] };
    },
  },

  // ── MLS ──
  "mls": {
    decideFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const a = argmax(logits);
      return { score: "max logit", val: Math.max(...logits), thr: ctx.thr.mlsThr, lo: false, cls: a };
    },
    classScoresFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      return { type: "logit", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: logits[i] })) };
    },
    scoreFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const a = argmax(logits);
      return { classIdx: a, score: logits[a] };
    },
  },
  "fc-mls": {
    decideFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      const a = argmax(fl);
      return { score: "max logit", val: fl[a], thr: ctx.thr.fcMlsThr, lo: false, cls: a };
    },
    classScoresFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      return { type: "logit", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: fl[i] })) };
    },
    scoreFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      const a = argmax(fl);
      return { classIdx: a, score: fl[a] };
    },
  },

  // ── EBO ──
  "ebo": {
    decideFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const a = argmax(logits);
      return { score: "energy", val: logsumexp(logits), thr: ctx.thr.eboThr, lo: false, cls: a };
    },
    classScoresFn: null,
    scoreFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      const a = argmax(logits);
      return { classIdx: a, score: logsumexp(logits) };
    },
  },
  "fc-ebo": {
    decideFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      const a = argmax(fl);
      return { score: "energy", val: logsumexp(fl), thr: ctx.thr.fcEboThr, lo: false, cls: a };
    },
    classScoresFn: null,
    scoreFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      const a = argmax(fl);
      return { classIdx: a, score: logsumexp(fl) };
    },
  },

  // ── TempScale ──
  "ts": {
    decideFn(x, ctx) {
      const logits = ctx.W.map(w => (ctx.TAU / ctx.bestT) * v3dot(x, w));
      const probs = softmax(logits);
      const a = protoArgmax(probs);
      return { score: "max P(T)", val: probs[a], thr: ctx.thr.tsGamma, lo: false, cls: a };
    },
    classScoresFn(x, ctx) {
      const probs = softmax(ctx.W.map(w => (ctx.TAU / ctx.bestT) * v3dot(x, w)));
      return { type: "prob", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: probs[i] })) };
    },
    scoreFn(x, ctx) {
      const logits = ctx.W.map(w => (ctx.TAU / ctx.bestT) * v3dot(x, w));
      const probs = softmax(logits);
      const a = protoArgmax(probs);
      return { classIdx: a, score: probs[a] };
    },
  },
  "fc-ts": {
    decideFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      const fp = softmax(fl.map(l => l / ctx.fc.bestT));
      const a = protoArgmax(fp);
      return { score: "max P(T)", val: fp[a], thr: ctx.thr.fcTsGamma, lo: false, cls: a };
    },
    classScoresFn(x, ctx) {
      const fp = softmax(ctx.fc.logitsFn(x).map(v => v / ctx.fc.bestT));
      return { type: "prob", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: fp[i] })) };
    },
    scoreFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      const fp = softmax(fl.map(l => l / ctx.fc.bestT));
      const a = protoArgmax(fp);
      return { classIdx: a, score: fp[a] };
    },
  },

  // ── ODIN ──
  "odin": {
    decideFn(x, ctx) {
      const getLogits = x2 => protoLogits(x2, ctx);
      const weights = ctx.W.map(w => v3scale(w, ctx.TAU));
      const r = odinScore(x, getLogits, weights, ctx.odinT, ctx.odinEps);
      return { score: "ODIN P", val: r.score, thr: ctx.thr.odinGamma, lo: false, cls: r.argmax };
    },
    classScoresFn(x, ctx) {
      const probs = softmax(protoLogits(x, ctx));
      return { type: "prob", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: probs[i] })) };
    },
    scoreFn(x, ctx) {
      const getLogits = x2 => protoLogits(x2, ctx);
      const weights = ctx.W.map(w => v3scale(w, ctx.TAU));
      const r = odinScore(x, getLogits, weights, ctx.odinT, ctx.odinEps);
      return { classIdx: r.argmax, score: r.score };
    },
  },
  "fc-odin": {
    decideFn(x, ctx) {
      const r = odinScore(x, ctx.fc.logitsFn, ctx.fc.W, ctx.odinT, ctx.odinEps);
      return { score: "ODIN P", val: r.score, thr: ctx.thr.fcOdinGamma, lo: false, cls: r.argmax };
    },
    classScoresFn(x, ctx) {
      const fp = softmax(ctx.fc.logitsFn(x));
      return { type: "prob", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: fp[i] })) };
    },
    scoreFn(x, ctx) {
      const r = odinScore(x, ctx.fc.logitsFn, ctx.fc.W, ctx.odinT, ctx.odinEps);
      return { classIdx: r.argmax, score: r.score };
    },
  },

  // ── ViM ──
  "vim": {
    decideFn(x, ctx) {
      const getLogits = x2 => protoLogits(x2, ctx);
      const r = vimScoreFn(x, getLogits, ctx.vimU, ctx.muAll, ctx.vimAlpha);
      return { score: "ViM E", val: r.energy, thr: ctx.thr.vimThr, lo: false, cls: r.argmax };
    },
    classScoresFn(x, ctx) {
      const logits = protoLogits(x, ctx);
      return { type: "logit", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: logits[i] })) };
    },
    scoreFn(x, ctx) {
      const getLogits = x2 => protoLogits(x2, ctx);
      const r = vimScoreFn(x, getLogits, ctx.vimU, ctx.muAll, ctx.vimAlpha);
      return { classIdx: r.argmax, score: r.energy };
    },
  },
  "fc-vim": {
    decideFn(x, ctx) {
      const r = vimScoreFn(x, ctx.fc.logitsFn, ctx.vimU, ctx.muAll, ctx.fcVimAlpha);
      return { score: "ViM E", val: r.energy, thr: ctx.thr.fcVimThr, lo: false, cls: r.argmax };
    },
    classScoresFn(x, ctx) {
      const fl = ctx.fc.logitsFn(x);
      return { type: "logit", items: ctx.classes.map((c, i) => ({ l: c.label, c: c.color, v: fl[i] })) };
    },
    scoreFn(x, ctx) {
      const r = vimScoreFn(x, ctx.fc.logitsFn, ctx.vimU, ctx.muAll, ctx.fcVimAlpha);
      return { classIdx: r.argmax, score: r.energy };
    },
  },

  // ── Cosine ──
  "cos": {
    decideFn(x, ctx) {
      return perClassDecide(
        x, ctx.classes.length,
        (x2, ci) => v3dot(x2, ctx.classes[ci].muHat),
        ctx.thr.cosPerClass, "max cos"
      );
    },
    classScoresFn(x, ctx) {
      return { type: "ratio", items: ctx.classes.map((c, i) => ({
        l: c.label, c: c.color, v: v3dot(x, c.muHat) / ctx.thr.cosPerClass[i],
      })) };
    },
    scoreFn: null, // procedural — no score-field path
  },
  "fc-cos": {
    decideFn(x, ctx) {
      return perClassDecide(
        x, ctx.classes.length,
        (x2, ci) => v3dot(x2, ctx.fc.wDirs[ci]),
        ctx.thr.fcCosPerClass, "max cos"
      );
    },
    classScoresFn(x, ctx) {
      return { type: "ratio", items: ctx.classes.map((c, i) => ({
        l: c.label, c: c.color, v: v3dot(x, ctx.fc.wDirs[i]) / ctx.thr.fcCosPerClass[i],
      })) };
    },
    scoreFn: null,
  },

  // ── vMF ──
  "vmf": {
    decideFn(x, ctx) {
      return perClassDecide(
        x, ctx.classes.length,
        (x2, ci) => ctx.classes[ci].kappa * v3dot(x2, ctx.classes[ci].muHat),
        ctx.thr.vmfBands.map(b => b[0]), "\u03BA\u00B7cos"
      );
    },
    classScoresFn(x, ctx) {
      return { type: "ratio", items: ctx.classes.map((c, i) => ({
        l: c.label, c: c.color, v: (c.kappa * v3dot(x, c.muHat)) / ctx.thr.vmfBands[i][0],
      })) };
    },
    scoreFn: null,
  },
  "fc-vmf": {
    decideFn(x, ctx) {
      return perClassDecide(
        x, ctx.classes.length,
        (x2, ci) => ctx.fc.fcKappas[ci] * v3dot(x2, ctx.fc.wDirs[ci]),
        ctx.thr.fcVmfBands.map(b => b[0]), "\u03BA\u00B7cos"
      );
    },
    classScoresFn(x, ctx) {
      return { type: "ratio", items: ctx.classes.map((c, i) => ({
        l: c.label, c: c.color, v: (ctx.fc.fcKappas[i] * v3dot(x, ctx.fc.wDirs[i])) / ctx.thr.fcVmfBands[i][0],
      })) };
    },
    scoreFn: null,
  },

  // ── Kent ──
  "kent": {
    decideFn(x, ctx) {
      return perClassDecide(
        x, ctx.classes.length,
        (x2, ci) => ctx.classes[ci].kent.kentScore(x2),
        ctx.thr.kentBands.map(b => b[0]), "kent"
      );
    },
    classScoresFn(x, ctx) {
      return { type: "ratio", items: ctx.classes.map((c, i) => ({
        l: c.label, c: c.color, v: ctx.classes[i].kent.kentScore(x) / ctx.thr.kentBands[i][0],
      })) };
    },
    scoreFn: null,
  },
  "fc-kent": {
    decideFn(x, ctx) {
      return perClassDecide(
        x, ctx.classes.length,
        (x2, ci) => ctx.fc.fcKents[ci].kentScore(x2),
        ctx.thr.fcKentBands.map(b => b[0]), "kent"
      );
    },
    classScoresFn(x, ctx) {
      return { type: "ratio", items: ctx.classes.map((c, i) => ({
        l: c.label, c: c.color, v: ctx.fc.fcKents[i].kentScore(x) / ctx.thr.fcKentBands[i][0],
      })) };
    },
    scoreFn: null,
  },

  // ── MDS (per-class covariance) ──
  "mds": {
    decideFn(x, ctx) {
      let minD = Infinity, best = 0;
      for (let ci = 0; ci < ctx.classes.length; ci++) {
        const d = mahalDist2(x, ctx.classes[ci].muBar, ctx.classes[ci].invCov);
        if (d < minD) { minD = d; best = ci; }
      }
      return { score: "d\u00B2_M", val: minD, thr: ctx.thr.mahalPCThr, lo: true, cls: best };
    },
    classScoresFn(x, ctx) {
      return { type: "dist", items: ctx.classes.map(c => ({
        l: c.label, c: c.color, v: mahalDist2(x, c.muBar, c.invCov),
      })) };
    },
    scoreFn: null, // ellipsoid renderKind
  },

  // ── MDS (shared covariance) ──
  "mds-s": {
    decideFn(x, ctx) {
      let minD = Infinity, best = 0;
      for (let ci = 0; ci < ctx.classes.length; ci++) {
        const d = mahalDist2(x, ctx.classes[ci].muBar, ctx.poolInvCov);
        if (d < minD) { minD = d; best = ci; }
      }
      return { score: "d\u00B2_M", val: minD, thr: ctx.thr.mahalSThr, lo: true, cls: best };
    },
    classScoresFn(x, ctx) {
      return { type: "dist", items: ctx.classes.map(c => ({
        l: c.label, c: c.color, v: mahalDist2(x, c.muBar, ctx.poolInvCov),
      })) };
    },
    scoreFn: null,
  },

  // ── RMDS ──
  "rmds": {
    decideFn(x, ctx) {
      const r = rmdsScore(x, ctx.classes, ctx.poolInvCov, ctx.muAll);
      return { score: "rel d\u00B2", val: r.score, thr: ctx.thr.rmdsThr, lo: true, cls: r.bestClass };
    },
    classScoresFn: null,
    scoreFn: null, // procedural renderKind — shader computes this
  },

  // ── KNN ──
  "knn": {
    decideFn(x, ctx) {
      const r = knnResult(x, ctx.allTrainPts, ctx.allTrainLabels, ctx.kVal);
      return { score: "k-th dist", val: r.kthDist, thr: ctx.thr.knnThr, lo: true, cls: r.nearestLabel };
    },
    classScoresFn: null,
    scoreFn(x, ctx) {
      const r = knnResult(x, ctx.allTrainPts, ctx.allTrainLabels, ctx.kVal);
      return { classIdx: r.nearestLabel, score: r.kthDist };
    },
  },

  // ── KDE ──
  "kde": {
    decideFn(x, ctx) {
      const r = kdeScore(x, ctx.classes, ctx.kdeBandwidth);
      return { score: "log-den.", val: r.density, thr: ctx.thr.kdeThr, lo: false, cls: r.bestClass };
    },
    classScoresFn(x, ctx) {
      const items = ctx.classes.map((c, ci) => {
        const dots = c.points.map(p => ctx.kdeBandwidth * v3dot(x, p));
        const mx = Math.max(...dots);
        const logD = mx + Math.log(dots.reduce((s, d) => s + Math.exp(d - mx), 0)) - Math.log(c.n);
        return { l: c.label, c: c.color, v: logD };
      });
      return { type: "logit", items };
    },
    scoreFn(x, ctx) {
      const r = kdeScore(x, ctx.classes, ctx.kdeBandwidth);
      return { classIdx: r.bestClass, score: r.density };
    },
  },
};
