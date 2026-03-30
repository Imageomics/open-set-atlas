// ═══════════════════════════════════════════════════════════════════
//  Conformal prediction: calibration and prediction set computation.
//  Pure logic — no rendering (that's in shader-manager).
// ═══════════════════════════════════════════════════════════════════

import { v3dot } from '../math/vec3.js';
import { softmax, quantile } from '../math/stats.js';

export function conformalCalibrateScore(logitsFn, alpha, scoreType, classes) {
  const scores = [];
  for (let ci = 0; ci < classes.length; ci++) {
    for (const p of classes[ci].points) {
      if (scoreType === "emb") {
        scores.push(1 - v3dot(p, classes[ci].muHat));
      } else {
        const probs = softmax(logitsFn(p));
        if (scoreType === "aps") {
          const sorted = probs.map((pr, k) => ({ pr, k })).sort((a, b) => b.pr - a.pr);
          let cum = 0;
          for (const s of sorted) { cum += s.pr; if (s.k === ci) break; }
          scores.push(cum);
        } else {
          scores.push(1 - probs[ci]);
        }
      }
    }
  }
  const n = scores.length;
  const level = Math.min(1, Math.ceil((n + 1) * (1 - alpha)) / n);
  return quantile(scores, level);
}

export function conformalPredictScore(probs, cosSims, qhat, scoreType) {
  if (scoreType === "emb") {
    const thr = 1 - qhat;
    return cosSims.map((s, k) => ({ s, k })).filter(o => o.s >= thr).map(o => o.k);
  } else if (scoreType === "aps") {
    const sorted = probs.map((pr, k) => ({ pr, k })).sort((a, b) => b.pr - a.pr);
    const set = [];
    let cum = 0;
    for (const s of sorted) { set.push(s.k); cum += s.pr; if (cum >= qhat) break; }
    return set;
  } else {
    const thr = 1 - qhat;
    return probs.map((pr, k) => ({ pr, k })).filter(s => s.pr >= thr).map(s => s.k);
  }
}
