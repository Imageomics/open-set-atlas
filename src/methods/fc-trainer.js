// ═══════════════════════════════════════════════════════════════════
//  FC layer training and reconstruction.
//  trainFC3D: logistic regression on S² embeddings.
//  reconstructFCVariant: rebuild FC variant from precomputed data.
//  computeFcOdinData / computeFcVimData: FC-specific threshold data.
// ═══════════════════════════════════════════════════════════════════

import { v3dot, v3norm, v3len } from '../math/vec3.js';
import { softmax, logsumexp, quantile } from '../math/stats.js';
import { kappaFromRbar3D } from '../math/spherical.js';
import { fitKent, makeKentScoreFn } from '../math/fitting.js';
import { odinScore, vimResidual, makeFCLogitsFn } from './scoring.js';

export function reconstructFCVariant(preFC) {
  const logitsFn = makeFCLogitsFn(preFC.W, preFC.B);
  const fcKents = preFC.fcKents.map(k => ({
    kappa: k.kappa, beta: k.beta,
    gamma1: k.gamma1, gamma2: k.gamma2, gamma3: k.gamma3,
    bands: k.bands,
    kentScore: makeKentScoreFn(k),
  }));
  return {
    W: preFC.W, B: preFC.B, logitsFn,
    gamma: preFC.gamma, mlsThr: preFC.mlsThr, eboThr: preFC.eboThr,
    bestT: preFC.bestT, tsGamma: preFC.tsGamma,
    wNorms: preFC.wNorms, wDirs: preFC.wDirs,
    fcCosThresholds: preFC.fcCosThresholds, fcKappas: preFC.fcKappas,
    fcVmBands: preFC.fcVmBands, fcKents,
  };
}

export function trainFC3D(classes, TAU, Q, balanced) {
  const nClasses = classes.length;
  const fcTrainX = [], fcTrainY = [];
  for (let ci = 0; ci < nClasses; ci++) {
    for (const p of classes[ci].points) { fcTrainX.push(p); fcTrainY.push(ci); }
  }
  const fcN = fcTrainX.length;
  const classWeights = classes.map(c => fcN / (nClasses * c.n));

  const W = [], B = new Array(nClasses).fill(0);
  for (let ci = 0; ci < nClasses; ci++)
    W.push([TAU * classes[ci].muHat[0], TAU * classes[ci].muHat[1], TAU * classes[ci].muHat[2]]);

  for (let iter = 0; iter < 500; iter++) {
    const gW = [], gB = new Array(nClasses).fill(0);
    for (let ci = 0; ci < nClasses; ci++) gW.push([0, 0, 0]);
    for (let si = 0; si < fcN; si++) {
      const x = fcTrainX[si], y = fcTrainY[si];
      const cw = balanced ? classWeights[y] : 1;
      const logits = [];
      for (let ci = 0; ci < nClasses; ci++) logits.push(v3dot(W[ci], x) + B[ci]);
      const probs = softmax(logits);
      for (let ci = 0; ci < nClasses; ci++) {
        const err = probs[ci] - (ci === y ? 1 : 0);
        gW[ci][0] += cw * err * x[0] / fcN;
        gW[ci][1] += cw * err * x[1] / fcN;
        gW[ci][2] += cw * err * x[2] / fcN;
        gB[ci] += cw * err / fcN;
      }
    }
    for (let ci = 0; ci < nClasses; ci++) {
      W[ci][0] -= 0.05 * gW[ci][0]; W[ci][1] -= 0.05 * gW[ci][1]; W[ci][2] -= 0.05 * gW[ci][2];
      B[ci] -= 0.05 * gB[ci];
    }
  }

  function logitsFn(x) {
    const l = [];
    for (let ci = 0; ci < nClasses; ci++) l.push(v3dot(W[ci], x) + B[ci]);
    return l;
  }

  const gamma = classes.map((c, i) =>
    quantile(c.points.map(p => softmax(logitsFn(p))[i]), Q)).reduce((a, b) => a + b, 0) / nClasses;
  const mlsThr = quantile(fcTrainX.map(p => Math.max(...logitsFn(p))), Q);
  const eboThr = quantile(fcTrainX.map(p => logsumexp(logitsFn(p))), Q);

  let bestT = 1.0, bestNLL = Infinity;
  for (let t = 0.2; t <= 10.0; t += 0.1) {
    let nll = 0;
    for (let si = 0; si < fcN; si++)
      nll -= Math.log(Math.max(softmax(logitsFn(fcTrainX[si]).map(l => l / t))[fcTrainY[si]], 1e-15));
    if (nll < bestNLL) { bestNLL = nll; bestT = t; }
  }

  const tsGamma = classes.map((c, i) =>
    quantile(c.points.map(p => softmax(logitsFn(p).map(l => l / bestT))[i]), Q)).reduce((a, b) => a + b, 0) / nClasses;
  const wDirs = W.map(w => v3norm(w));
  const fcCosThresholds = classes.map((c, ci) => quantile(c.points.map(p => v3dot(p, wDirs[ci])), Q));
  const fcKappas = classes.map((c, ci) => {
    const Rbar = c.points.map(p => v3dot(p, wDirs[ci])).reduce((a, b) => a + b, 0) / c.points.length;
    return kappaFromRbar3D(Math.max(0, Math.min(0.999, Rbar)));
  });
  const fcVmBands = classes.map((c, ci) => {
    const scores = c.points.map(p => fcKappas[ci] * v3dot(p, wDirs[ci]));
    return [0.05, 0.20, 0.50].map(q => quantile(scores, q));
  });
  const fcKents = classes.map((c, ci) => fitKent(c.points, wDirs[ci]));

  return {
    W, B, logitsFn, gamma, mlsThr, eboThr, bestT, tsGamma,
    wNorms: W.map(w => v3len(w)), wDirs, fcCosThresholds, fcKappas, fcVmBands, fcKents,
  };
}

export function computeFcOdinData(fcVariant, classes, odinT, odinEps, Q) {
  const fcWeights = fcVariant.W;
  const fcGetLogits = fcVariant.logitsFn;
  const perClass = classes.map((c, ci) =>
    c.points.map(p => {
      const r = odinScore(p, fcGetLogits, fcWeights, odinT, odinEps);
      return r.argmax === ci ? r.score : 0;
    })
  );
  const gamma = perClass.map(xs => quantile(xs, Q)).reduce((a, b) => a + b, 0) / classes.length;
  return { perClass, gamma };
}

export function computeFcVimData(fcVariant, allTrainPts, vimU, muAll, vimMeanResNorm, Q) {
  const fcGetLogits = fcVariant.logitsFn;
  const fcMaxLogits = allTrainPts.map(p => Math.max(...fcGetLogits(p)));
  const fcMeanMax = fcMaxLogits.reduce((a, b) => a + b, 0) / fcMaxLogits.length;
  const alpha = vimMeanResNorm > 1e-10 ? fcMeanMax / vimMeanResNorm : 0;
  const scores = allTrainPts.map(p => {
    const logits = fcGetLogits(p);
    const res = vimResidual(p, vimU, muAll);
    const vLogit = alpha * v3len(res);
    return logsumexp([...logits, vLogit]);
  });
  return { thr: quantile(scores, Q), alpha, scores };
}
