"use strict";

// ═══════════════════════════════════════════════════════════════════
//  Threshold engine: derived state that recomputes when q or FC
//  variant changes. Owns sorted score arrays and the cachedThr
//  object that every method reads from.
//
//  This is the most latency-sensitive module — allThresholds() runs
//  on every q slider tick at 30fps during play animation. Everything
//  here is O(1) quantile lookups on pre-sorted arrays.
// ═══════════════════════════════════════════════════════════════════

import { quantileSorted } from '../math/stats.js';
import { registerEffect } from './app-state.js';

let _sortedProto = null;   // sorted arrays for prototype methods
let _sortedFC = null;       // sorted arrays for current FC variant
let _cachedThr = null;      // latest threshold object

// ─── Initialization ──────────────────────────────────────────────
// Called once after data is loaded. Accepts the raw training score
// arrays and pre-sorts them for O(1) quantile lookups.

export function initSortedScores(proto, fc) {
  _sortedProto = proto;
  _sortedFC = fc;
}

export function rebuildSortedFC(buildFn) {
  _sortedFC = buildFn();
}

// ─── Core: compute all thresholds at quantile q ──────────────────
// Returns a flat object with every method's threshold. This is the
// single source of truth that rendering, decision tables, and
// descriptions all read from.

export function allThresholds(q, nClasses) {
  const sp = _sortedProto;
  const sf = _sortedFC;

  const mspGamma = sp.trainMsp.map(xs => quantileSorted(xs, q)).reduce((a, b) => a + b, 0) / nClasses;
  const mlsThr = quantileSorted(sp.mlsScores, q);
  const eboThr = quantileSorted(sp.eboScores, q);
  const tsGamma = sp.tsMsp.map(xs => quantileSorted(xs, q)).reduce((a, b) => a + b, 0) / nClasses;
  const cosPerClass = sp.cosScores.map(xs => quantileSorted(xs, q));
  const vmfBands = sp.vmScores.map(xs =>
    [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantileSorted(xs, bq))
  );
  const kentBands = sp.kentScores.map(xs =>
    [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantileSorted(xs, bq))
  );
  const mahalPCThr = quantileSorted(sp.mahalPC, 1 - q);
  const mahalSThr = quantileSorted(sp.mahalS, 1 - q);
  const knnThr = quantileSorted(sp.knnDists, 1 - q);
  const rmdsThr = quantileSorted(sp.rmds, 1 - q);
  const kdeThr = quantileSorted(sp.kdeScores, q);
  const odinGamma = sp.odinPerClass.map(xs => quantileSorted(xs, q)).reduce((a, b) => a + b, 0) / nClasses;
  const vimThr = quantileSorted(sp.vimScores, q);

  // FC thresholds
  const fcMspGamma = sf.fcMspPerClass.map(xs => quantileSorted(xs, q)).reduce((a, b) => a + b, 0) / nClasses;
  const fcMlsThr = quantileSorted(sf.fcMlsScores, q);
  const fcEboThr = quantileSorted(sf.fcEboScores, q);
  const fcTsGamma = sf.fcTsPerClass.map(xs => quantileSorted(xs, q)).reduce((a, b) => a + b, 0) / nClasses;
  const fcCosPerClass = sf.fcCosScores.map(xs => quantileSorted(xs, q));
  const fcVmfBands = sf.fcVmScores.map(xs =>
    [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantileSorted(xs, bq))
  );
  const fcKentBands = sf.fcKentScores.map(xs =>
    [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantileSorted(xs, bq))
  );
  const fcOdinGamma = sf.fcOdinSorted.map(xs => quantileSorted(xs, q)).reduce((a, b) => a + b, 0) / nClasses;
  const fcVimThr = quantileSorted(sf.fcVimSorted, q);

  return {
    mspGamma, mlsThr, eboThr, tsGamma, cosPerClass,
    vmfBands, kentBands, mahalPCThr, mahalSThr,
    knnThr, rmdsThr, kdeThr, odinGamma, vimThr,
    fcMspGamma, fcMlsThr, fcEboThr, fcTsGamma,
    fcCosPerClass, fcVmfBands, fcKentBands,
    fcOdinGamma, fcVimThr,
  };
}

export function getCachedThresholds() {
  return _cachedThr;
}

export function setCachedThresholds(thr) {
  _cachedThr = thr;
}

// ─── Effect registration ─────────────────────────────────────────
// Recompute thresholds when currentQ OR fcBalanced changes.
// When fcBalanced changes, sorted FC scores must be rebuilt FIRST
// (by the FC effect), then thresholds recomputed. The `after`
// constraint ensures ordering; the dep on both keys ensures this
// effect fires for both slider drags and FC toggles.

registerEffect("recomputeThresholds", ["currentQ", "fcBalanced"], (state) => {
  if (_sortedProto && _sortedFC) {
    _cachedThr = allThresholds(state.currentQ, _sortedProto.trainMsp.length);
  }
}, { after: ["rebuildSortedFCScores"] });
