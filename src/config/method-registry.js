// ═══════════════════════════════════════════════════════════════════
//  Method registry: the single data structure that defines every
//  OOD detection method's behavior.
//
//  Adding a new method means adding ONE entry here. No switch
//  statements to find. No shader strings to edit. No scattered
//  threshold variables to wire.
//
//  Each entry owns:
//    - Identity: key, name, group, buttonId
//    - Execution path: isFC, base method key
//    - Rendering: renderKind, shaderKey, buildUniforms, isGradient
//    - Threshold: shape, inverted, getThreshold, getPerClassThresholds
//    - Cache: cacheKey computation
//    - Scoring: decideFn, classScoresFn, scoreFn (from scoring.js)
//    - Description: template function (Phase 5)
// ═══════════════════════════════════════════════════════════════════

import { methodScoring } from '../methods/scoring.js';
import { methodUniforms } from '../methods/uniforms.js';

// ─── Render kinds ────────────────────────────────────────────────
// "procedural"  — GPU shader computes score from direction. Uniform
//                 updates only when q changes. No precomputation.
// "scoreField"  — CPU precomputes score grid → DataTexture. GPU
//                 shader thresholds it. Used for KNN/KDE.
// "conformal"   — Separate shader + calibration pipeline.
// "ellipsoid"   — Mahalanobis ellipsoid geometry (no shader).
// "none"        — Data-only view, no method overlay.

// ─── Threshold shapes ────────────────────────────────────────────
// "single"      — One global threshold. Shader uniform: `threshold`.
// "perClass"    — Per-class thresholds. Shader uniform: `perClassThr[5]`.
// "bands"       — Per-class with nested quantile bands (vMF, Kent).

const registry = new Map();

function def(entry) {
  registry.set(entry.key, Object.freeze(entry));
}

// ─── Helper: proto/FC dual definitions ───────────────────────────
// Most methods exist as both prototype and learned-FC variants.
// The FC variant uses fc.W/fc.B instead of proto W with TAU scaling.

function defProtoFC(base, overrides = {}) {
  const proto = overrides.proto || {};
  const fc = overrides.fc || {};

  // Prototype variant
  def({
    key: base.key,
    name: base.name + " (proto)",
    group: base.group,
    buttonId: "b-" + base.key,
    isFC: false,
    baseKey: base.key,
    renderKind: base.renderKind || "procedural",
    shaderKey: base.shaderKey,
    buildUniforms: proto.buildUniforms || base.buildUniforms,
    thresholdShape: base.thresholdShape || "single",
    invertedThreshold: base.invertedThreshold || false,
    isGradient: base.isGradient || false,
    getThreshold: proto.getThreshold || base.getThreshold,
    getPerClassThresholds: proto.getPerClassThresholds || null,
    cacheKey: (st) => base.key,
    ...proto,
  });

  // FC variant
  def({
    key: "fc-" + base.key,
    name: base.name + " (learned FC)",
    group: base.group,
    buttonId: "b-fc-" + base.key,
    isFC: true,
    baseKey: base.key,
    renderKind: base.renderKind || "procedural",
    shaderKey: base.shaderKey,
    buildUniforms: fc.buildUniforms || base.buildUniforms,
    thresholdShape: base.thresholdShape || "single",
    invertedThreshold: base.invertedThreshold || false,
    isGradient: base.isGradient || false,
    getThreshold: fc.getThreshold || base.getThreshold,
    getPerClassThresholds: fc.getPerClassThresholds || null,
    cacheKey: (st) => "fc-" + base.key + "-" + st.fcBalanced,
    ...fc,
  });
}

// ═══════════════════════════════════════════════════════════════════
//  SOFTMAX-BASED METHODS
// ═══════════════════════════════════════════════════════════════════

defProtoFC({
  key: "msp",
  name: "MSP",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
  getThreshold: null, // set per-variant below
}, {
  proto: { getThreshold: thr => thr.mspGamma, ...methodScoring["msp"], ...methodUniforms["msp"] },
  fc:    { getThreshold: thr => thr.fcMspGamma, ...methodScoring["fc-msp"], ...methodUniforms["fc-msp"] },
});

defProtoFC({
  key: "mls",
  name: "MLS",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
}, {
  proto: { getThreshold: thr => thr.mlsThr, ...methodScoring["mls"], ...methodUniforms["mls"] },
  fc:    { getThreshold: thr => thr.fcMlsThr, ...methodScoring["fc-mls"], ...methodUniforms["fc-mls"] },
});

defProtoFC({
  key: "ebo",
  name: "EBO",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
}, {
  proto: { getThreshold: thr => thr.eboThr, ...methodScoring["ebo"], ...methodUniforms["ebo"] },
  fc:    { getThreshold: thr => thr.fcEboThr, ...methodScoring["fc-ebo"], ...methodUniforms["fc-ebo"] },
});

defProtoFC({
  key: "ts",
  name: "TempScale",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
}, {
  proto: { getThreshold: thr => thr.tsGamma, ...methodScoring["ts"], ...methodUniforms["ts"] },
  fc:    { getThreshold: thr => thr.fcTsGamma, ...methodScoring["fc-ts"], ...methodUniforms["fc-ts"] },
});

defProtoFC({
  key: "odin",
  name: "ODIN",
  group: "softmax",
  shaderKey: "odin",
  thresholdShape: "single",
  isGradient: true,
}, {
  proto: { getThreshold: thr => thr.odinGamma, ...methodScoring["odin"], ...methodUniforms["odin"] },
  fc:    { getThreshold: thr => thr.fcOdinGamma, ...methodScoring["fc-odin"], ...methodUniforms["fc-odin"] },
});

defProtoFC({
  key: "vim",
  name: "ViM",
  group: "softmax",
  shaderKey: "vim",
  thresholdShape: "single",
  isGradient: true,
}, {
  proto: { getThreshold: thr => thr.vimThr, ...methodScoring["vim"], ...methodUniforms["vim"] },
  fc:    { getThreshold: thr => thr.fcVimThr, ...methodScoring["fc-vim"], ...methodUniforms["fc-vim"] },
});

// ═══════════════════════════════════════════════════════════════════
//  GEOMETRIC / DIRECTIONAL METHODS
// ═══════════════════════════════════════════════════════════════════

defProtoFC({
  key: "cos",
  name: "Cosine",
  group: "directional",
  shaderKey: "cos-vmf",
  thresholdShape: "perClass",
}, {
  proto: { getPerClassThresholds: thr => thr.cosPerClass, ...methodScoring["cos"], ...methodUniforms["cos"] },
  fc:    { getPerClassThresholds: thr => thr.fcCosPerClass, ...methodScoring["fc-cos"], ...methodUniforms["fc-cos"] },
});

defProtoFC({
  key: "vmf",
  name: "vMF",
  group: "directional",
  shaderKey: "cos-vmf",
  thresholdShape: "bands",
  isGradient: true,
}, {
  proto: { getPerClassThresholds: thr => thr.vmfBands.map(b => b[0]), ...methodScoring["vmf"], ...methodUniforms["vmf"] },
  fc:    { getPerClassThresholds: thr => thr.fcVmfBands.map(b => b[0]), ...methodScoring["fc-vmf"], ...methodUniforms["fc-vmf"] },
});

defProtoFC({
  key: "kent",
  name: "Kent",
  group: "directional",
  shaderKey: "kent",
  thresholdShape: "bands",
  isGradient: true,
}, {
  proto: { getPerClassThresholds: thr => thr.kentBands.map(b => b[0]), ...methodScoring["kent"], ...methodUniforms["kent"] },
  fc:    { getPerClassThresholds: thr => thr.fcKentBands.map(b => b[0]), ...methodScoring["fc-kent"], ...methodUniforms["fc-kent"] },
});

// ═══════════════════════════════════════════════════════════════════
//  DISTANCE-BASED METHODS (prototype only)
// ═══════════════════════════════════════════════════════════════════

def({
  key: "mds",
  name: "Mahalanobis (per-class)",
  group: "distance",
  buttonId: "b-mds",
  isFC: false,
  baseKey: "mds",
  renderKind: "ellipsoid",
  shaderKey: null,
  thresholdShape: "single",
  invertedThreshold: true,
  isGradient: false,
  getThreshold: thr => thr.mahalPCThr,
  getPerClassThresholds: null,
  cacheKey: () => "mds",
  ...methodScoring["mds"],
});

def({
  key: "mds-s",
  name: "Mahalanobis (shared)",
  group: "distance",
  buttonId: "b-mds-s",
  isFC: false,
  baseKey: "mds-s",
  renderKind: "ellipsoid",
  shaderKey: null,
  thresholdShape: "single",
  invertedThreshold: true,
  isGradient: false,
  getThreshold: thr => thr.mahalSThr,
  getPerClassThresholds: null,
  cacheKey: () => "mds-s",
  ...methodScoring["mds-s"],
});

def({
  key: "rmds",
  name: "RMDS",
  group: "distance",
  buttonId: "b-rmds",
  isFC: false,
  baseKey: "rmds",
  renderKind: "procedural",
  shaderKey: "rmds",
  thresholdShape: "single",
  invertedThreshold: true,
  isGradient: true,
  getThreshold: thr => thr.rmdsThr,
  getPerClassThresholds: null,
  cacheKey: () => "rmds",
  ...methodScoring["rmds"],
  ...methodUniforms["rmds"],
});

def({
  key: "knn",
  name: "KNN",
  group: "distance",
  buttonId: "b-knn",
  isFC: false,
  baseKey: "knn",
  renderKind: "scoreField",
  shaderKey: "score-field-a",
  thresholdShape: "single",
  invertedThreshold: true,
  isGradient: true,
  getThreshold: thr => thr.knnThr,
  getPerClassThresholds: null,
  cacheKey: () => "knn",
  ...methodScoring["knn"],
});

def({
  key: "kde",
  name: "KDE",
  group: "distance",
  buttonId: "b-kde",
  isFC: false,
  baseKey: "kde",
  renderKind: "scoreField",
  shaderKey: "score-field-a",
  thresholdShape: "single",
  invertedThreshold: false,
  isGradient: true,
  getThreshold: thr => thr.kdeThr,
  getPerClassThresholds: null,
  cacheKey: () => "kde",
  ...methodScoring["kde"],
});

// ═══════════════════════════════════════════════════════════════════
//  CONFORMAL PREDICTION
// ═══════════════════════════════════════════════════════════════════

def({
  key: "cp-proto",
  name: "Conformal (proto)",
  group: "conformal",
  buttonId: "b-cp-proto",
  isFC: false,
  baseKey: "cp-proto",
  renderKind: "conformal",
  shaderKey: null,  // determined by cpScoreType at runtime
  thresholdShape: "single",
  invertedThreshold: false,
  isGradient: false,
  getThreshold: null,
  getPerClassThresholds: null,
  cacheKey: (st) => "cp-proto-" + st.cpScoreType,
});

def({
  key: "cp-fc",
  name: "Conformal (learned FC)",
  group: "conformal",
  buttonId: "b-cp-fc",
  isFC: true,
  baseKey: "cp-fc",
  renderKind: "conformal",
  shaderKey: null,
  thresholdShape: "single",
  invertedThreshold: false,
  isGradient: false,
  getThreshold: null,
  getPerClassThresholds: null,
  cacheKey: (st) => "cp-fc-" + st.cpScoreType + "-" + st.fcBalanced,
});

// ═══════════════════════════════════════════════════════════════════
//  DATA-ONLY VIEW
// ═══════════════════════════════════════════════════════════════════

def({
  key: "none",
  name: "Data only",
  group: "none",
  buttonId: "b-none",
  isFC: false,
  baseKey: "none",
  renderKind: "none",
  shaderKey: null,
  thresholdShape: "single",
  invertedThreshold: false,
  isGradient: false,
  getThreshold: () => 0,
  getPerClassThresholds: null,
  cacheKey: () => "none",
});

// ═══════════════════════════════════════════════════════════════════
//  REGISTRY API
// ═══════════════════════════════════════════════════════════════════

export function getMethod(key) {
  return registry.get(key) || null;
}

export function allMethods() {
  return [...registry.values()];
}

export function allKeys() {
  return [...registry.keys()];
}

export function methodsByGroup(group) {
  return [...registry.values()].filter(m => m.group === group);
}

export function fcMethods() {
  return [...registry.values()].filter(m => m.isFC);
}

export function proceduralMethods() {
  return [...registry.values()].filter(m => m.renderKind === "procedural");
}

export function scoreFieldMethods() {
  return [...registry.values()].filter(m => m.renderKind === "scoreField");
}

export function conformalMethods() {
  return [...registry.values()].filter(m => m.renderKind === "conformal");
}

// ─── Dispatch helpers (replace switch statements) ────────────────

export function getThresholdForMethod(key, thr) {
  const m = registry.get(key);
  if (!m || !m.getThreshold) return 0;
  return m.getThreshold(thr);
}

export function getPerClassThresholdsForMethod(key, thr) {
  const m = registry.get(key);
  if (!m || !m.getPerClassThresholds) return [0, 0, 0, 0, 0];
  return m.getPerClassThresholds(thr);
}

export function isInverted(key) {
  const m = registry.get(key);
  return m ? m.invertedThreshold : false;
}

export function isGradient(key) {
  const m = registry.get(key);
  return m ? m.isGradient : false;
}

export function isPerClass(key) {
  const m = registry.get(key);
  return m ? (m.thresholdShape === "perClass" || m.thresholdShape === "bands") : false;
}
