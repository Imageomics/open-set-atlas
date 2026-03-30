"use strict";

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
//    - Scoring: CPU-side score/decide/classScores functions
//    - Description: template function
//
//  This replaces:
//    - oodDecision3d (186-line switch, html:3534-3720)
//    - classScores3d (switch, html:3725-3777)
//    - _getProceduralConfig (280-line switch, html:2677-2960)
//    - _updateProceduralUniforms (switch, html:2950-2999)
//    - _getShaderThreshold (switch, html:3314-3332)
//    - _getShaderPerClassThresholds (switch, html:3335-3344)
//    - _isInvertedThreshold (html:3374-3376)
//    - _shaderTypeB set (html:2149)
//    - _shaderGradient set (html:2151)
//    - methodNames object (html:3523-3531)
// ═══════════════════════════════════════════════════════════════════

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
  proto: { getThreshold: thr => thr.mspGamma },
  fc:    { getThreshold: thr => thr.fcMspGamma },
});

defProtoFC({
  key: "mls",
  name: "MLS",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
}, {
  proto: { getThreshold: thr => thr.mlsThr },
  fc:    { getThreshold: thr => thr.fcMlsThr },
});

defProtoFC({
  key: "ebo",
  name: "EBO",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
}, {
  proto: { getThreshold: thr => thr.eboThr },
  fc:    { getThreshold: thr => thr.fcEboThr },
});

defProtoFC({
  key: "ts",
  name: "TempScale",
  group: "softmax",
  shaderKey: "logit",
  thresholdShape: "single",
}, {
  proto: { getThreshold: thr => thr.tsGamma },
  fc:    { getThreshold: thr => thr.fcTsGamma },
});

defProtoFC({
  key: "odin",
  name: "ODIN",
  group: "softmax",
  shaderKey: "odin",
  thresholdShape: "single",
  isGradient: true,
}, {
  proto: { getThreshold: thr => thr.odinGamma },
  fc:    { getThreshold: thr => thr.fcOdinGamma },
});

defProtoFC({
  key: "vim",
  name: "ViM",
  group: "softmax",
  shaderKey: "vim",
  thresholdShape: "single",
  isGradient: true,
}, {
  proto: { getThreshold: thr => thr.vimThr },
  fc:    { getThreshold: thr => thr.fcVimThr },
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
  proto: { getPerClassThresholds: thr => thr.cosPerClass },
  fc:    { getPerClassThresholds: thr => thr.fcCosPerClass },
});

defProtoFC({
  key: "vmf",
  name: "vMF",
  group: "directional",
  shaderKey: "cos-vmf",
  thresholdShape: "bands",
  isGradient: true,
}, {
  proto: { getPerClassThresholds: thr => thr.vmfBands.map(b => b[0]) },
  fc:    { getPerClassThresholds: thr => thr.fcVmfBands.map(b => b[0]) },
});

defProtoFC({
  key: "kent",
  name: "Kent",
  group: "directional",
  shaderKey: "kent",
  thresholdShape: "bands",
  isGradient: true,
}, {
  proto: { getPerClassThresholds: thr => thr.kentBands.map(b => b[0]) },
  fc:    { getPerClassThresholds: thr => thr.fcKentBands.map(b => b[0]) },
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
