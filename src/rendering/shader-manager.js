// ═══════════════════════════════════════════════════════════════════
//  Shader Manager: owns all GPU shader caches and pipelines.
//  Procedural shaders, score-field textures, conformal overlays,
//  background precomputation — all dispatched through the registry.
//
//  Initialized via initShaderManager(deps). deps carries runtime
//  state and THREE.js reference.  DOM manipulation is NOT done here;
//  an onReadyChange callback notifies the UI layer.
// ═══════════════════════════════════════════════════════════════════

import { getMethod } from '../config/method-registry.js';
import { vertShader } from '../shaders/vert.js';
import { logitFrag } from '../shaders/logit-frag.js';
import { cosVmfFrag } from '../shaders/cos-vmf-frag.js';
import { kentFrag } from '../shaders/kent-frag.js';
import { odinFrag } from '../shaders/odin-frag.js';
import { vimFrag } from '../shaders/vim-frag.js';
import { rmdsFrag } from '../shaders/rmds-frag.js';
import { scoreFieldAFrag } from '../shaders/score-field-a-frag.js';
// scoreFieldBFrag not used — Type B path is dead code (per-class methods use procedural shaders)
import { conformalMspEmbFrag } from '../shaders/conformal-msp-emb-frag.js';
import { conformalApsFrag } from '../shaders/conformal-aps-frag.js';
import { hexToRGB } from '../math/spherical.js';
import { dirFromUV } from '../math/spherical.js';
import { softmax } from '../math/stats.js';
import { quantileSorted } from '../math/stats.js';
import { v3dot } from '../math/vec3.js';

// ─── Shader key → GLSL fragment string ──────────────────────────
const _shaderFragMap = {
  "logit":    logitFrag,
  "cos-vmf":  cosVmfFrag,
  "kent":     kentFrag,
  "odin":     odinFrag,
  "vim":      vimFrag,
  "rmds":     rmdsFrag,
};

// ─── Module state (set by init) ─────────────────────────────────
let THREE;
let _d;  // deps object
let _proceduralCache, _shaderCache, _cpProceduralCache, _cpCalibScores;
let _shaderComputing;
let _classColorsVec3, _sharedOverlayGeo;
let _bgRunning = false;

// ═══════════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════════

export function initShaderManager(deps) {
  THREE = deps.THREE;
  _d = deps;

  _proceduralCache = new Map();
  _shaderCache = new Map();
  _cpProceduralCache = new Map();
  _cpCalibScores = new Map();
  _shaderComputing = new Set();

  _classColorsVec3 = deps.classes.map(function(c) {
    var rgb = hexToRGB(c.color);
    return new THREE.Vector3(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
  });

  _sharedOverlayGeo = new THREE.SphereGeometry(1.005, 128, 64);
}

// ─── Shader context for buildUniforms ───────────────────────────

function _shaderCtx() {
  return {
    vec3: (x, y, z) => new THREE.Vector3(x, y, z),
    mat3Set: function() {
      var m = new THREE.Matrix3();
      m.set.apply(m, arguments);
      return m;
    },
    classColors: _classColorsVec3,
    W: _d.W, TAU: _d.TAU, classes: _d.classes,
    poolInvCov: _d.poolInvCov, muAll: _d.muAll, bestT: _d.bestT,
    vimU: _d.vimU, vimAlpha: _d.vimAlpha, fcVimAlpha: _d.getFcVimAlpha(),
    odinT: _d.odinT, odinEps: _d.odinEps,
    fc: _d.getFC(),
    thr: _d.getCachedThr(),
    trainMsp: _d.trainMsp,
    sortedMLSscores: _d.sortedMLSscores,
    sortedEBOscores: _d.sortedEBOscores,
    sortedTSMSP: _d.sortedTSMSP,
    sortedOdinPerClass: _d.sortedOdinPerClass,
    sortedVimScores: _d.sortedVimScores,
    sortedRMDS: _d.sortedRMDS,
    sortedFC: _d.getSortedFC(),
  };
}

// ─── Cache key ──────────────────────────────────────────────────

function _shaderMethodKey(method) {
  var m = getMethod(method);
  if (m && m.isFC) return method + "-" + _d.getFCBalanced();
  return method;
}

// ═══════════════════════════════════════════════════════════════════
//  Procedural shader pipeline
// ═══════════════════════════════════════════════════════════════════

export function buildProceduralMesh(method) {
  var m = getMethod(method);
  if (!m || !m.buildUniforms) return null;

  var ctx = _shaderCtx();
  var uniforms = m.buildUniforms(ctx);
  var fragShader = _shaderFragMap[m.shaderKey];
  if (!fragShader) return null;

  var material = new THREE.ShaderMaterial({
    uniforms: uniforms,
    vertexShader: vertShader,
    fragmentShader: fragShader,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: false,
  });
  var mesh = new THREE.Mesh(_sharedOverlayGeo, material);
  var key = _shaderMethodKey(method);
  _proceduralCache.set(key, { mesh: mesh, material: material });
  return { mesh: mesh, material: material };
}

export function updateProceduralUniforms(method, thr) {
  var key = _shaderMethodKey(method);
  var entry = _proceduralCache.get(key);
  if (!entry) return false;
  var mat = entry.material;
  var m = getMethod(method);
  if (!m) return false;

  if (m.thresholdShape === "perClass" || m.thresholdShape === "bands") {
    if (m.getPerClassThresholds) {
      mat.uniforms.perClassThr.value = m.getPerClassThresholds(thr);
    }
  } else if (m.getThreshold) {
    mat.uniforms.threshold.value = m.getThreshold(thr);
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════
//  Score-field pipeline (KNN / KDE)
// ═══════════════════════════════════════════════════════════════════

export function computeScoreFieldAsync(method, w, h, onDone) {
  var totalPx = w * h;
  var classIdx = new Uint8Array(totalPx);
  var scores = new Float32Array(totalPx);
  var ctx = _d.getScoringCtx();
  var m = getMethod(method);

  var py = 0;
  var ROWS = 8;

  function chunk() {
    var end = Math.min(py + ROWS, h);
    for (; py < end; py++) {
      for (var px = 0; px < w; px++) {
        var u = px / w;
        var v = py / h;
        var dir = dirFromUV(u, v);
        var pidx = py * w + px;
        var result = m && m.scoreFn ? m.scoreFn(dir, ctx) : { classIdx: 0, score: 0 };
        classIdx[pidx] = result.classIdx;
        scores[pidx] = result.score;
      }
    }
    if (py < h) requestAnimationFrame(chunk);
    else onDone({ type: "A", classIdx: classIdx, scores: scores });
  }
  requestAnimationFrame(chunk);
}

function _createTypeATextures(data, w, h) {
  var classIdxTex = new THREE.DataTexture(data.classIdx, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  classIdxTex.minFilter = THREE.NearestFilter;
  classIdxTex.magFilter = THREE.NearestFilter;
  classIdxTex.flipY = true;
  classIdxTex.needsUpdate = true;

  var scoreTex = new THREE.DataTexture(data.scores, w, h, THREE.RedFormat, THREE.FloatType);
  scoreTex.minFilter = THREE.NearestFilter;
  scoreTex.magFilter = THREE.NearestFilter;
  scoreTex.flipY = true;
  scoreTex.needsUpdate = true;

  return { classIdxTex: classIdxTex, scoreTex: scoreTex };
}

function _getMaxTrainScore(method) {
  if (method === "kde") return Math.max.apply(null, _d.trainKDEscores);
  if (method === "knn") return Math.min.apply(null, _d.trainKNNdists);
  return 1.0;
}

function _buildShaderMesh(method, scoreData, w, h) {
  var m = getMethod(method);
  var textures = _createTypeATextures(scoreData, w, h);
  var thr = _d.getCachedThr();
  var thrVal = m && m.getThreshold ? m.getThreshold(thr) : 0;
  var inverted = m ? m.invertedThreshold : false;
  var isGrad = m ? m.isGradient : false;

  var material = new THREE.ShaderMaterial({
    uniforms: {
      scoreMap: { value: textures.scoreTex },
      classIdxMap: { value: textures.classIdxTex },
      threshold: { value: thrVal },
      classColors: { value: _classColorsVec3 },
      isGradient: { value: isGrad ? 1.0 : 0.0 },
      maxTrainScore: { value: _getMaxTrainScore(method) },
      invertThreshold: { value: inverted ? 1.0 : 0.0 },
    },
    vertexShader: vertShader,
    fragmentShader: scoreFieldAFrag,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: false,
  });
  var mesh = new THREE.Mesh(_sharedOverlayGeo, material);
  return { mesh: mesh, material: material };
}

function _updateShaderUniforms(method, thr) {
  var key = _shaderMethodKey(method);
  var entry = _shaderCache.get(key);
  if (!entry) return false;
  var m = getMethod(method);
  if (m && m.getThreshold) {
    entry.material.uniforms.threshold.value = m.getThreshold(thr);
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════
//  Conformal pipeline
// ═══════════════════════════════════════════════════════════════════

function _cpShaderKey(mode, scoreType, balanced) {
  if (scoreType === "emb") return "cp-emb";
  if (mode === "cp-proto") return "cp-proto-" + scoreType;
  return "cp-fc-" + scoreType + "-" + balanced;
}

function _cpEnsureCalibScores(mode, scoreType, balanced) {
  var key = _cpShaderKey(mode, scoreType, balanced);
  if (_cpCalibScores.has(key)) return _cpCalibScores.get(key);
  var logitsFn = (mode === "cp-fc")
    ? (balanced ? _d.getFCBal() : _d.getFCUnbal()).logitsFn
    : function(x) { return _d.W.map(function(w) { return _d.TAU * v3dot(x, w); }); };
  var scores = [];
  for (var ci = 0; ci < _d.classes.length; ci++) {
    for (var pi = 0; pi < _d.classes[ci].points.length; pi++) {
      var p = _d.classes[ci].points[pi];
      if (scoreType === "emb") {
        scores.push(1 - v3dot(p, _d.classes[ci].muHat));
      } else {
        var probs = softmax(logitsFn(p));
        if (scoreType === "aps") {
          var sorted = probs.map(function(pr, k) { return { pr: pr, k: k }; }).sort(function(a, b) { return b.pr - a.pr; });
          var cum = 0;
          for (var si = 0; si < sorted.length; si++) { cum += sorted[si].pr; if (sorted[si].k === ci) break; }
          scores.push(cum);
        } else {
          scores.push(1 - probs[ci]);
        }
      }
    }
  }
  scores.sort(function(a, b) { return a - b; });
  _cpCalibScores.set(key, scores);
  return scores;
}

export function cpQhat(mode, scoreType, balanced, alpha) {
  var scores = _cpEnsureCalibScores(mode, scoreType, balanced);
  var n = scores.length;
  var level = Math.min(1, Math.ceil((n + 1) * (1 - alpha)) / n);
  return quantileSorted(scores, level);
}

export function buildConformalProceduralMesh(mode, scoreType, balanced) {
  var isFC = (mode === "cp-fc");
  var fragShader, uniforms;

  if (scoreType === "aps") {
    fragShader = conformalApsFrag;
    var wArr, bArr;
    if (isFC) {
      var variant = balanced ? _d.getFCBal() : _d.getFCUnbal();
      wArr = variant.W.map(function(w) { return new THREE.Vector3(w[0], w[1], w[2]); });
      bArr = variant.B.slice();
    } else {
      wArr = _d.W.map(function(w) { return new THREE.Vector3(w[0]*_d.TAU, w[1]*_d.TAU, w[2]*_d.TAU); });
      bArr = [0, 0, 0, 0, 0];
    }
    uniforms = {
      classW: { value: wArr },
      classB: { value: bArr },
      threshold: { value: 0.5 },
      classColors: { value: _classColorsVec3 },
    };
  } else {
    fragShader = conformalMspEmbFrag;
    var wArr2, bArr2;
    if (isFC) {
      var variant2 = balanced ? _d.getFCBal() : _d.getFCUnbal();
      wArr2 = variant2.W.map(function(w) { return new THREE.Vector3(w[0], w[1], w[2]); });
      bArr2 = variant2.B.slice();
    } else {
      wArr2 = _d.W.map(function(w) { return new THREE.Vector3(w[0]*_d.TAU, w[1]*_d.TAU, w[2]*_d.TAU); });
      bArr2 = [0, 0, 0, 0, 0];
    }
    uniforms = {
      classW: { value: wArr2 },
      classB: { value: bArr2 },
      classMuHat: { value: _d.classes.map(function(c) { return new THREE.Vector3(c.muHat[0], c.muHat[1], c.muHat[2]); }) },
      threshold: { value: 0.5 },
      classColors: { value: _classColorsVec3 },
      scoreMode: { value: scoreType === "emb" ? 1 : 0 },
    };
  }

  var material = new THREE.ShaderMaterial({
    uniforms: uniforms,
    vertexShader: vertShader,
    fragmentShader: fragShader,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: false,
  });
  var mesh = new THREE.Mesh(_sharedOverlayGeo, material);
  var key = _cpShaderKey(mode, scoreType, balanced);
  _cpProceduralCache.set(key, { mesh: mesh, material: material });
  return { mesh: mesh, material: material };
}

export function updateConformalProceduralUniforms(mode) {
  var cpScoreType = _d.getCpScoreType();
  var fcBalanced = _d.getFCBalanced();
  var cpAlpha = _d.getCpAlpha();
  var key = _cpShaderKey(mode, cpScoreType, fcBalanced);
  var entry = _cpProceduralCache.get(key);
  if (!entry) return false;
  var qhat = cpQhat(mode, cpScoreType, fcBalanced, cpAlpha);
  entry.material.uniforms.threshold.value = 1 - qhat;
  return true;
}

export function drawConformal(mode) {
  var g = _d.layers[mode];
  var cpScoreType = _d.getCpScoreType();
  var fcBalanced = _d.getFCBalanced();
  var key = _cpShaderKey(mode, cpScoreType, fcBalanced);

  if (!_cpProceduralCache.has(key)) {
    buildConformalProceduralMesh(mode, cpScoreType, fcBalanced);
  }
  var entry = _cpProceduralCache.get(key);
  if (g.children.indexOf(entry.mesh) < 0) {
    while (g.children.length) g.remove(g.children[0]);
    g.add(entry.mesh);
  }
  updateConformalProceduralUniforms(mode);
  if (_d.onReadyChange) _d.onReadyChange();
}

// ═══════════════════════════════════════════════════════════════════
//  showMethodAtQ — main dispatcher
// ═══════════════════════════════════════════════════════════════════

export function showMethodAtQ(method, q) {
  if (["cp-proto", "cp-fc", "mds", "mds-s", "none"].includes(method)) return;
  var g = _d.layers[method];
  var m = getMethod(method);
  if (!m) return;

  // PROCEDURAL PATH
  if (m.renderKind === "procedural") {
    var pKey = _shaderMethodKey(method);
    if (!_proceduralCache.has(pKey)) {
      buildProceduralMesh(method);
    }
    var entry = _proceduralCache.get(pKey);
    if (g.children.indexOf(entry.mesh) < 0) {
      while (g.children.length) g.remove(g.children[0]);
      g.add(entry.mesh);
    }
    var currentQ = _d.getCurrentQ();
    var thr = (Math.abs(q - currentQ) < 0.001) ? _d.getCachedThr() : _d.getAllThresholds(q);
    updateProceduralUniforms(method, thr);
    if (_d.onReadyChange) _d.onReadyChange();
    return;
  }

  // SCORE-FIELD PATH (KNN / KDE)
  var shaderKey = _shaderMethodKey(method);
  if (_shaderCache.has(shaderKey)) {
    var sfEntry = _shaderCache.get(shaderKey);
    if (g.children.indexOf(sfEntry.mesh) < 0) {
      while (g.children.length) g.remove(g.children[0]);
      g.add(sfEntry.mesh);
    }
    var currentQ2 = _d.getCurrentQ();
    var thr2 = (Math.abs(q - currentQ2) < 0.001) ? _d.getCachedThr() : _d.getAllThresholds(q);
    _updateShaderUniforms(method, thr2);
    return;
  }

  // Not yet computed — signal loading
  if (_d.onLoading) _d.onLoading();
  if (_shaderComputing.has(shaderKey)) return;

  _shaderComputing.add(shaderKey);
  var tw = 1024, th = 512;

  computeScoreFieldAsync(method, tw, th, function(scoreData) {
    _shaderComputing.delete(shaderKey);
    var built = _buildShaderMesh(method, scoreData, tw, th);
    _shaderCache.set(shaderKey, {
      mesh: built.mesh,
      material: built.material,
      w: tw, h: th,
    });
    _updateShaderUniforms(method, _d.getCachedThr());
    if (_d.getActiveMode() === method) {
      while (g.children.length) g.remove(g.children[0]);
      g.add(built.mesh);
    }
    if (_d.onReadyChange) _d.onReadyChange();
  });
}

// ═══════════════════════════════════════════════════════════════════
//  Background precomputation
// ═══════════════════════════════════════════════════════════════════

var _bgPrecomputeMethods = ["knn", "kde"];

function _bgPickNext() {
  for (var i = 0; i < _bgPrecomputeMethods.length; i++) {
    var m = _bgPrecomputeMethods[i];
    var mKey = _shaderMethodKey(m);
    if (!_shaderCache.has(mKey) && !_shaderComputing.has(mKey)) return m;
  }
  return null;
}

function _bgNext() {
  var method = _bgPickNext();
  if (!method) {
    _bgRunning = false;
    if (_d.onReadyChange) _d.onReadyChange();
    return;
  }
  _bgRunning = true;
  var shaderKey = _shaderMethodKey(method);
  if (_shaderCache.has(shaderKey) || _shaderComputing.has(shaderKey)) {
    setTimeout(_bgNext, 10);
    return;
  }
  _shaderComputing.add(shaderKey);
  var tw = 1024, th = 512;
  computeScoreFieldAsync(method, tw, th, function(scoreData) {
    _shaderComputing.delete(shaderKey);
    var built = _buildShaderMesh(method, scoreData, tw, th);
    _shaderCache.set(shaderKey, {
      mesh: built.mesh,
      material: built.material,
      w: tw, h: th,
    });
    _updateShaderUniforms(method, _d.getCachedThr());
    if (_d.getActiveMode() === method) {
      var g = _d.layers[method];
      while (g.children.length) g.remove(g.children[0]);
      g.add(built.mesh);
    }
    if (_d.onReadyChange) _d.onReadyChange();
    _bgNext();
  });
}

export function bgRestart() {
  if (!_bgRunning) setTimeout(_bgNext, 50);
}

export function bgStart() {
  setTimeout(_bgNext, 500);
}

// ═══════════════════════════════════════════════════════════════════
//  Cache management (called by FC toggle, etc.)
// ═══════════════════════════════════════════════════════════════════

export function invalidateProceduralCache(method) {
  var key = _shaderMethodKey(method);
  if (_proceduralCache.has(key)) {
    var entry = _proceduralCache.get(key);
    if (entry.material) entry.material.dispose();
    _proceduralCache.delete(key);
  }
}

export function invalidateAllFCProceduralCaches() {
  for (var [key, entry] of _proceduralCache) {
    if (key.indexOf("fc-") === 0) {
      if (entry.material) entry.material.dispose();
      _proceduralCache.delete(key);
    }
  }
}

export function invalidateConformalCache() {
  for (var [, entry] of _cpProceduralCache) {
    if (entry.material) entry.material.dispose();
  }
  _cpProceduralCache.clear();
}

export function clearCalibScores() {
  _cpCalibScores.clear();
}

export function clearShaderCache() {
  for (var [, entry] of _shaderCache) {
    if (entry.material) entry.material.dispose();
  }
  _shaderCache.clear();
}

export function isScoreFieldReady(method) {
  var m = getMethod(method);
  if (!m) return false;
  if (m.renderKind === "procedural" || m.renderKind === "conformal" ||
      m.renderKind === "ellipsoid" || m.renderKind === "none") return true;
  return _shaderCache.has(_shaderMethodKey(method));
}
