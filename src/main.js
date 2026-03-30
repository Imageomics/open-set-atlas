// ═══════════════════════════════════════════════════════════════════
//  Composition root: initializes data, creates scene, wires state,
//  binds effects, starts animation. All JS lives in src/.
// ═══════════════════════════════════════════════════════════════════

import { v3add, v3sub, v3scale, v3dot, v3cross, v3len, v3norm } from './math/vec3.js';
import { rotateToDir, computeMeanDir, kappaFromRbar3D, hexToRGB, dirFromUV } from './math/spherical.js';
import { resetSeed, rand, sampleVMF, sampleAniso, fibGrid } from './math/sampling.js';
import { covMatrix3, pooledCov, inv3, mahalDist2, eig3sym } from './math/linalg.js';
import { quantile, quantileSorted, softmax, logsumexp } from './math/stats.js';
import { fitKent, makeKentScoreFn } from './math/fitting.js';
import { TAU, Q, ODIN_T as odinT, ODIN_EPS as odinEps } from './config/constants.js';
import { classDefs } from './config/class-defs.js';
import { getMethod } from './config/method-registry.js';
import { odinScore, vimResidual, vimScoreFn, rmdsScore, knnResult, kdeScore, makeFCLogitsFn } from './methods/scoring.js';
import { initShaderManager, showMethodAtQ as smShowMethodAtQ, drawConformal as smDrawConformal, bgRestart, bgStart, invalidateAllFCProceduralCaches, invalidateConformalCache, clearCalibScores, isScoreFieldReady, updateConformalProceduralUniforms, cpQhat } from './rendering/shader-manager.js';
import { initOrbitControls, lookAtPoint } from './rendering/orbit-controls.js';
import { addEllipsoids, buildPerClassEllipsoids } from './rendering/ellipsoids.js';
import { initControls, setActive as smSetActive, qUpdateTrackBar, qShowSpinner, qHideSpinner } from './ui/controls.js';
import { initDecisionTable, updateDecisionTable as dtUpdateDecisionTable } from './ui/decision-table.js';
import { trainFC3D as smTrainFC3D, reconstructFCVariant as smReconstructFCVariant, computeFcOdinData as smComputeFcOdinData, computeFcVimData as smComputeFcVimData } from './methods/fc-trainer.js';
import { conformalCalibrateScore, conformalPredictScore } from './methods/conformal.js';
import * as AppState from './state/app-state.js';
import { initSortedScores, rebuildSortedFC, allThresholds as engineAllThresholds, getCachedThresholds, setCachedThresholds } from './state/threshold-engine.js';
import { bindFCEffects } from './state/fc-effects.js';
const PRE = window.OOD_PRECOMPUTED || null;

// ── helpers for reconstructing functions from precomputed data ──
// makeFCLogitsFn — imported from src/methods/scoring.js

function loadTexturePNG(filePath, layerGroup, onLoad, radius) {
  radius = radius || 1.005;
  const img = new Image();
  img.onload = function() {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    const geo = new THREE.SphereGeometry(radius, 128, 64);
    const mat = new THREE.MeshBasicMaterial({
      map: tex, transparent: true, side: THREE.FrontSide, depthWrite: false
    });
    layerGroup.add(new THREE.Mesh(geo, mat));
    if (onLoad) onLoad();
  };
  img.onerror = function() {
    console.warn("Failed to load texture:", filePath);
    if (onLoad) onLoad();
  };
  img.src = filePath;
}

// ── Math functions imported from src/math/ ──





// ══════════════════════════════════════════════
//  DATA GENERATION
// ══════════════════════════════════════════════

let currentQ = Q;

let classes, poolCov, poolInvCov, W, trainMsp, gamma, grid4k, mspScores4k, oodPoints;

if (PRE) {
  // ── Reconstruct from precomputed data ──
  classes = PRE.classes.map((pc, ci) => {
    const kentObj = {
      kappa: pc.kent.kappa, beta: pc.kent.beta,
      gamma1: pc.kent.gamma1, gamma2: pc.kent.gamma2, gamma3: pc.kent.gamma3,
      bands: pc.kent.bands,
      kentScore: makeKentScoreFn(pc.kent)
    };
    return {
      label: pc.label, color: pc.color, dir: pc.dir,
      n: pc.n, aniso: pc.aniso, stretch: pc.stretch,
      points: pc.points, muBar: pc.muBar, muHat: pc.muHat, Rbar: pc.Rbar,
      kappa: pc.kappa,
      cov: pc.cov || covMatrix3(pc.points, pc.muBar),
      invCov: pc.invCov || inv3(pc.cov || covMatrix3(pc.points, pc.muBar)),
      cosScores: pc.cosScores, cosThreshold: pc.cosThreshold,
      vmScores: pc.vmScores, vmBands: pc.vmBands,
      kent: kentObj, kentScores: pc.kentScores
    };
  });
  poolCov = PRE.poolCov;
  poolInvCov = PRE.poolInvCov;
  W = PRE.W;
  trainMsp = PRE.trainMsp;
  gamma = PRE.gamma;
  grid4k = fibGrid(4000);
  mspScores4k = PRE.mspScores4k;
  oodPoints = PRE.oodPoints;
} else {
  // ── Compute from scratch (fallback) ──
  classes = classDefs.map(def => {
    const points = def.aniso
      ? sampleAniso(def.dir, def.kappa, def.stretch, def.n)
      : sampleVMF(def.dir, def.kappa, def.n);
    const stats = computeMeanDir(points);
    const kappa = kappaFromRbar3D(stats.Rbar);
    const cov = covMatrix3(points, stats.muBar);
    const invCov = inv3(cov);
    const cosScores = points.map(p => v3dot(p, stats.muHat));
    const cosThreshold = quantile(cosScores, Q);
    const vmScores = points.map(p => kappa * v3dot(p, stats.muHat));
    const vmBands = [0.05,0.20,0.50].map(q => quantile(vmScores, q));
    const kent = fitKent(points, stats.muHat);
    const kentScores = points.map(p => kent.kentScore(p));
    return { ...def, points, ...stats, kappa, cov, invCov, cosScores, cosThreshold, vmScores, vmBands, kent, kentScores };
  });

  poolCov = pooledCov(classes);
  poolInvCov = inv3(poolCov);

  // MSP setup
  W = classes.map(c => c.muHat);
  trainMsp = classes.map((c, i) =>
    c.points.map(p => {
      const logits = W.map(w => TAU * v3dot(p, w));
      return softmax(logits)[i];
    })
  );
  const perClassGamma = trainMsp.map(xs => quantile(xs, Q));
  gamma = perClassGamma.reduce((a,b) => a+b, 0) / perClassGamma.length;

  // OOD points
  grid4k = fibGrid(4000);
  mspScores4k = grid4k.map(p => {
    const logits = W.map(w => TAU * v3dot(p, w));
    const probs = softmax(logits);
    return Math.max(...probs);
  });
  // pick 4 well-separated low-MSP points
  const sorted4k = mspScores4k.map((s,i) => [s,i]).sort((a,b) => a[0]-b[0]);
  const oodIdx = [];
  for (const [,idx] of sorted4k) {
    const p = grid4k[idx];
    const ok = oodIdx.every(j => {
      const d = 1 - v3dot(grid4k[j], p);
      return d > 0.3;
    });
    if (ok) oodIdx.push(idx);
    if (oodIdx.length >= 4) break;
  }
  oodPoints = oodIdx.map(i => grid4k[i]);
}

// (grid-based evaluation moved to texture rendering below)

// ══════════════════════════════════════════════
//  THREE.JS SCENE
// ══════════════════════════════════════════════

const container = document.getElementById("canvas-wrap");
const W3 = container.clientWidth, H3 = 500;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, W3/H3, 0.1, 100);
camera.position.set(2.6, 1.6, 2.6);
camera.lookAt(0,0,0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(W3, H3);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dLight = new THREE.DirectionalLight(0xffffff, 0.4);
dLight.position.set(3,5,4);
scene.add(dLight);

// unit sphere wireframe
const wireGeo = new THREE.SphereGeometry(1.0, 48, 24);
scene.add(new THREE.Mesh(wireGeo, new THREE.MeshBasicMaterial({
  color: 0x999999, wireframe: true, transparent: true, opacity: 0.06
})));

// translucent sphere shell
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(0.998, 48, 24),
  new THREE.MeshPhongMaterial({ color: 0xcccccc, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false })
));

// origin dot
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(0.012, 8, 8),
  new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.4 })
));

// ── data points ──
function makePointCloud(pts, color, size) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(pts.length * 3);
  pts.forEach((p, i) => { pos[i*3]=p[0]; pos[i*3+1]=p[1]; pos[i*3+2]=p[2]; });
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(geo, new THREE.PointsMaterial({ color: new THREE.Color(color), size, sizeAttenuation: true }));
}

for (const c of classes) scene.add(makePointCloud(c.points, c.color, 0.025));
const farOodGroup = new THREE.Group();
scene.add(farOodGroup);
farOodGroup.add(makePointCloud(oodPoints, "#E24B4A", 0.055));

// Raycast sphere for interactive probing (Feature 2)
const raycastSphere = new THREE.Mesh(
  new THREE.SphereGeometry(1.0, 48, 24),
  new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false })
);
scene.add(raycastSphere);

// Probe marker — wireframe diamond + solid core, bright cyan
const probeMarker = new THREE.Group();
probeMarker.add(new THREE.Mesh(
  new THREE.OctahedronGeometry(0.035),
  new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true, transparent: true, opacity: 0.9 })
));
probeMarker.add(new THREE.Mesh(
  new THREE.OctahedronGeometry(0.015),
  new THREE.MeshBasicMaterial({ color: 0x00e5ff })
));
probeMarker.visible = false;
scene.add(probeMarker);

// prototypes (in a group so we can toggle visibility for FC modes)
const protoMarkers = new THREE.Group();
scene.add(protoMarkers);
for (const c of classes) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 12, 12),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(c.color) })
  );
  m.position.set(...c.muHat);
  protoMarkers.add(m);
}

// ── method layers ──
const layers = {};
const layerKeys = ["mds","mds-s","msp","mls","ebo","ts","odin","vim","cos","vmf","kent","knn","rmds","kde","fc-msp","fc-mls","fc-ebo","fc-ts","fc-odin","fc-vim","fc-cos","fc-vmf","fc-kent","cp-proto","cp-fc"];
layerKeys.forEach(k => { layers[k] = new THREE.Group(); layers[k].visible = false; scene.add(layers[k]); });

// ── Mahalanobis ellipsoids — moved to src/rendering/ellipsoids.js ──
buildPerClassEllipsoids(THREE, layers.mds, classes);
addEllipsoids(THREE, layers["mds-s"], poolCov, classes);
// ── shell-based methods: texture-mapped sphere ──
// hexToRGB imported from src/math/spherical.js

const TEX_W = 2048, TEX_H = 1024;

// dirFromUV imported from src/math/spherical.js

function makeMethodSphere(evalFn, layerGroup) {
  const canvas = document.createElement("canvas");
  canvas.width = TEX_W;
  canvas.height = TEX_H;
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(TEX_W, TEX_H);
  const pixels = imgData.data;

  for (let py2 = 0; py2 < TEX_H; py2++) {
    for (let px2 = 0; px2 < TEX_W; px2++) {
      const u = px2 / TEX_W;
      const v = py2 / TEX_H;
      const dir = dirFromUV(u, v);
      const result = evalFn(dir);
      const idx = (py2 * TEX_W + px2) * 4;
      if (result) {
        const [r, g, b] = hexToRGB(result.color);
        const a = Math.round((result.opacity || 0.5) * 255);
        pixels[idx] = r;
        pixels[idx+1] = g;
        pixels[idx+2] = b;
        pixels[idx+3] = a;
      } else {
        pixels[idx] = 0;
        pixels[idx+1] = 0;
        pixels[idx+2] = 0;
        pixels[idx+3] = 0;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;

  const geo = new THREE.SphereGeometry(1.005, 128, 64);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: false
  });
  layerGroup.add(new THREE.Mesh(geo, mat));
}

// logsumexp imported from src/math/stats.js

// ── KNN / RMDS helpers (always needed for runtime evaluation) ──
const allTrainPts = PRE ? PRE.allTrainPts : [];
const allTrainLabels = PRE ? PRE.allTrainLabels : [];
const kVal = PRE ? PRE.kVal : 5;

if (!PRE) {
  for (let ci = 0; ci < classes.length; ci++) {
    for (const p of classes[ci].points) { allTrainPts.push(p); allTrainLabels.push(ci); }
  }
}

// knnResult — imported from src/methods/scoring.js (parameterized by allTrainPts, allTrainLabels, kVal)

const muAll = PRE ? PRE.muAll : (() => {
  let m = [0,0,0]; let n = 0;
  for (const c of classes) { for (const p of c.points) { m = v3add(m, p); n++; } }
  return v3scale(m, 1/n);
})();

// odinScore — imported from src/methods/scoring.js (same signature)

// Prototype ODIN helpers
// protoGetLogits is an alias used before protoLogitsFn is defined (they are identical)
function protoGetLogits(x) {
  return W.map(w => TAU * v3dot(x, w));
}
const protoOdinWeights = W.map(w => v3scale(w, TAU));
// Note: protoLogitsFn (defined later) is the same function, used in conformal code.

// ── ViM components (Wang et al. 2022) ──
// Compute covariance of all training points centered on muAll
const vimD = 2; // keep 2 principal components in 3D (1D null space)
var vimU, vimAlpha;
if (PRE && PRE.vimU) {
  vimU = PRE.vimU;
  vimAlpha = PRE.vimAlpha;
} else {
  const vimCov = (function() {
    const S = [[0,0,0],[0,0,0],[0,0,0]];
    for (const p of allTrainPts) {
      const d = v3sub(p, muAll);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          S[i][j] += d[i]*d[j];
    }
    const n = allTrainPts.length;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        S[i][j] /= Math.max(1, n);
    return S;
  })();
  const vimEig = eig3sym(vimCov);
  vimU = [vimEig.vectors[0], vimEig.vectors[1]]; // top 2 eigenvectors
  // Calibrate alpha: mean max real logit / mean ||residual|| on training data
  const vimTrainMaxLogits = allTrainPts.map(p => Math.max(...W.map(w => TAU * v3dot(p, w))));
  const vimTrainResNorms = allTrainPts.map(p => v3len(vimResidual(p, vimU, muAll)));
  const vimMeanMaxLogit = vimTrainMaxLogits.reduce((a,b) => a+b, 0) / vimTrainMaxLogits.length;
  var vimMeanResNorm = vimTrainResNorms.reduce((a,b) => a+b, 0) / vimTrainResNorms.length;
  vimAlpha = vimMeanResNorm > 1e-10 ? vimMeanMaxLogit / vimMeanResNorm : 0;
}
// vimResidual, vimScoreFn — imported from src/methods/scoring.js (parameterized)
// vimMeanResNorm needed for FC ViM -- compute if not already set from non-PRE path
if (typeof vimMeanResNorm === 'undefined') {
  var vimMeanResNorm = allTrainPts.map(p => v3len(vimResidual(p, vimU, muAll))).reduce((a,b) => a+b, 0) / allTrainPts.length;
}

// rmdsScore, kdeScore — imported from src/methods/scoring.js (parameterized)
const kdeBandwidth = quantile(classes.map(c => c.kappa), 0.5);

// ── Threshold variables (set from PRE or computed) ──
let trainMLSscores, mlsThreshold, trainEBOscores, eboThreshold;
let bestT, trainTSMSP, tsGamma;
let trainKNNdists, knnThreshold, trainRMDS, rmdsThreshold;
let trainKDEscores;

if (PRE) {
  trainMLSscores = PRE.trainMLSscores;
  mlsThreshold = PRE.mlsThreshold;
  trainEBOscores = PRE.trainEBOscores;
  eboThreshold = PRE.eboThreshold;
  bestT = PRE.bestT;
  trainTSMSP = PRE.trainTSMSP;
  tsGamma = PRE.tsGamma;
  trainKNNdists = PRE.trainKNNdists;
  knnThreshold = PRE.knnThreshold;
  trainRMDS = PRE.trainRMDS;
  rmdsThreshold = PRE.rmdsThreshold;
} else {
  // ── Compute thresholds from scratch ──
  trainMLSscores = [];
  for (const c of classes) {
    for (const p of c.points) {
      trainMLSscores.push(Math.max(...W.map(w => TAU * v3dot(p, w))));
    }
  }
  mlsThreshold = quantile(trainMLSscores, Q);

  trainEBOscores = [];
  for (const c of classes) {
    for (const p of c.points) {
      trainEBOscores.push(logsumexp(W.map(w => TAU * v3dot(p, w))));
    }
  }
  eboThreshold = quantile(trainEBOscores, Q);

  bestT = 1.0;
  let bestNLL = Infinity;
  for (let t = 0.2; t <= 10.0; t += 0.1) {
    let nll = 0;
    for (let ci = 0; ci < classes.length; ci++) {
      for (const p of classes[ci].points) {
        const logits = W.map(w => (TAU / t) * v3dot(p, w));
        const probs = softmax(logits);
        nll -= Math.log(Math.max(probs[ci], 1e-15));
      }
    }
    if (nll < bestNLL) { bestNLL = nll; bestT = t; }
  }
  trainTSMSP = classes.map((c, i) =>
    c.points.map(p => softmax(W.map(w => (TAU / bestT) * v3dot(p, w)))[i])
  );
  tsGamma = trainTSMSP.map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / classes.length;

  trainKNNdists = allTrainPts.map((p, idx) => knnResult(p, allTrainPts, allTrainLabels, kVal, idx).kthDist);
  knnThreshold = quantile(trainKNNdists, 1 - Q);

  trainRMDS = allTrainPts.map(p => rmdsScore(p, classes, poolInvCov, muAll).score);
  rmdsThreshold = quantile(trainRMDS, 1 - Q);
}

// KDE training scores (always recompute — kdeScore now returns log-density)
trainKDEscores = allTrainPts.map(p => kdeScore(p, classes, kdeBandwidth).density);

// ── ODIN threshold ──
var odinGamma, fcOdinGamma;
// Stored training score arrays for allThresholds recomputation
var _storedOdinPerClass, _storedTrainVimScores;
if (PRE && PRE.trainOdinPerClass) {
  _storedOdinPerClass = PRE.trainOdinPerClass;
  odinGamma = PRE.odinGamma;
} else {
  _storedOdinPerClass = classes.map((c, ci) =>
    c.points.map(p => {
      const r = odinScore(p, protoGetLogits, protoOdinWeights, odinT, odinEps);
      return r.argmax === ci ? r.score : 0;
    })
  );
  odinGamma = _storedOdinPerClass.map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / classes.length;
}

// ── ViM threshold ──
var vimThreshold, fcVimThreshold;
if (PRE && PRE.trainVimScores) {
  _storedTrainVimScores = PRE.trainVimScores;
  vimThreshold = PRE.vimThreshold;
} else {
  _storedTrainVimScores = allTrainPts.map(p => vimScoreFn(p, protoGetLogits, vimU, muAll, vimAlpha).energy);
  vimThreshold = quantile(_storedTrainVimScores, Q);
}

// ── Prototype textures ──
if (!PRE) {
  // MSP texture
  makeMethodSphere((p) => {
    const logits = W.map(w => TAU * v3dot(p, w));
    const probs = softmax(logits);
    let argmax = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[argmax]) argmax = i;
    if (probs[argmax] >= gamma) return { color: classes[argmax].color, opacity: 0.55 };
    return null;
  }, layers.msp);

  // Per-class cosine texture (ratio-based overlap resolution)
  makeMethodSphere((p) => {
    let best = null, bestRatio = -Infinity;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = v3dot(p, classes[ci].muHat);
      const ratio = classes[ci].cosThreshold !== 0 ? s / classes[ci].cosThreshold : s;
      if (s >= classes[ci].cosThreshold && ratio > bestRatio) { best = classes[ci]; bestRatio = ratio; }
    }
    return best ? { color: best.color, opacity: 0.55 } : null;
  }, layers.cos);

  // vMF texture
  (function() {
    const canvas = document.createElement("canvas");
    canvas.width = TEX_W; canvas.height = TEX_H;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(TEX_W, TEX_H);
    const pixels = imgData.data;
    for (let py2 = 0; py2 < TEX_H; py2++) {
      for (let px2 = 0; px2 < TEX_W; px2++) {
        const dir = dirFromUV(px2 / TEX_W, py2 / TEX_H);
        const idx = (py2 * TEX_W + px2) * 4;
        let bestCI = -1, bestRatio = -Infinity, bestS = 0;
        for (let ci = 0; ci < classes.length; ci++) {
          const s = classes[ci].kappa * v3dot(dir, classes[ci].muHat);
          const thrVal = classes[ci].vmBands[0];
          if (s >= thrVal) {
            const ratio = thrVal !== 0 ? s / thrVal : s;
            if (ratio > bestRatio) { bestCI = ci; bestRatio = ratio; bestS = s; }
          }
        }
        if (bestCI >= 0) {
          const maxS = Math.max(...classes[bestCI].vmScores);
          const thrVal = classes[bestCI].vmBands[0];
          const t = maxS > thrVal ? (bestS - thrVal) / (maxS - thrVal) : 1;
          const [r, g, b] = hexToRGB(classes[bestCI].color);
          pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b;
          pixels[idx+3] = Math.round((0.01 + 0.74 * Math.min(1, t)) * 255);
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    const geo = new THREE.SphereGeometry(1.005, 128, 64);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.FrontSide, depthWrite: false });
    layers.vmf.add(new THREE.Mesh(geo, mat));
  })();

  // Kent texture
  (function() {
    const canvas = document.createElement("canvas");
    canvas.width = TEX_W; canvas.height = TEX_H;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(TEX_W, TEX_H);
    const pixels = imgData.data;
    for (let py2 = 0; py2 < TEX_H; py2++) {
      for (let px2 = 0; px2 < TEX_W; px2++) {
        const dir = dirFromUV(px2 / TEX_W, py2 / TEX_H);
        const idx = (py2 * TEX_W + px2) * 4;
        let bestCI = -1, bestRatio = -Infinity, bestS = 0;
        for (let ci = 0; ci < classes.length; ci++) {
          const s = classes[ci].kent.kentScore(dir);
          const thrVal = classes[ci].kent.bands[0];
          if (s >= thrVal) {
            const ratio = thrVal !== 0 ? s / thrVal : s;
            if (ratio > bestRatio) { bestCI = ci; bestRatio = ratio; bestS = s; }
          }
        }
        if (bestCI >= 0) {
          const maxS = Math.max(...classes[bestCI].kentScores);
          const thrVal = classes[bestCI].kent.bands[0];
          const t = maxS > thrVal ? (bestS - thrVal) / (maxS - thrVal) : 1;
          const [r, g, b] = hexToRGB(classes[bestCI].color);
          pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b;
          pixels[idx+3] = Math.round((0.01 + 0.74 * Math.min(1, t)) * 255);
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    layers.kent.add(new THREE.Mesh(
      new THREE.SphereGeometry(1.005, 128, 64),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.FrontSide, depthWrite: false })
    ));
  })();

  // MLS texture
  makeMethodSphere((p) => {
    const logits = W.map(w => TAU * v3dot(p, w));
    const maxL = Math.max(...logits);
    if (maxL >= mlsThreshold) {
      let argmax = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
      return { color: classes[argmax].color, opacity: 0.55 };
    }
    return null;
  }, layers.mls);

  // EBO texture
  makeMethodSphere((p) => {
    const logits = W.map(w => TAU * v3dot(p, w));
    const energy = logsumexp(logits);
    if (energy >= eboThreshold) {
      let argmax = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
      return { color: classes[argmax].color, opacity: 0.55 };
    }
    return null;
  }, layers.ebo);

  // TempScale texture
  makeMethodSphere((p) => {
    const logits = W.map(w => (TAU / bestT) * v3dot(p, w));
    const probs = softmax(logits);
    let argmax = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[argmax]) argmax = i;
    if (probs[argmax] >= tsGamma) return { color: classes[argmax].color, opacity: 0.55 };
    return null;
  }, layers.ts);

  // KNN texture (reduced resolution)
  (function() {
    const KW = 1024, KH = 512;
    const canvas = document.createElement("canvas");
    canvas.width = KW; canvas.height = KH;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(KW, KH);
    const pixels = imgData.data;
    for (let py2 = 0; py2 < KH; py2++) {
      for (let px2 = 0; px2 < KW; px2++) {
        const dir = dirFromUV(px2/KW, py2/KH);
        const result = knnResult(dir, allTrainPts, allTrainLabels, kVal);
        const idx = (py2 * KW + px2) * 4;
        if (result.kthDist <= knnThreshold) {
          const [r, g, b] = hexToRGB(classes[result.nearestLabel].color);
          pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b;
          pixels[idx+3] = Math.round(0.55 * 255);
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    const geo = new THREE.SphereGeometry(1.005, 128, 64);
    layers.knn.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      map: tex, transparent: true, side: THREE.FrontSide, depthWrite: false
    })));
  })();

  // RMDS texture
  makeMethodSphere((p) => {
    const result = rmdsScore(p, classes, poolInvCov, muAll);
    if (result.score <= rmdsThreshold) {
      return { color: classes[result.bestClass].color, opacity: 0.55 };
    }
    return null;
  }, layers.rmds);
} // end if (!PRE) for prototype textures

// (Textures are computed lazily on-demand — no init-time PNG loading needed)

// ══════════════════════════════════════════════
//  LEARNED FC — trained both balanced and unbalanced, toggled at runtime
// ══════════════════════════════════════════════
const nClasses = classes.length;
const fcTrainX = [], fcTrainY = [];
for (let ci = 0; ci < nClasses; ci++) {
  for (const p of classes[ci].points) { fcTrainX.push(p); fcTrainY.push(ci); }
}
const fcN = fcTrainX.length;
const classWeights = classes.map(c => fcN / (nClasses * c.n));

function reconstructFCVariant(preFC) { return smReconstructFCVariant(preFC); }

function trainFC3D(balanced) { return smTrainFC3D(classes, TAU, Q, balanced); }

let fcBal, fcUnbal;
if (PRE) {
  fcBal = reconstructFCVariant(PRE.fcBal);
  fcUnbal = reconstructFCVariant(PRE.fcUnbal);
} else {
  fcBal = trainFC3D(true);
  fcUnbal = trainFC3D(false);
}
let fc = fcBal;
let fcBalanced = true;

function fcLogits(x) { return fc.logitsFn(x); }

// ── FC ODIN/ViM thresholds (computed after FC is available) ──
// Stored per-variant arrays for allThresholds recomputation
var _storedFcBalOdinPerClass, _storedFcUnbalOdinPerClass;
var _storedFcBalVimScores, _storedFcUnbalVimScores;

function computeFcOdinData(fcVariant) { return smComputeFcOdinData(fcVariant, classes, odinT, odinEps, Q); }
function computeFcVimData(fcVariant) { return smComputeFcVimData(fcVariant, allTrainPts, vimU, muAll, vimMeanResNorm, Q); }

if (PRE && PRE.fcBal && PRE.fcBal.odinPerClass) {
  _storedFcBalOdinPerClass = PRE.fcBal.odinPerClass;
  _storedFcUnbalOdinPerClass = PRE.fcUnbal.odinPerClass;
  fcOdinGamma = PRE.fcBal.odinGamma;
  _storedFcBalVimScores = PRE.fcBal.trainVimScores;
  _storedFcUnbalVimScores = PRE.fcUnbal.trainVimScores;
  fcVimThreshold = PRE.fcBal.vimThreshold;
  var fcVimAlphaVal = PRE.fcBal.vimAlpha;
} else {
  const balOdin = computeFcOdinData(fcBal);
  _storedFcBalOdinPerClass = balOdin.perClass;
  fcOdinGamma = balOdin.gamma;
  const unbalOdin = computeFcOdinData(fcUnbal);
  _storedFcUnbalOdinPerClass = unbalOdin.perClass;
  const balVim = computeFcVimData(fcBal);
  _storedFcBalVimScores = balVim.scores;
  fcVimThreshold = balVim.thr;
  var fcVimAlphaVal = balVim.alpha;
  const unbalVim = computeFcVimData(fcUnbal);
  _storedFcUnbalVimScores = unbalVim.scores;
}

// Helper: get current FC variant stored arrays
function _currentFcOdinPerClass() { return fcBalanced ? _storedFcBalOdinPerClass : _storedFcUnbalOdinPerClass; }
function _currentFcVimScores() { return fcBalanced ? _storedFcBalVimScores : _storedFcUnbalVimScores; }

// fcVimScoreFn replaced by vimScoreFn(x, fc.logitsFn, vimU, muAll, fcVimAlphaVal)

// Precompute texture spheres for both variants into sub-groups
function makeFCTextures(variant, parentGroups) {
  const fn = variant.logitsFn;
  const subGroups = {};
  function evalMSP(p) {
    const l = fn(p), pr = softmax(l); let a = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[a]) a = i;
    return pr[a] >= variant.gamma ? { color: classes[a].color, opacity: 0.55 } : null;
  }
  function evalMLS(p) {
    const l = fn(p), mx = Math.max(...l);
    if (mx >= variant.mlsThr) { let a = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[a]) a = i;
      return { color: classes[a].color, opacity: 0.55 }; } return null;
  }
  function evalEBO(p) {
    const l = fn(p);
    if (logsumexp(l) >= variant.eboThr) { let a = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[a]) a = i;
      return { color: classes[a].color, opacity: 0.55 }; } return null;
  }
  function evalTS(p) {
    const l = fn(p), pr = softmax(l.map(v => v / variant.bestT)); let a = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[a]) a = i;
    return pr[a] >= variant.tsGamma ? { color: classes[a].color, opacity: 0.55 } : null;
  }
  function evalCos(p) {
    let best = null, bestRatio = -Infinity;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = v3dot(p, variant.wDirs[ci]);
      const ratio = variant.fcCosThresholds[ci] !== 0 ? s / variant.fcCosThresholds[ci] : s;
      if (s >= variant.fcCosThresholds[ci] && ratio > bestRatio) { best = classes[ci]; bestRatio = ratio; }
    }
    return best ? { color: best.color, opacity: 0.55 } : null;
  }
  function evalVmf(p) {
    let bestCI = -1, bestRatio = -Infinity, bestS = 0;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = variant.fcKappas[ci] * v3dot(p, variant.wDirs[ci]);
      const thrVal = variant.fcVmBands[ci][0];
      if (s >= thrVal) {
        const ratio = thrVal !== 0 ? s / thrVal : s;
        if (ratio > bestRatio) { bestCI = ci; bestRatio = ratio; bestS = s; }
      }
    }
    if (bestCI < 0) return null;
    const scores = classes[bestCI].points.map(pt => variant.fcKappas[bestCI] * v3dot(pt, variant.wDirs[bestCI]));
    const maxS = Math.max(...scores);
    const thrVal = variant.fcVmBands[bestCI][0];
    const t = maxS > thrVal ? (bestS - thrVal) / (maxS - thrVal) : 1;
    return { color: classes[bestCI].color, opacity: 0.01 + 0.74 * Math.min(1, t) };
  }
  function evalKent(p) {
    let bestCI = -1, bestRatio = -Infinity, bestS = 0;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = variant.fcKents[ci].kentScore(p);
      const thrVal = variant.fcKents[ci].bands[0];
      if (s >= thrVal) {
        const ratio = thrVal !== 0 ? s / thrVal : s;
        if (ratio > bestRatio) { bestCI = ci; bestRatio = ratio; bestS = s; }
      }
    }
    if (bestCI < 0) return null;
    const scores = classes[bestCI].points.map(pt => variant.fcKents[bestCI].kentScore(pt));
    const maxS = Math.max(...scores);
    const thrVal = variant.fcKents[bestCI].bands[0];
    const t = maxS > thrVal ? (bestS - thrVal) / (maxS - thrVal) : 1;
    return { color: classes[bestCI].color, opacity: 0.01 + 0.74 * Math.min(1, t) };
  }
  const evals = { "fc-msp": evalMSP, "fc-mls": evalMLS, "fc-ebo": evalEBO, "fc-ts": evalTS, "fc-cos": evalCos, "fc-vmf": evalVmf, "fc-kent": evalKent };
  for (const key of ["fc-msp", "fc-mls", "fc-ebo", "fc-ts", "fc-cos", "fc-vmf", "fc-kent"]) {
    const g = new THREE.Group();
    makeMethodSphere(evals[key], g);
    parentGroups[key].add(g);
    subGroups[key] = g;
  }
  return subGroups;
}

let fcBalGroups = {}, fcUnbalGroups = {};
if (!PRE) {
  fcBalGroups = makeFCTextures(fcBal, layers);
  fcUnbalGroups = makeFCTextures(fcUnbal, layers);
  for (const key of ["fc-msp", "fc-mls", "fc-ebo", "fc-ts", "fc-cos", "fc-vmf", "fc-kent"]) fcUnbalGroups[key].visible = false;
}
// (FC textures for PRE path are computed lazily via showMethodAtQ)

// ── FC 3D markers ──
const fcMarkers = new THREE.Group();
fcMarkers.visible = false;
scene.add(fcMarkers);

function rebuildFCMarkers() {
  // Dispose old GPU resources before removing
  for (var i = fcMarkers.children.length - 1; i >= 0; i--) {
    var child = fcMarkers.children[i];
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
    fcMarkers.remove(child);
  }
  for (const c of classes) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.020, 10, 10),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(c.color), transparent: true, opacity: 0.35 }));
    m.position.set(...c.muHat); fcMarkers.add(m);
    const wire = new THREE.Mesh(new THREE.SphereGeometry(0.021, 10, 10),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.4 }));
    wire.position.set(...c.muHat); fcMarkers.add(wire);
  }
  for (let ci = 0; ci < nClasses; ci++) {
    const wDir = v3norm(fc.W[ci]), mu = classes[ci].muHat;
    const ang = Math.acos(Math.max(-1, Math.min(1, v3dot(mu, wDir))));
    if (ang >= 0.5 * Math.PI / 180) {
      const nSeg = Math.max(8, Math.ceil(ang / 0.02)), sinA = Math.sin(ang), pts = [];
      for (let s = 0; s <= nSeg; s++) {
        const t = s / nSeg, wa = Math.sin((1-t)*ang)/sinA, wb = Math.sin(t*ang)/sinA;
        const p = v3norm(v3add(v3scale(mu, wa), v3scale(wDir, wb)));
        pts.push(new THREE.Vector3(p[0]*1.008, p[1]*1.008, p[2]*1.008));
      }
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineDashedMaterial({ color: new THREE.Color(classes[ci].color),
          dashSize: 0.015, gapSize: 0.01, transparent: true, opacity: 0.5 }));
      line.computeLineDistances(); fcMarkers.add(line);
    }
  }
  for (let ci = 0; ci < nClasses; ci++) {
    const wDir = v3norm(fc.W[ci]);
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.022),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(classes[ci].color) }));
    m.position.set(wDir[0]*1.008, wDir[1]*1.008, wDir[2]*1.008); fcMarkers.add(m);
  }
}
rebuildFCMarkers();

// ══════════════════════════════════════════════
//  CONFORMAL PREDICTION
// ══════════════════════════════════════════════
let cpScoreType = "aps";  // "aps", "msp", or "emb"
let cpAlpha = 0.10;

// conformalCalibrateScore, conformalPredictScore — imported from src/methods/conformal.js

function protoLogitsFn(x) {
  return W.map(w => TAU * v3dot(x, w));
}

// _cpShaderKey, _cpEnsureCalibScores, _cpQhat, drawConformal — moved to shader-manager

const cpAlphaLabel = document.getElementById("cp-alpha-label");
const cpAlphaSliderEl = document.getElementById("cp-alpha");

function cpUpdateCacheBar() {
  document.getElementById("alpha-play").style.display = "";
}

function cpHideSpinner() {
  cpAlphaLabel.innerHTML = "\u03B1:";
}

// ── orbit controls — moved to src/rendering/orbit-controls.js ──
initOrbitControls({ THREE, camera, renderer, scene, container, H3, raycastSphere, showProbe });
// ── descriptions ──
const descs = {
  none:
    "Training embeddings on S\u00B2 (the unit sphere in \u211D\u00B3) because each vector is L2-normalized. " +
    "Large dots are class prototypes (\u03BC\u0302 = mean direction). Red dots are OOD locations with especially low MSP. " +
    "Drag to rotate. Class D is deliberately anisotropic \u2014 elongated along one tangent direction.",
  mds:
    "Per-class covariance Mahalanobis: each class gets its own mean and covariance matrix \u03A3\u1D62 " +
    "(describing how that class\u2019s points spread in each direction). Ellipsoids show 1\u03C3, 2\u03C3, 3\u03C3 contours. " +
    "Note: ellipsoids are centered at the ambient mean (inside the sphere) and extend through the interior. " +
    "Class D\u2019s ellipsoid is elongated, reflecting its anisotropic spread. Also called QDA (Quadratic Discriminant Analysis).",
  "mds-s":
    "Shared covariance Mahalanobis: per-class means but one pooled covariance \u03A3_pool. " +
    "All classes share the same ellipsoid shape \u2014 only the center differs. This is what MDS (Lee et al. 2018) and sklearn\u2019s LDA use. " +
    "The pooled shape is dominated by class A (n=200). Class D\u2019s anisotropy is lost. Class C\u2019s 15 points have negligible influence on the shape.",
  msp: "", cos: "", vmf: "", kent: "", mls: "", ebo: "", ts: "", knn: "", rmds: "", kde: "", odin: "", vim: "",
  "fc-msp": "", "fc-mls": "", "fc-ebo": "", "fc-ts": "", "fc-odin": "", "fc-vim": "",
  "fc-cos": "", "fc-vmf": "", "fc-kent": "",
  "cp-proto": "", "cp-fc": ""
};

var cachedThr = null;  // declared early; assigned after allThresholds is defined

function qNoteStr() {
  return " \u2014 Threshold quantile q = " + currentQ.toFixed(3) +
    ": the " + (currentQ * 100).toFixed(0) + "th percentile of training scores sets the accept/reject boundary. " +
    "Adjust q with the slider in the table to contract (higher q) or expand (lower q) the accepted region.";
}

function updateFCDescs() {
  const bal = fcBalanced;
  const bStr = bal ? "class-balanced" : "unweighted";
  const bNote = bal
    ? "Class-balanced loss (w\u1D62 = N/Kn\u1D62) gives each class equal gradient influence. "
    : "Unweighted loss \u2014 majority classes dominate gradients; minority classes may get smaller norms and less favorable biases. ";
  const qN = qNoteStr();
  descs["fc-msp"] =
    "MSP (learned FC): logits = W\u1D62\u1D40x + b\u1D62, fit by " + bStr + " logistic regression (500 iters, lr\u200a=\u200a0.05). " +
    "Spheres = prototype directions (initial); octahedra = learned weight directions (final); dashed arcs = angular drift. " +
    bNote +
    "Biases [" + fc.B.map(b => b.toFixed(2)).join(", ") + "], norms [" + fc.wNorms.map(n => n.toFixed(2)).join(", ") + "]. \u03B3 = " + (cachedThr ? cachedThr.fcMspGamma : fc.gamma).toFixed(3) + "." + qN;
  descs["fc-mls"] =
    "MLS (learned FC): score = max(W\u1D62\u1D40x + b\u1D62). " +
    (bal ? "Class-balanced training produces more equitable norms and biases across classes."
         : "Unweighted training lets majority classes claim more logit space; minority classes may shrink.") + qN;
  descs["fc-ebo"] =
    "EBO (learned FC): score = log \u03A3\u1D62 exp(W\u1D62\u1D40x + b\u1D62). " +
    (bal ? "Class-balanced weighting gives minority classes comparable weight norms, preventing patch-vanishing."
         : "Unweighted: minority classes contribute less to the energy sum, so their patches may shrink or vanish.") + qN;
  descs["fc-ts"] =
    "TempScale (learned FC): MSP on learned logits, T = " + fc.bestT.toFixed(1) + ". " +
    (bal ? "Class-balanced weights mean T calibrates over evenly-sized class contributions."
         : "Temperature rescaling adjusts confidence but does not fix class-imbalance effects in W and b.") +
    " \u03B3 = " + (cachedThr ? cachedThr.fcTsGamma : fc.tsGamma).toFixed(3) + "." + qN;
  descs["fc-cos"] =
    "Per-class cosine (learned FC): accept if cos(x, w\u0302\u1D62) \u2265 t\u1D62, using learned weight directions instead of prototype means. " +
    bNote + "Compare with prototype Cosine to see how learning shifts the acceptance caps." + qN;
  descs["fc-vmf"] =
    "von Mises-Fisher (learned FC): p\u1D62(x) \u221D exp(\u03BA\u1D62 \u00B7 cos(x, w\u0302\u1D62)), with \u03BA\u1D62 estimated from " +
    "training cosines to the learned weight directions. " + bNote +
    "Compare with prototype vMF." + qN;
  descs["fc-kent"] =
    "Kent (learned FC): anisotropic extension of vMF using learned weight directions as anchors. " +
    "Tangent-plane covariance fitted relative to normalize(W\u1D62). " + bNote +
    "Compare with prototype Kent to see how learning shifts the elliptical caps." + qN;
  descs["fc-odin"] =
    "ODIN (learned FC): MSP with T = " + odinT + " and \u03B5 = " + odinEps + ", using learned FC logits. " +
    bNote + "The gradient for input perturbation is computed from the learned weight vectors W\u1D62. " +
    "Compare with prototype ODIN to see how learning shifts the accepted region." + qN;
  descs["fc-vim"] =
    "ViM (learned FC): virtual-logit matching using FC logits but the same feature-space residual. " +
    bNote + "\u03B1 is recalibrated to match FC max logit magnitudes. " +
    "Same banded pattern as prototype ViM \u2014 the 1D null space is a feature-space property, " +
    "not a logit-space property, so learning new weights cannot fix it. " +
    "This illustrates ViM\u2019s fundamental requirement: feature_dim \u226B num_classes." + qN;
}
updateFCDescs();

function updateCPDescs() {
  const scoreNames = { aps: "APS (adaptive prediction sets)", msp: "1\u2212MSP", emb: "Embedding cosine" };
  const scoreStr = scoreNames[cpScoreType];
  const isEmb = cpScoreType === "emb";
  const scoreNote = isEmb
    ? "Non-conformity score = 1 \u2212 cos(x, \u03BC\u0302_y). Include class k if cos(x, \u03BC\u0302_k) \u2265 1 \u2212 q\u0302. " +
      "Operates directly in embedding space \u2014 no softmax or logits needed. "
    : "Non-conformity score from softmax probabilities. ";
  const alphaNote = " \u2014 \u03B1 = " + cpAlpha.toFixed(3) +
    " (miscoverage rate): the prediction set contains the true class with probability \u2265 " +
    (1-cpAlpha).toFixed(3) + ". Same role as in hypothesis testing, but the guarantee is distribution-free. " +
    "Lower \u03B1 = larger sets, stronger guarantee. Higher \u03B1 = smaller sets; if empty, the point is OOD.";
  descs["cp-proto"] =
    "Conformal prediction (" + (isEmb ? "embedding space" : "prototype logits") + ", " + scoreStr + "): " +
    "prediction sets C(x) with finite-sample coverage guarantee. " +
    scoreNote +
    "Solid = single-class prediction. Faded = multi-class (ambiguous). Empty = OOD (no class passes)." + alphaNote;
  descs["cp-fc"] =
    "Conformal prediction (" + (isEmb ? "embedding space" : "learned FC logits") + ", " + scoreStr + "): " +
    (isEmb ? "same as Proto (embedding scores don\u2019t use learned weights). " : "calibrated on FC softmax probabilities. ") +
    "Solid = single-class. Faded = multi-class. Empty = OOD." + alphaNote;
}
updateCPDescs();

const classNotes = {
  A: "",
  B: "",
  C: "few-shot",
  D: "stretched \u00d73 along one tangent axis",
  E: "direction near B"
};
const legendBody = document.getElementById("class-legend-body");
classes.forEach((c, i) => {
  const tr = document.createElement("tr");
  tr.style.borderBottom = "1px solid #f3f4f6";
  const note = classNotes[c.label] || "";
  tr.innerHTML =
    `<td style="padding:3px 6px;"><span class="chip" style="background:${c.color}"></span><b>${c.label}</b></td>` +
    `<td style="padding:3px 6px; text-align:right; font-family:ui-monospace,monospace;">${c.n}</td>` +
    `<td style="padding:3px 6px; text-align:right; font-family:ui-monospace,monospace;">${classDefs[i].kappa}</td>` +
    `<td style="padding:3px 6px; color:#6b7280;">${note}</td>`;
  legendBody.appendChild(tr);
});

// ── OOD point index labels (sprites) ──
oodPoints.forEach((p, k) => {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 64;
  const ctx2 = c.getContext("2d");
  ctx2.fillStyle = "#E24B4A";
  ctx2.font = "bold 48px Inter, system-ui, sans-serif";
  ctx2.textAlign = "center";
  ctx2.textBaseline = "middle";
  ctx2.fillText(String(k + 1), 32, 32);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.position.set(p[0] * 1.12, p[1] * 1.12, p[2] * 1.12);
  sprite.scale.set(0.08, 0.08, 1);
  farOodGroup.add(sprite);
});

// ── OOD decision computation ──
// Training Mahalanobis thresholds
let trainMahalPC, mahalPCThr, trainMahalS, mahalSThr;
if (PRE) {
  trainMahalPC = PRE.trainMahalPC;
  mahalPCThr = PRE.mahalPCThr;
  trainMahalS = PRE.trainMahalS;
  mahalSThr = PRE.mahalSThr;
} else {
  trainMahalPC = [];
  for (let ci = 0; ci < classes.length; ci++) {
    for (const pt of classes[ci].points) {
      trainMahalPC.push(mahalDist2(pt, classes[ci].muBar, classes[ci].invCov));
    }
  }
  mahalPCThr = quantile(trainMahalPC, 1 - Q);
  trainMahalS = [];
  for (let ci = 0; ci < classes.length; ci++) {
    for (const pt of classes[ci].points) {
      trainMahalS.push(mahalDist2(pt, classes[ci].muBar, poolInvCov));
    }
  }
  mahalSThr = quantile(trainMahalS, 1 - Q);
}

// ── allThresholds: recompute all thresholds at a given q (Feature 3) ──
// Pre-sort all training score arrays for O(1) quantile lookups.
// Also pre-compute FC training scores (were being recomputed on every q change).
var _sortedTrainMsp = trainMsp.map(function(xs) { return xs.slice().sort(function(a,b){return a-b}); });
var _sortedMLSscores = trainMLSscores.slice().sort(function(a,b){return a-b});
var _sortedEBOscores = trainEBOscores.slice().sort(function(a,b){return a-b});
var _sortedTSMSP = trainTSMSP.map(function(xs) { return xs.slice().sort(function(a,b){return a-b}); });
var _sortedCosScores = classes.map(function(c) { return c.cosScores.slice().sort(function(a,b){return a-b}); });
var _sortedVmScores = classes.map(function(c) { return c.vmScores.slice().sort(function(a,b){return a-b}); });
var _sortedKentScores = classes.map(function(c) { return c.kentScores.slice().sort(function(a,b){return a-b}); });
var _sortedMahalPC = trainMahalPC.slice().sort(function(a,b){return a-b});
var _sortedMahalS = trainMahalS.slice().sort(function(a,b){return a-b});
var _sortedKNNdists = trainKNNdists.slice().sort(function(a,b){return a-b});
var _sortedRMDS = trainRMDS.slice().sort(function(a,b){return a-b});
var _sortedKDEscores = trainKDEscores.slice().sort(function(a,b){return a-b});
var _sortedOdinPerClass = _storedOdinPerClass.map(function(xs) { return xs.slice().sort(function(a,b){return a-b}); });
var _sortedVimScores = _storedTrainVimScores.slice().sort(function(a,b){return a-b});

// Pre-compute FC training scores (these depend on the current FC variant)
function _buildSortedFCScores() {
  var fcMspPerClass = classes.map(function(c, i) {
    return c.points.map(function(p) { return softmax(fc.logitsFn(p))[i]; }).sort(function(a,b){return a-b});
  });
  var fcMlsScores = fcTrainX.map(function(p) { return Math.max.apply(null, fc.logitsFn(p)); }).sort(function(a,b){return a-b});
  var fcEboScores = fcTrainX.map(function(p) { return logsumexp(fc.logitsFn(p)); }).sort(function(a,b){return a-b});
  var fcTsPerClass = classes.map(function(c, i) {
    return c.points.map(function(p) { return softmax(fc.logitsFn(p).map(function(l){return l/fc.bestT;}))[i]; }).sort(function(a,b){return a-b});
  });
  var fcCosScores = classes.map(function(c, ci) {
    return c.points.map(function(p) { return v3dot(p, fc.wDirs[ci]); }).sort(function(a,b){return a-b});
  });
  var fcVmScores = classes.map(function(c, ci) {
    return c.points.map(function(p) { return fc.fcKappas[ci] * v3dot(p, fc.wDirs[ci]); }).sort(function(a,b){return a-b});
  });
  var fcKentScores = classes.map(function(c, ci) {
    return c.points.map(function(p) { return fc.fcKents[ci].kentScore(p); }).sort(function(a,b){return a-b});
  });
  var fcOdinSorted = _currentFcOdinPerClass().map(function(xs) { return xs.slice().sort(function(a,b){return a-b}); });
  var fcVimSorted = _currentFcVimScores().slice().sort(function(a,b){return a-b});
  return { fcMspPerClass: fcMspPerClass, fcMlsScores: fcMlsScores, fcEboScores: fcEboScores,
           fcTsPerClass: fcTsPerClass, fcCosScores: fcCosScores, fcVmScores: fcVmScores,
           fcKentScores: fcKentScores, fcOdinSorted: fcOdinSorted, fcVimSorted: fcVimSorted };
}
var _sortedFC = _buildSortedFCScores();

// ── Initialize threshold engine with sorted arrays ──
initSortedScores({
  trainMsp: _sortedTrainMsp, mlsScores: _sortedMLSscores, eboScores: _sortedEBOscores,
  tsMsp: _sortedTSMSP, cosScores: _sortedCosScores, vmScores: _sortedVmScores,
  kentScores: _sortedKentScores, mahalPC: _sortedMahalPC, mahalS: _sortedMahalS,
  knnDists: _sortedKNNdists, rmds: _sortedRMDS, kdeScores: _sortedKDEscores,
  odinPerClass: _sortedOdinPerClass, vimScores: _sortedVimScores,
}, _sortedFC);

// allThresholds delegates to the threshold engine
function allThresholds(q) {
  return engineAllThresholds(q, nClasses);
}
if (PRE) {
  cachedThr = PRE.cachedThr;
  // Add ODIN/ViM thresholds that may not be in precomputed data
  if (!cachedThr.odinGamma) {
    cachedThr.odinGamma = odinGamma;
    cachedThr.fcOdinGamma = fcOdinGamma;
    cachedThr.vimThr = vimThreshold;
    cachedThr.fcVimThr = fcVimThreshold;
  }
  // Always recompute KDE threshold (kdeScore now returns log-density)
  cachedThr.kdeThr = quantile(trainKDEscores, Q);
} else {
  cachedThr = allThresholds(Q);
}

// qShowSpinner, qHideSpinner — moved to src/ui/controls.js

// ── Initialize shader manager ──
initShaderManager({
  THREE,
  layers,
  classes, W, TAU, poolInvCov, muAll, bestT,
  vimU, vimAlpha, getFcVimAlpha: () => fcVimAlphaVal,
  odinT, odinEps,
  trainMsp, sortedMLSscores: _sortedMLSscores, sortedEBOscores: _sortedEBOscores,
  sortedTSMSP: _sortedTSMSP, sortedOdinPerClass: _sortedOdinPerClass,
  sortedVimScores: _sortedVimScores, sortedRMDS: _sortedRMDS,
  trainKDEscores, trainKNNdists: trainKNNdists || [],
  getFC: () => fc,
  getFCBalanced: () => fcBalanced,
  getFCBal: () => fcBal,
  getFCUnbal: () => fcUnbal,
  getCachedThr: () => cachedThr,
  getCurrentQ: () => currentQ,
  getActiveMode: () => activeMode,
  getCpScoreType: () => cpScoreType,
  getCpAlpha: () => cpAlpha,
  getSortedFC: () => _sortedFC,
  getAllThresholds: (q) => allThresholds(q),
  getScoringCtx: () => _scoringCtx(),
  onReadyChange: () => { if (typeof qUpdateTrackBar === "function") qUpdateTrackBar(); },
  onLoading: () => qShowSpinner(),
});

// Aliases: replace inline functions with shader-manager versions
function showMethodAtQ(method, q) { smShowMethodAtQ(method, q); }
function drawConformal(mode) { smDrawConformal(mode); }

// ── Bind FC-effects to actual rendering/cache functions ──
bindFCEffects({
  switchVariant: (balanced) => {
    fcBalanced = balanced;
    fc = balanced ? fcBal : fcUnbal;
    for (const key of ["fc-msp","fc-mls","fc-ebo","fc-ts","fc-odin","fc-vim","fc-cos","fc-vmf","fc-kent"]) {
      if (fcBalGroups[key]) fcBalGroups[key].visible = balanced;
      if (fcUnbalGroups[key]) fcUnbalGroups[key].visible = !balanced;
    }
  },
  rebuildMarkers: () => rebuildFCMarkers(),
  invalidateProceduralCache: () => invalidateAllFCProceduralCaches(),
  invalidateConformalCache: () => invalidateConformalCache(),
  clearCalibScores: () => clearCalibScores(),
  recomputeOdinVim: (balanced) => {
    fcOdinGamma = _currentFcOdinPerClass().map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / classes.length;
    fcVimThreshold = quantile(_currentFcVimScores(), Q);
    if (PRE && balanced && PRE.fcBal && PRE.fcBal.vimAlpha !== undefined) {
      fcVimAlphaVal = PRE.fcBal.vimAlpha;
    } else if (PRE && !balanced && PRE.fcUnbal && PRE.fcUnbal.vimAlpha !== undefined) {
      fcVimAlphaVal = PRE.fcUnbal.vimAlpha;
    } else {
      var fcMML = allTrainPts.map(p => Math.max(...fc.logitsFn(p))).reduce((a,b) => a+b, 0) / allTrainPts.length;
      fcVimAlphaVal = vimMeanResNorm > 1e-10 ? fcMML / vimMeanResNorm : 0;
    }
  },
  rebuildSortedScores: () => {
    _sortedFC = _buildSortedFCScores();
    rebuildSortedFC(() => _sortedFC);
  },
  updateDescriptions: () => { cachedThr = getCachedThresholds() || allThresholds(currentQ); updateQDescs(); updateFCDescs(); },
  rerenderActive: (mode) => {
    cachedThr = allThresholds(currentQ);
    if (mode && mode !== "none") {
      const isCP = cpModes.has(mode);
      if (isCP) drawConformal(mode);
      else if (mode !== "mds" && mode !== "mds-s") showMethodAtQ(mode, currentQ);
    }
    updateDecisionTable(activeMode);
    if (typeof bgRestart === "function") bgRestart();
  },
});

// Shader strings, procedural/score-field/conformal caches, showMethodAtQ,
// _getProceduralConfig, _buildProceduralMesh, _updateProceduralUniforms,
// _buildShaderMesh, _updateShaderUniforms, computeScoreFieldAsync,
// _buildConformalProceduralMesh, _updateConformalProceduralUniforms
// — all moved to src/rendering/shader-manager.js

// Temporary set for FC mode check before fcModes is defined
const fcModes2 = new Set(["fc-msp", "fc-mls", "fc-ebo", "fc-ts", "fc-odin", "fc-vim", "fc-cos", "fc-vmf", "fc-kent"]);

let nearOodPoints = []; // will be computed after oodDecision3d is defined
const nearOodGroup = new THREE.Group();
nearOodGroup.visible = false;
scene.add(nearOodGroup);
let activeOodPoints = oodPoints;

const methodNames = {
  none: "Data only", mds: "Mahalanobis (per-class)", "mds-s": "Mahalanobis (shared)",
  rmds: "RMDS", knn: "KNN", kde: "KDE", cos: "Per-class cosine", vmf: "vMF", kent: "Kent",
  msp: "MSP (proto)", mls: "MLS (proto)", ebo: "EBO (proto)", ts: "TempScale (proto)",
  odin: "ODIN (proto)", vim: "ViM (proto)",
  "fc-msp": "MSP (learned FC)", "fc-mls": "MLS (learned FC)", "fc-ebo": "EBO (learned FC)", "fc-ts": "TempScale (learned FC)",
  "fc-odin": "ODIN (learned FC)", "fc-vim": "ViM (learned FC)",
  "fc-cos": "Cosine (learned FC)", "fc-vmf": "vMF (learned FC)", "fc-kent": "Kent (learned FC)",
  "cp-proto": "Conformal (proto)", "cp-fc": "Conformal (learned FC)"
};

// ── Scoring context: carries runtime state for registry decideFn/classScoresFn ──
function _scoringCtx() {
  return {
    W, TAU, classes, poolInvCov, muAll,
    allTrainPts, allTrainLabels, kVal, kdeBandwidth,
    bestT, vimU, vimAlpha,
    fc, fcVimAlpha: fcVimAlphaVal,
    thr: cachedThr, odinT, odinEps,
  };
}

function oodDecision3d(x, method) {
  const m = getMethod(method);
  if (!m || !m.decideFn) return null;
  return m.decideFn(x, _scoringCtx());
}

const oodDiv = document.getElementById("ood-decisions");

function classScores3d(x, method) {
  const m = getMethod(method);
  if (!m || !m.classScoresFn) return null;
  return m.classScoresFn(x, _scoringCtx());
}

// renderMiniBar — moved to src/ui/decision-table.js

// ── Near-OOD points computation (Feature 1) ──
// Must be after oodDecision3d is defined
if (PRE) {
  nearOodPoints.length = 0;
  for (const p of PRE.nearOodPoints) nearOodPoints.push(p);
} else {
  (function computeNearOod() {
    const methods = ["msp", "cos", "mds", "rmds"];
    const MIN_SEP = 0.35;

    // Generate inter-class boundary candidates: walk the geodesic between each
    // class pair, sampling points from 30% to 70% of the arc. These target the
    // decision boundaries where methods are most likely to disagree.
    const boundaryCandidates = [];
    for (let i = 0; i < classes.length; i++) {
      for (let j = i + 1; j < classes.length; j++) {
        const mi = classes[i].muHat, mj = classes[j].muHat;
        const ang = Math.acos(Math.max(-1, Math.min(1, v3dot(mi, mj))));
        if (ang < 0.01) continue;
        const sinA = Math.sin(ang);
        for (let t = 3; t <= 7; t++) {
          const frac = t / 10;
          const wa = Math.sin((1 - frac) * ang) / sinA;
          const wb = Math.sin(frac * ang) / sinA;
          boundaryCandidates.push(v3norm(v3add(v3scale(mi, wa), v3scale(mj, wb))));
        }
      }
    }

    // Score = disagreement × proximity to nearest class.
    // Points far from all classes ("pacific ocean") get near-zero scores.
    function scorePt(pt) {
      const decisions = [];
      for (const m of methods) {
        const d = oodDecision3d(pt, m);
        if (!d) continue;
        const accepted = d.accepted !== undefined ? d.accepted
          : (d.lo ? d.val <= d.thr : d.val >= d.thr);
        const cls = d.acceptCls !== undefined && d.acceptCls >= 0 ? d.acceptCls : d.cls;
        decisions.push({ accepted, cls });
      }
      if (decisions.length < 2) return 0;
      const acceptCount = decisions.filter(d => d.accepted).length;
      const rejectCount = decisions.length - acceptCount;
      const arScore = (acceptCount > 0 && rejectCount > 0) ? Math.min(acceptCount, rejectCount) * 2 : 0;
      const acceptedClasses = new Set(decisions.filter(d => d.accepted).map(d => d.cls));
      const classScore = acceptedClasses.size > 1 ? acceptedClasses.size : 0;
      const disagree = arScore + classScore;
      if (disagree === 0) return 0;
      // Proximity: max cosine similarity to any class prototype
      const maxCos = Math.max(...classes.map(c => v3dot(pt, c.muHat)));
      return disagree * Math.max(0, maxCos);
    }

    // Score all grid points
    const candidates = [];
    for (let gi = 0; gi < grid4k.length; gi++) {
      const pt = grid4k[gi];
      if (!oodPoints.every(fp => (1 - v3dot(fp, pt)) > MIN_SEP)) continue;
      const s = scorePt(pt);
      if (s > 0) candidates.push({ pt, score: s, src: "grid" });
    }
    // Score boundary candidates
    for (const pt of boundaryCandidates) {
      if (!oodPoints.every(fp => (1 - v3dot(fp, pt)) > MIN_SEP)) continue;
      const s = scorePt(pt);
      if (s > 0) candidates.push({ pt, score: s, src: "boundary" });
    }

    candidates.sort((a, b) => b.score - a.score);

    const selected = [];
    for (const cand of candidates) {
      const ok = selected.every(sp => (1 - v3dot(sp, cand.pt)) > MIN_SEP);
      if (ok) {
        selected.push(cand.pt);
        if (selected.length >= 4) break;
      }
    }
    nearOodPoints.length = 0;
    for (const p of selected) nearOodPoints.push(p);
  })();
}
// Build visuals (always runs)
nearOodGroup.add(makePointCloud(nearOodPoints, "#E24B4A", 0.055));
nearOodPoints.forEach((p, k) => {
  const c2 = document.createElement("canvas");
  c2.width = 64; c2.height = 64;
  const ctx3 = c2.getContext("2d");
  ctx3.fillStyle = "#E24B4A";
  ctx3.font = "bold 48px Inter, system-ui, sans-serif";
  ctx3.textAlign = "center";
  ctx3.textBaseline = "middle";
  ctx3.fillText(String(k + 1), 32, 32);
  const tex2 = new THREE.CanvasTexture(c2);
  const mat2 = new THREE.SpriteMaterial({ map: tex2, transparent: true, depthWrite: false });
  const sprite2 = new THREE.Sprite(mat2);
  sprite2.position.set(p[0] * 1.12, p[1] * 1.12, p[2] * 1.12);
  sprite2.scale.set(0.08, 0.08, 1);
  nearOodGroup.add(sprite2);
});

// ── UI constants (must be before initDecisionTable / initControls) ──
const desc = document.getElementById("desc");
let activeMode = "none";
const fcModes = new Set(["fc-msp","fc-mls","fc-ebo","fc-ts","fc-odin","fc-vim","fc-cos","fc-vmf","fc-kent"]);
const cpModes = new Set(["cp-proto","cp-fc"]);
const allIds = ["none","mds","mds-s","msp","mls","ebo","ts","odin","vim","cos","vmf","kent","knn","rmds","kde","fc-msp","fc-mls","fc-ebo","fc-ts","fc-odin","fc-vim","fc-cos","fc-vmf","fc-kent","cp-proto","cp-fc"];

// updateDecisionTable, renderMiniBar, click handler — moved to src/ui/decision-table.js
initDecisionTable({
  classes, methodNames, cpModes,
  getOodDiv: () => oodDiv,
  getActiveOodPoints: () => activeOodPoints,
  getNearOodPoints: () => nearOodPoints,
  getRandomOodPoints: () => randomOodPoints,
  getProbeDir: () => probeDir,
  getFC: () => fc,
  getFCBalanced: () => fcBalanced,
  getCpScoreType: () => cpScoreType,
  getCpAlpha: () => cpAlpha,
  protoLogitsFn,
  oodDecision3d, classScores3d,
  conformalPredictScore, cpQhat,
});
function updateDecisionTable(mode) { dtUpdateDecisionTable(mode); }
oodDiv.addEventListener("click", function(e) {
  var td = e.target.closest(".pt-link");
  if (!td) return;
  var idx = parseInt(td.dataset.ptIdx);
  if (isNaN(idx) || idx < 0 || idx >= activeOodPoints.length) return;
  lookAtPoint(activeOodPoints[idx]);
});
// ── UI controls — moved to src/ui/controls.js ──
initControls({
  layers, layerKeys, fcModes, cpModes, allIds, descEl: desc,
  protoMarkers, fcMarkers,
  fcBalancedLabel: document.getElementById("fc-balanced-label"),
  cpSettingsEl: document.getElementById("cp-settings"),
  getDescs: () => descs,
  getActiveMode: () => activeMode, setActiveMode: (m) => { activeMode = m; },
  getCurrentQ: () => currentQ, setCurrentQ: (v) => { currentQ = v; },
  getCpAlpha: () => cpAlpha, setCpAlpha: (v) => { cpAlpha = v; },
  getCpScoreType: () => cpScoreType, setCpScoreType: (v) => { cpScoreType = v; },
  getFCBalanced: () => fcBalanced, setFCBalanced: (v) => { fcBalanced = v; fc = v ? fcBal : fcUnbal; },
  setCachedThr: (t) => { cachedThr = t; },
  showMethodAtQ, drawConformal, bgRestart: () => bgRestart(),
  updateDecisionTable, updateQDescs, updateFCDescs, updateCPDescs,
  updateConformalUniforms: updateConformalProceduralUniforms,
  qHideSpinner, cpHideSpinner, cpUpdateCacheBar,
  isScoreFieldReady,
});
function setActive(mode) { smSetActive(mode); }

// ── Probe (Feature 2) ──
let probeDir = null;

function showProbe(dir) {
  probeDir = dir;
  AppState.set({ probeDir: dir });
  probeMarker.position.set(dir[0] * 1.015, dir[1] * 1.015, dir[2] * 1.015);
  probeMarker.visible = true;
  updateDecisionTable(activeMode);
}

function clearProbe() {
  probeDir = null;
  AppState.set({ probeDir: null });
  probeMarker.visible = false;
  updateDecisionTable(activeMode);
}

// ── OOD toggle wiring (Feature 1) ──
const oodLegendDesc = document.getElementById("ood-legend-desc");
document.querySelectorAll('input[name="ood-set"]').forEach(radio => {
  radio.addEventListener("change", function() {
    AppState.set({ oodSetType: this.value });
    randomOodGroup.visible = false;
    if (this.value === "far") {
      farOodGroup.visible = true;
      nearOodGroup.visible = false;
      activeOodPoints = oodPoints;
      oodLegendDesc.textContent = "unknown / low-MSP shell points";
    } else {
      farOodGroup.visible = false;
      nearOodGroup.visible = true;
      activeOodPoints = nearOodPoints;
      oodLegendDesc.textContent = "boundary points where methods disagree";
    }
    updateDecisionTable(activeMode);
  });
});

// ── Random test points ──
var randomOodGroup = new THREE.Group();
randomOodGroup.visible = false;
scene.add(randomOodGroup);
var randomOodPoints = [];

function generateRandomOodPoints() {
  // Dispose old GPU resources before clearing
  for (var di = randomOodGroup.children.length - 1; di >= 0; di--) {
    var child = randomOodGroup.children[di];
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (child.material.map) child.material.map.dispose();
      child.material.dispose();
    }
    randomOodGroup.remove(child);
  }
  randomOodPoints.length = 0;
  // Generate 4 random points on S²
  for (var ri = 0; ri < 4; ri++) {
    var th = Math.acos(2 * Math.random() - 1);
    var ph = 2 * Math.PI * Math.random();
    randomOodPoints.push([Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)]);
  }
  // Build markers (same style as far-OOD)
  randomOodGroup.add(makePointCloud(randomOodPoints, "#E24B4A", 0.055));
  randomOodPoints.forEach(function(p, k) {
    var c2 = document.createElement("canvas");
    c2.width = 64; c2.height = 64;
    var ctx3 = c2.getContext("2d");
    ctx3.fillStyle = "#E24B4A";
    ctx3.font = "bold 48px Inter, system-ui, sans-serif";
    ctx3.textAlign = "center"; ctx3.textBaseline = "middle";
    ctx3.fillText(String(k + 1), 32, 32);
    var tex2 = new THREE.CanvasTexture(c2);
    var mat2 = new THREE.SpriteMaterial({ map: tex2, transparent: true, depthWrite: false });
    var sprite2 = new THREE.Sprite(mat2);
    sprite2.position.set(p[0] * 1.12, p[1] * 1.12, p[2] * 1.12);
    sprite2.scale.set(0.08, 0.08, 1);
    randomOodGroup.add(sprite2);
  });
}

document.getElementById("ood-random").addEventListener("click", function() {
  // Deselect radio buttons
  document.querySelectorAll('input[name="ood-set"]').forEach(function(r) { r.checked = false; });
  farOodGroup.visible = false;
  nearOodGroup.visible = false;
  generateRandomOodPoints();
  randomOodGroup.visible = true;
  activeOodPoints = randomOodPoints;
  oodLegendDesc.textContent = "random points on S\u00B2";
  updateDecisionTable(activeMode);
});

// ── Description templates ──
function updateQDescs() {
  const qNote = qNoteStr();
  descs.msp = "Maximum Softmax Probability: logits = \u03C4 \u00B7 cos(x, \u03BC\u0302\u1D62), then softmax to probabilities. Colored patches show where max softmax \u2265 \u03B3 = " + cachedThr.mspGamma.toFixed(3) + ". All classes get the same global threshold \u2014 no per-class calibration." + qNote;
  descs.mls = "Maximum Logit Score: score(x) = max\u1D62 \u03C4\u00B7w\u1D62\u1D40x. Unlike MSP, this preserves absolute magnitude." + qNote;
  descs.ebo = "Energy-Based OOD (Liu et al. 2020): score(x) = log \u03A3\u1D62 exp(\u03C4\u00B7w\u1D62\u1D40x). Aggregates signal across all classes." + qNote;
  descs.ts = "TempScale (Guo et al. 2017): MSP with calibrated temperature T = " + bestT.toFixed(1) + ". \u03B3 = " + cachedThr.tsGamma.toFixed(3) + "." + qNote;
  descs.odin = "ODIN (Liang et al. 2018): MSP with T = " + odinT + " and \u03B5 = " + odinEps + ". Gradient-based input perturbation increases ID/OOD gap." + qNote;
  descs.vim = "ViM (Wang et al. 2022): Virtual-logit Matching. Augments logits with residual in null space. \u03B1 = " + vimAlpha.toFixed(2) + "." + qNote;
  descs.cos = "Per-class cosine threshold: accept if cos(x, \u03BC\u0302\u1D62) \u2265 t\u1D62." + qNote;
  descs.vmf = "von Mises-Fisher: p\u1D62(x) \u221D exp(\u03BA\u1D62 \u00B7 cos(x, \u03BC\u0302\u1D62))." + qNote;
  descs.kent = "Kent distribution (FB\u2085): anisotropic vMF with elliptical tangent-plane contours." + qNote;
  descs.knn = "KNN (Sun et al. 2022): score = cosine distance to k-th nearest neighbor (k = " + kVal + ")." + qNote;
  descs.rmds = "Relative Mahalanobis (Ren et al. 2021): min\u1D62 d\u00B2(x, \u03BC\u1D62) \u2212 d\u00B2(x, \u03BC_all)." + qNote;
  descs.kde = "KDE: per-class vMF kernel density on S\u00B2. Bandwidth \u03BA_bw = " + kdeBandwidth.toFixed(1) + "." + qNote;
}
updateQDescs();

// Remove loading placeholder
const loadingEl = document.getElementById("loading-placeholder");
if (loadingEl) { loadingEl.style.opacity = "0"; setTimeout(() => loadingEl.remove(), 600); }

// q slider, play buttons, track bar — moved to src/ui/controls.js

// Background precompute — moved to shader-manager
bgStart();
