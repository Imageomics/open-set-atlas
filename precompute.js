"use strict";
// ================================================================
//  precompute.js -- Precompute all expensive data for OOD 3D viz
//  Run: node precompute.js
//  Output: precomputed.js, tex/*.png
// ================================================================

const fs   = require("fs");
const path = require("path");
const zlib = require("zlib");

const t0 = Date.now();

// ── Minimal PNG encoder using Node built-in zlib ──
const crcTable = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  crcTable[n] = c;
}
function crc32(buf) {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) crc = crcTable[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function writePNG(filePath, width, height, rgbaBuffer) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  function makeChunk(type, data) {
    const typeBuf = Buffer.from(type, "ascii");
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const crcInput = Buffer.concat([typeBuf, data]);
    const crcVal = crc32(crcInput);
    const crcBuf = Buffer.alloc(4);
    crcBuf.writeUInt32BE(crcVal >>> 0, 0);
    return Buffer.concat([len, typeBuf, data, crcBuf]);
  }

  // IHDR
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6; // 8-bit RGBA

  // Filtered rows (prepend 0 filter byte to each row)
  const rowLen = width * 4;
  const raw = Buffer.alloc((rowLen + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (rowLen + 1)] = 0; // no filter
    rgbaBuffer.copy(raw, y * (rowLen + 1) + 1, y * rowLen, (y + 1) * rowLen);
  }

  const compressed = zlib.deflateSync(raw, { level: 9 });

  const png = Buffer.concat([
    signature,
    makeChunk("IHDR", ihdr),
    makeChunk("IDAT", compressed),
    makeChunk("IEND", Buffer.alloc(0))
  ]);

  fs.writeFileSync(filePath, png);
  console.log("  " + filePath + " (" + (png.length / 1024).toFixed(0) + " KB)");
}

// ================================================================
//  SEEDED RNG -- must match HTML exactly (Park-Miller LCG, seed=42)
// ================================================================
let _seed = 42;
function rand() { _seed = (_seed * 16807 + 0) % 2147483647; return _seed / 2147483647; }

// ================================================================
//  VECTOR MATH (3D arrays) -- copied verbatim from HTML
// ================================================================
function v3add(a,b){ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function v3sub(a,b){ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function v3scale(a,s){ return [a[0]*s, a[1]*s, a[2]*s]; }
function v3dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function v3cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function v3len(a){ return Math.sqrt(v3dot(a,a)); }
function v3norm(a){ const l=v3len(a); return l<1e-12?[0,0,0]:[a[0]/l,a[1]/l,a[2]/l]; }

// ── rotation: rotate vector v so that (0,0,1) maps to dir ──
function rotateToDir(v, dir) {
  const z = [0,0,1];
  const d = v3dot(z, dir);
  if (d > 0.9999) return v.slice();
  if (d < -0.9999) return [-v[0], -v[1], -v[2]];
  const axis = v3norm(v3cross(z, dir));
  const angle = Math.acos(Math.max(-1, Math.min(1, d)));
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const [x,y,zz] = axis;
  // Rodrigues
  return [
    (t*x*x+c)*v[0] + (t*x*y-s*zz)*v[1] + (t*x*zz+s*y)*v[2],
    (t*x*y+s*zz)*v[0] + (t*y*y+c)*v[1] + (t*y*zz-s*x)*v[2],
    (t*x*zz-s*y)*v[0] + (t*y*zz+s*x)*v[1] + (t*zz*zz+c)*v[2]
  ];
}

// ── vMF sampling on S^2 ──
function sampleVMF(mu, kappa, n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const u = rand();
    const w = 1 + (1/kappa) * Math.log(u + (1-u)*Math.exp(-2*kappa));
    const phi = 2 * Math.PI * rand();
    const r = Math.sqrt(Math.max(0, 1 - w*w));
    const local = [r*Math.cos(phi), r*Math.sin(phi), w];
    pts.push(v3norm(rotateToDir(local, mu)));
  }
  return pts;
}

// ── anisotropic sampling: vMF then stretch one tangent direction ──
function sampleAniso(mu, kappa, stretchFactor, n) {
  // Build tangent frame at mu
  let t1 = Math.abs(mu[0]) < 0.9 ? [1,0,0] : [0,1,0];
  t1 = v3norm(v3sub(t1, v3scale(mu, v3dot(t1, mu))));
  const t2 = v3cross(mu, t1);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const u = rand();
    const w = 1 + (1/kappa) * Math.log(u + (1-u)*Math.exp(-2*kappa));
    const phi = 2 * Math.PI * rand();
    const r = Math.sqrt(Math.max(0, 1 - w*w));
    // local tangent coords
    let tx = r * Math.cos(phi);
    let ty = r * Math.sin(phi);
    // stretch t1 direction
    tx *= stretchFactor;
    // reconstruct
    const p = v3add(v3add(v3scale(t1, tx), v3scale(t2, ty)), v3scale(mu, w));
    pts.push(v3norm(p));
  }
  return pts;
}

// ── statistics ──
function computeMeanDir(pts) {
  let m = [0,0,0];
  for (const p of pts) m = v3add(m, p);
  m = v3scale(m, 1/pts.length);
  return { muBar: m, muHat: v3norm(m), Rbar: v3len(m) };
}

function kappaFromRbar3D(Rbar) {
  if (Rbar < 1e-8) return 0;
  return Rbar * (3 - Rbar*Rbar) / (1 - Rbar*Rbar);
}

// Kent distribution fitting
function fitKent(pts, anchorDir) {
  const mu = anchorDir;
  // Build tangent frame at mu
  let t1 = Math.abs(mu[0]) < 0.9 ? [1,0,0] : [0,1,0];
  t1 = v3norm(v3sub(t1, v3scale(mu, v3dot(t1, mu))));
  const t2 = v3cross(mu, t1);
  // Project points onto tangent plane
  const projections = pts.map(p => [v3dot(p, t1), v3dot(p, t2)]);
  // 2x2 tangent-plane covariance
  const n = projections.length;
  let s00 = 0, s01 = 0, s11 = 0;
  for (const [x, y] of projections) { s00 += x*x; s01 += x*y; s11 += y*y; }
  s00 /= n; s01 /= n; s11 /= n;
  // Eigendecomposition of 2x2 tangent covariance
  const tr = s00 + s11;
  const disc = Math.sqrt(Math.max(0, (s00 - s11) * (s00 - s11) + 4 * s01 * s01));
  const l1 = (tr + disc) / 2;  // larger eigenvalue
  const l2 = (tr - disc) / 2;  // smaller eigenvalue
  let ev1;
  if (Math.abs(s01) > 1e-12) {
    ev1 = v3norm(v3add(v3scale(t1, l1 - s11), v3scale(t2, s01)));
  } else {
    ev1 = s00 >= s11 ? t1 : t2;
  }
  const ev2 = v3cross(mu, ev1);  // orthogonal in tangent plane
  // Estimate kappa and beta from moments
  const Rbar = v3len(pts.reduce((a, p) => v3add(a, p), [0,0,0])) / n;
  const kappa = kappaFromRbar3D(Rbar);
  // beta from eigenvalue ratio: beta ~ kappa * (l1 - l2) / (2 * (l1 + l2))
  const beta = (l1 + l2 > 1e-12) ? kappa * (l1 - l2) / (2 * (l1 + l2)) : 0;
  // Kent log-density score
  function kentScore(x) {
    return kappa * v3dot(mu, x) + beta * (v3dot(ev1, x) * v3dot(ev1, x) - v3dot(ev2, x) * v3dot(ev2, x));
  }
  // Compute training score quantile bands
  const trainScores = pts.map(p => kentScore(p));
  const bands = [0.05, 0.20, 0.50].map(q => quantile(trainScores, q));
  return { kappa, beta, gamma1: mu, gamma2: ev1, gamma3: ev2, kentScore, bands };
}

function covMatrix3(pts, mu) {
  const n = pts.length;
  const S = [[0,0,0],[0,0,0],[0,0,0]];
  for (const p of pts) {
    const d = v3sub(p, mu);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        S[i][j] += d[i]*d[j];
  }
  const denom = Math.max(1, n-1);
  const ridge = 1e-5;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) S[i][j] /= denom;
    S[i][i] += ridge;
  }
  return S;
}

function pooledCov(classes) {
  const S = [[0,0,0],[0,0,0],[0,0,0]];
  let totalN = 0;
  for (const c of classes) {
    for (const p of c.points) {
      const d = v3sub(p, c.muBar);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          S[i][j] += d[i]*d[j];
      totalN++;
    }
  }
  const denom = Math.max(1, totalN - classes.length);
  const ridge = 1e-5;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) S[i][j] /= denom;
    S[i][i] += ridge;
  }
  return S;
}

function inv3(M) {
  const [[a,b,c],[d,e,f],[g,h,k]] = M;
  const det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);
  if (Math.abs(det) < 1e-20) return null;
  const id = 1/det;
  return [
    [(e*k-f*h)*id, (c*h-b*k)*id, (b*f-c*e)*id],
    [(f*g-d*k)*id, (a*k-c*g)*id, (c*d-a*f)*id],
    [(d*h-e*g)*id, (b*g-a*h)*id, (a*e-b*d)*id]
  ];
}

function mahalDist2(x, mu, invCov) {
  const d = v3sub(x, mu);
  let result = 0;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      result += d[i] * invCov[i][j] * d[j];
  return result;
}

function quantile(arr, q) {
  const xs = [...arr].sort((a,b) => a-b);
  if (xs.length <= 1) return xs[0] || 0;
  const pos = (xs.length-1)*q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return xs[lo];
  return xs[lo]*(1-(pos-lo)) + xs[hi]*(pos-lo);
}

function softmax(z) {
  const m = Math.max(...z);
  const ex = z.map(v => Math.exp(v-m));
  const s = ex.reduce((a,b) => a+b, 0);
  return ex.map(v => v/s);
}

// ── 3x3 symmetric eigendecomposition (Cardano) ──
function eig3sym(A) {
  const a=A[0][0], b=A[0][1], c=A[0][2], d=A[1][1], e=A[1][2], f=A[2][2];
  const p1 = b*b + c*c + e*e;
  if (p1 < 1e-20) {
    return { values:[a,d,f], vectors:[[1,0,0],[0,1,0],[0,0,1]] };
  }
  const q = (a+d+f)/3;
  const p2 = (a-q)*(a-q) + (d-q)*(d-q) + (f-q)*(f-q) + 2*p1;
  const p = Math.sqrt(p2/6);
  const B = [[(a-q)/p, b/p, c/p],[b/p,(d-q)/p,e/p],[c/p,e/p,(f-q)/p]];
  const detB = B[0][0]*(B[1][1]*B[2][2]-B[1][2]*B[2][1])
             - B[0][1]*(B[1][0]*B[2][2]-B[1][2]*B[2][0])
             + B[0][2]*(B[1][0]*B[2][1]-B[1][1]*B[2][0]);
  let r = Math.max(-1, Math.min(1, detB/2));
  const phi = Math.acos(r)/3;
  const l1 = q + 2*p*Math.cos(phi);
  const l3 = q + 2*p*Math.cos(phi + 2*Math.PI/3);
  const l2 = 3*q - l1 - l3;
  const vals = [l1,l2,l3].sort((x,y) => y-x);

  function eigvec(lam) {
    const M = [[a-lam,b,c],[b,d-lam,e],[c,e,f-lam]];
    const r0 = M[0], r1 = M[1], r2 = M[2];
    let v = v3cross(r0, r1);
    if (v3len(v) < 1e-10) v = v3cross(r0, r2);
    if (v3len(v) < 1e-10) v = v3cross(r1, r2);
    if (v3len(v) < 1e-10) return [1,0,0];
    return v3norm(v);
  }
  const vecs = vals.map(l => eigvec(l));
  return { values: vals, vectors: vecs };
}

// ── fibonacci sphere grid ──
function fibGrid(n) {
  const pts = [];
  const phi = (1+Math.sqrt(5))/2;
  for (let i = 0; i < n; i++) {
    const y = 1 - (2*i+1)/n;
    const r = Math.sqrt(Math.max(0, 1-y*y));
    const theta = 2*Math.PI*i/phi;
    pts.push(v3norm([r*Math.cos(theta), y, r*Math.sin(theta)]));
  }
  return pts;
}

function logsumexp(logits) {
  const m = Math.max(...logits);
  return m + Math.log(logits.reduce((s, l) => s + Math.exp(l - m), 0));
}

function hexToRGB(hex) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return [r, g, b];
}

function dirFromUV(u, v) {
  // Match Three.js SphereGeometry UV convention
  const phi = u * 2 * Math.PI;
  const theta = v * Math.PI;
  const sinT = Math.sin(theta);
  return [-sinT * Math.cos(phi), Math.cos(theta), sinT * Math.sin(phi)];
}

// ================================================================
//  DATA GENERATION -- copied verbatim from HTML
// ================================================================
console.log("=== OOD 3D Precomputation ===\n");
console.log("Generating class data...");

const TAU = 5.0;
const Q = 0.05;

const classDefs = [
  { label:"A", color:"#534AB7", dir:v3norm([0.75,0.55,0.35]), kappa:80, n:200, aniso:false },
  { label:"B", color:"#1D9E75", dir:v3norm([-0.35,0.8,0.5]), kappa:30, n:80, aniso:false },
  { label:"C", color:"#D85A30", dir:v3norm([0.5,-0.65,0.55]), kappa:80, n:15, aniso:false },
  { label:"D", color:"#D4537E", dir:v3norm([-0.55,-0.35,0.75]), kappa:30, n:80, aniso:true, stretch:3.0 },
  { label:"E", color:"#378ADD", dir:v3norm([0.1,0.6,0.8]), kappa:30, n:60, aniso:false },
];

const classes = classDefs.map(def => {
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

console.log("  Classes generated: " + classes.map(c => c.label + " (n=" + c.n + ")").join(", "));

const poolCov = pooledCov(classes);
const poolInvCov = inv3(poolCov);

// MSP setup
const W = classes.map(c => c.muHat);
const trainMsp = classes.map((c, i) =>
  c.points.map(p => {
    const logits = W.map(w => TAU * v3dot(p, w));
    return softmax(logits)[i];
  })
);
const perClassGamma = trainMsp.map(xs => quantile(xs, Q));
const gamma = perClassGamma.reduce((a,b) => a+b, 0) / perClassGamma.length;

console.log("  MSP gamma = " + gamma.toFixed(6));

// OOD points
const grid4k = fibGrid(4000);
const mspScores4k = grid4k.map(p => {
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
const oodPoints = oodIdx.map(i => grid4k[i]);
console.log("  Far-OOD points: " + oodPoints.length);

// ── MLS ──
console.log("Computing MLS...");
const trainMLSscores = [];
for (const c of classes) {
  for (const p of c.points) {
    trainMLSscores.push(Math.max(...W.map(w => TAU * v3dot(p, w))));
  }
}
const mlsThreshold = quantile(trainMLSscores, Q);
console.log("  MLS threshold = " + mlsThreshold.toFixed(6));

// ── EBO ──
console.log("Computing EBO...");
const trainEBOscores = [];
for (const c of classes) {
  for (const p of c.points) {
    trainEBOscores.push(logsumexp(W.map(w => TAU * v3dot(p, w))));
  }
}
const eboThreshold = quantile(trainEBOscores, Q);
console.log("  EBO threshold = " + eboThreshold.toFixed(6));

// ── TempScale ──
console.log("Computing TempScale...");
let bestT = 1.0, bestNLL = Infinity;
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

const trainTSMSP = classes.map((c, i) =>
  c.points.map(p => softmax(W.map(w => (TAU / bestT) * v3dot(p, w)))[i])
);
const tsGamma = trainTSMSP.map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / classes.length;
console.log("  bestT = " + bestT.toFixed(4) + ", tsGamma = " + tsGamma.toFixed(6));

// ── KNN ──
console.log("Computing KNN...");
const allTrainPts = [];
const allTrainLabels = [];
for (let ci = 0; ci < classes.length; ci++) {
  for (const p of classes[ci].points) { allTrainPts.push(p); allTrainLabels.push(ci); }
}
const kVal = 5;

function knnResult(x, skipIdx) {
  if (skipIdx === undefined) skipIdx = -1;
  // Maintain top-k smallest distances without full sort
  const topK = new Array(kVal).fill(Infinity);
  const topKlabels = new Array(kVal).fill(0);
  let nearestDist = Infinity, nearestLabel = 0;
  for (let i = 0; i < allTrainPts.length; i++) {
    if (i === skipIdx) continue;
    const d = 1 - v3dot(x, allTrainPts[i]);
    if (d < nearestDist) { nearestDist = d; nearestLabel = allTrainLabels[i]; }
    if (d < topK[kVal-1]) {
      topK[kVal-1] = d;
      topKlabels[kVal-1] = allTrainLabels[i];
      // bubble down
      for (let j = kVal-1; j > 0 && topK[j] < topK[j-1]; j--) {
        const td = topK[j]; topK[j] = topK[j-1]; topK[j-1] = td;
        const tl = topKlabels[j]; topKlabels[j] = topKlabels[j-1]; topKlabels[j-1] = tl;
      }
    }
  }
  return { kthDist: topK[kVal-1], nearestLabel };
}

// Training threshold: 95th percentile of k-th neighbor distances (higher = OOD)
const trainKNNdists = allTrainPts.map((p, idx) => knnResult(p, idx).kthDist);
const knnThreshold = quantile(trainKNNdists, 1 - Q);
console.log("  KNN threshold = " + knnThreshold.toFixed(6));

// ── RMDS ──
console.log("Computing RMDS...");
const muAll = (() => {
  let m = [0,0,0];
  let n = 0;
  for (const c of classes) {
    for (const p of c.points) { m = v3add(m, p); n++; }
  }
  return v3scale(m, 1/n);
})();

function rmdsScore(x) {
  let minClassDist = Infinity;
  let bestClass = 0;
  for (let ci = 0; ci < classes.length; ci++) {
    const d2 = mahalDist2(x, classes[ci].muBar, poolInvCov);
    if (d2 < minClassDist) { minClassDist = d2; bestClass = ci; }
  }
  const bgDist = mahalDist2(x, muAll, poolInvCov);
  return { score: minClassDist - bgDist, bestClass };
}

const trainRMDS = allTrainPts.map(p => rmdsScore(p).score);
const rmdsThreshold = quantile(trainRMDS, 1 - Q);
console.log("  RMDS threshold = " + rmdsThreshold.toFixed(6));

// ── Mahalanobis thresholds ──
console.log("Computing Mahalanobis thresholds...");
const trainMahalPC = [];
for (let ci = 0; ci < classes.length; ci++) {
  for (const pt of classes[ci].points) {
    trainMahalPC.push(mahalDist2(pt, classes[ci].muBar, classes[ci].invCov));
  }
}
const mahalPCThr = quantile(trainMahalPC, 1 - Q);

const trainMahalS = [];
for (let ci = 0; ci < classes.length; ci++) {
  for (const pt of classes[ci].points) {
    trainMahalS.push(mahalDist2(pt, classes[ci].muBar, poolInvCov));
  }
}
const mahalSThr = quantile(trainMahalS, 1 - Q);
console.log("  mahalPCThr = " + mahalPCThr.toFixed(6) + ", mahalSThr = " + mahalSThr.toFixed(6));

// ================================================================
//  LEARNED FC -- copied verbatim from HTML
// ================================================================
console.log("Training FC (balanced)...");

const nClasses = classes.length;
const fcTrainX = [], fcTrainY = [];
for (let ci = 0; ci < nClasses; ci++) {
  for (const p of classes[ci].points) { fcTrainX.push(p); fcTrainY.push(ci); }
}
const fcN = fcTrainX.length;
const classWeights = classes.map(c => fcN / (nClasses * c.n));

function trainFC3D(balanced) {
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
        gW[ci][0] += cw * err * x[0] / fcN; gW[ci][1] += cw * err * x[1] / fcN; gW[ci][2] += cw * err * x[2] / fcN;
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
    quantile(c.points.map(p => softmax(logitsFn(p))[i]), Q)).reduce((a,b) => a+b, 0) / nClasses;
  const mlsThr = quantile(fcTrainX.map(p => Math.max(...logitsFn(p))), Q);
  const eboThr = quantile(fcTrainX.map(p => logsumexp(logitsFn(p))), Q);
  let bestT = 1.0, bestNLL = Infinity;
  for (let t = 0.2; t <= 10.0; t += 0.1) {
    let nll = 0;
    for (let si = 0; si < fcN; si++)
      nll -= Math.log(Math.max(softmax(logitsFn(fcTrainX[si]).map(l => l/t))[fcTrainY[si]], 1e-15));
    if (nll < bestNLL) { bestNLL = nll; bestT = t; }
  }
  const tsGamma = classes.map((c, i) =>
    quantile(c.points.map(p => softmax(logitsFn(p).map(l => l/bestT))[i]), Q)).reduce((a,b) => a+b, 0) / nClasses;
  // Cosine/vMF using learned weight directions as prototypes
  const wDirs = W.map(w => v3norm(w));
  const fcCosThresholds = classes.map((c, ci) => {
    const scores = c.points.map(p => v3dot(p, wDirs[ci]));
    return quantile(scores, Q);
  });
  const fcKappas = classes.map((c, ci) => {
    const scores = c.points.map(p => v3dot(p, wDirs[ci]));
    const Rbar = scores.reduce((a, b) => a + b, 0) / scores.length;
    return kappaFromRbar3D(Math.max(0, Math.min(0.999, Rbar)));
  });
  const fcVmBands = classes.map((c, ci) => {
    const scores = c.points.map(p => fcKappas[ci] * v3dot(p, wDirs[ci]));
    return [0.05, 0.20, 0.50].map(q => quantile(scores, q));
  });
  const fcKents = classes.map((c, ci) => fitKent(c.points, wDirs[ci]));
  return { W, B, logitsFn, gamma, mlsThr, eboThr, bestT, tsGamma,
           wNorms: W.map(w => v3len(w)), wDirs, fcCosThresholds, fcKappas, fcVmBands, fcKents };
}

const fcBal = trainFC3D(true);
console.log("  FC balanced done. gamma=" + fcBal.gamma.toFixed(6) + " bestT=" + fcBal.bestT.toFixed(4));

console.log("Training FC (unbalanced)...");
const fcUnbal = trainFC3D(false);
console.log("  FC unbalanced done. gamma=" + fcUnbal.gamma.toFixed(6) + " bestT=" + fcUnbal.bestT.toFixed(4));

// ── ODIN ──
console.log("Computing ODIN...");
const odinT = 1000, odinEps = 0.002;
const protoOdinWeights = W.map(w => v3scale(w, TAU));

function protoGetLogits(x) { return W.map(w => TAU * v3dot(x, w)); }

function odinScore(x, getLogits, weights, T, eps) {
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
  const xp = v3norm([x[0] + eps * Math.sign(grad[0]), x[1] + eps * Math.sign(grad[1]), x[2] + eps * Math.sign(grad[2])]);
  const logits2 = getLogits(xp);
  const probs2 = softmax(logits2.map(l => l / T));
  let argmax2 = 0;
  for (let i = 1; i < probs2.length; i++) if (probs2[i] > probs2[argmax2]) argmax2 = i;
  return { score: probs2[argmax2], argmax: argmax2 };
}

// Per-class ODIN training scores (prototype)
const trainOdinPerClass = classes.map((c, ci) =>
  c.points.map(p => {
    const r = odinScore(p, protoGetLogits, protoOdinWeights, odinT, odinEps);
    return r.argmax === ci ? r.score : 0;
  })
);
const odinGamma = trainOdinPerClass.map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / classes.length;
console.log("  ODIN gamma = " + odinGamma.toFixed(6));

// FC ODIN training scores (for both balanced and unbalanced)
function fcOdinScores(fcVariant) {
  const fcGetLogits = (x) => { const l = []; for (let ci = 0; ci < nClasses; ci++) l.push(v3dot(fcVariant.W[ci], x) + fcVariant.B[ci]); return l; };
  return classes.map((c, ci) =>
    c.points.map(p => {
      const r = odinScore(p, fcGetLogits, fcVariant.W, odinT, odinEps);
      return r.argmax === ci ? r.score : 0;
    })
  );
}
const fcBalOdinPerClass = fcOdinScores(fcBal);
const fcBalOdinGamma = fcBalOdinPerClass.map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / nClasses;
const fcUnbalOdinPerClass = fcOdinScores(fcUnbal);
const fcUnbalOdinGamma = fcUnbalOdinPerClass.map(xs => quantile(xs, Q)).reduce((a,b) => a+b, 0) / nClasses;
console.log("  FC bal ODIN gamma = " + fcBalOdinGamma.toFixed(6) + ", FC unbal ODIN gamma = " + fcUnbalOdinGamma.toFixed(6));

// ── ViM: PCA of training features ──
console.log("Computing ViM...");
const vimCov = covMatrix3(allTrainPts, muAll);
const vimEig = eig3sym(vimCov);
const vimU = [vimEig.vectors[0], vimEig.vectors[1]];

function vimResidual(x) {
  const xc = v3sub(x, muAll);
  let proj = [0, 0, 0];
  for (const u of vimU) {
    const coeff = v3dot(xc, u);
    proj = v3add(proj, v3scale(u, coeff));
  }
  return v3sub(xc, proj);
}

const trainMaxLogits = allTrainPts.map(p => Math.max(...W.map(w => TAU * v3dot(p, w))));
const trainResNorms = allTrainPts.map(p => v3len(vimResidual(p)));
const vimMeanMaxLogit = trainMaxLogits.reduce((a,b) => a+b, 0) / trainMaxLogits.length;
const vimMeanResNorm = trainResNorms.reduce((a,b) => a+b, 0) / trainResNorms.length;
const vimAlpha = vimMeanResNorm > 1e-10 ? vimMeanMaxLogit / vimMeanResNorm : 0;

function vimScoreFn(x, getLogits) {
  const logits = getLogits(x);
  const res = vimResidual(x);
  const vLogit = vimAlpha * v3len(res);
  const augmented = [...logits, vLogit];
  const m = Math.max(...augmented);
  const energy = m + Math.log(augmented.reduce((s, l) => s + Math.exp(l - m), 0));
  let argmax = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
  return { energy, argmax };
}

// ViM training scores (prototype)
const trainVimScores = allTrainPts.map(p => vimScoreFn(p, protoGetLogits).energy);
const vimThreshold = quantile(trainVimScores, Q);
console.log("  ViM alpha = " + vimAlpha.toFixed(6) + ", threshold = " + vimThreshold.toFixed(6));

// FC ViM (both variants)
function fcVimScores(fcVariant) {
  const fcGetLogits = (x) => { const l = []; for (let ci = 0; ci < nClasses; ci++) l.push(v3dot(fcVariant.W[ci], x) + fcVariant.B[ci]); return l; };
  const fcMaxLogits = allTrainPts.map(p => Math.max(...fcGetLogits(p)));
  const fcMML = fcMaxLogits.reduce((a,b) => a+b, 0) / fcMaxLogits.length;
  const fcVimAlpha = vimMeanResNorm > 1e-10 ? fcMML / vimMeanResNorm : 0;
  const scores = allTrainPts.map(p => {
    const logits = fcGetLogits(p);
    const res = vimResidual(p);
    const vLogit = fcVimAlpha * v3len(res);
    const aug = [...logits, vLogit];
    const mx = Math.max(...aug);
    return mx + Math.log(aug.reduce((s, l) => s + Math.exp(l - mx), 0));
  });
  return { scores, alpha: fcVimAlpha };
}
const fcBalVim = fcVimScores(fcBal);
const fcUnbalVim = fcVimScores(fcUnbal);
console.log("  FC bal ViM alpha = " + fcBalVim.alpha.toFixed(6) + ", FC unbal ViM alpha = " + fcUnbalVim.alpha.toFixed(6));

// ── KDE ──
console.log("Computing KDE...");
const kdeBandwidth = quantile(classes.map(c => c.kappa), 0.5);
function kdeScore(x) {
  // Returns log-density via logsumexp (matches HTML)
  let bestLogD = -Infinity, bestClass = 0;
  for (let ci = 0; ci < classes.length; ci++) {
    const dots = classes[ci].points.map(p => kdeBandwidth * v3dot(x, p));
    const mx = Math.max(...dots);
    const logD = mx + Math.log(dots.reduce((s, d) => s + Math.exp(d - mx), 0)) - Math.log(classes[ci].n);
    if (logD > bestLogD) { bestLogD = logD; bestClass = ci; }
  }
  return { density: bestLogD, bestClass };
}
const trainKDEscores = allTrainPts.map(p => kdeScore(p).density);
const kdeThreshold = quantile(trainKDEscores, Q);
console.log("  KDE bandwidth = " + kdeBandwidth.toFixed(6) + ", threshold = " + kdeThreshold.toFixed(6));

// ── allThresholds function ──
// We need fc reference for this computation -- use fcBal as default (same as HTML's initial state)
let fc = fcBal;

function allThresholds(q) {
  const mspGamma = trainMsp.map(xs => quantile(xs, q)).reduce((a,b) => a+b, 0) / classes.length;
  const mlsThr = quantile(trainMLSscores, q);
  const eboThr = quantile(trainEBOscores, q);
  const perTSMSP = trainTSMSP.map(xs => quantile(xs, q));
  const tsGam = perTSMSP.reduce((a,b) => a+b, 0) / classes.length;
  const cosPC = classes.map(c => quantile(c.cosScores, q));
  const vmBds = classes.map(c => [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantile(c.vmScores, bq)));
  const kentBds = classes.map(c => [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantile(c.kentScores, bq)));
  const mpcThr = quantile(trainMahalPC, 1 - q);
  const msThr = quantile(trainMahalS, 1 - q);
  const kThr = quantile(trainKNNdists, 1 - q);
  const rThr = quantile(trainRMDS, 1 - q);
  // FC thresholds
  const fcMspGam = classes.map((c, i) => quantile(c.points.map(p => softmax(fc.logitsFn(p))[i]), q)).reduce((a,b) => a+b, 0) / nClasses;
  const fcMlsThr2 = quantile(fcTrainX.map(p => Math.max(...fc.logitsFn(p))), q);
  const fcEboThr2 = quantile(fcTrainX.map(p => logsumexp(fc.logitsFn(p))), q);
  const fcTsGam = classes.map((c, i) => quantile(c.points.map(p => softmax(fc.logitsFn(p).map(l => l / fc.bestT))[i]), q)).reduce((a,b) => a+b, 0) / nClasses;
  const fcCosPC = classes.map((c, ci) => quantile(c.points.map(p => v3dot(p, fc.wDirs[ci])), q));
  const fcVmBds = classes.map((c, ci) => {
    const scores = c.points.map(p => fc.fcKappas[ci] * v3dot(p, fc.wDirs[ci]));
    return [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantile(scores, bq));
  });
  const fcKentBds = classes.map((c, ci) => {
    const scores = c.points.map(p => fc.fcKents[ci].kentScore(p));
    return [q, Math.max(q, 0.20), Math.max(q, 0.50)].map(bq => quantile(scores, bq));
  });
  // ODIN thresholds
  const odinGam = trainOdinPerClass.map(xs => quantile(xs, q)).reduce((a,b) => a+b, 0) / classes.length;
  const fcOdinGam = fcBalOdinPerClass.map(xs => quantile(xs, q)).reduce((a,b) => a+b, 0) / nClasses;
  // ViM thresholds
  const vimThr = quantile(trainVimScores, q);
  const fcVimThr = quantile(fcBalVim.scores, q);
  // KDE threshold
  const kdeThr = quantile(trainKDEscores, q);
  return { mspGamma: mspGamma, mlsThr, eboThr, tsGamma: tsGam, cosPerClass: cosPC,
           vmfBands: vmBds, kentBands: kentBds, mahalPCThr: mpcThr, mahalSThr: msThr,
           knnThr: kThr, rmdsThr: rThr, kdeThr,
           odinGamma: odinGam, fcOdinGamma: fcOdinGam, vimThr, fcVimThr,
           fcMspGamma: fcMspGam, fcMlsThr: fcMlsThr2, fcEboThr: fcEboThr2, fcTsGamma: fcTsGam,
           fcCosPerClass: fcCosPC, fcVmfBands: fcVmBds, fcKentBands: fcKentBds };
}

console.log("Computing cached thresholds (q=0.05)...");
const cachedThr = allThresholds(Q);

// ── Near-OOD points ──
console.log("Computing near-OOD points...");

// Need oodDecision3d for near-OOD computation
function oodDecision3d(x, method) {
  const logits = W.map(w => TAU * v3dot(x, w));
  const probs = softmax(logits);
  let pArgmax = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[pArgmax]) pArgmax = i;
  const thr = cachedThr;

  switch (method) {
    case "mds": {
      let minD = Infinity, best = 0;
      for (let ci = 0; ci < classes.length; ci++) {
        const d = mahalDist2(x, classes[ci].muBar, classes[ci].invCov);
        if (d < minD) { minD = d; best = ci; }
      }
      return { score: "d2_M", val: minD, thr: thr.mahalPCThr, lo: true, cls: best };
    }
    case "mds-s": {
      let minD = Infinity, best = 0;
      for (let ci = 0; ci < classes.length; ci++) {
        const d = mahalDist2(x, classes[ci].muBar, poolInvCov);
        if (d < minD) { minD = d; best = ci; }
      }
      return { score: "d2_M", val: minD, thr: thr.mahalSThr, lo: true, cls: best };
    }
    case "rmds": {
      const r = rmdsScore(x);
      return { score: "rel d2", val: r.score, thr: thr.rmdsThr, lo: true, cls: r.bestClass };
    }
    case "knn": {
      const r = knnResult(x);
      return { score: "k-th dist", val: r.kthDist, thr: thr.knnThr, lo: true, cls: r.nearestLabel };
    }
    case "cos": {
      let best = -1, bestS = -Infinity;
      for (let ci = 0; ci < classes.length; ci++) {
        const s = v3dot(x, classes[ci].muHat);
        if (s >= thr.cosPerClass[ci] && s > bestS) { best = ci; bestS = s; }
      }
      let nearCls = 0, maxCos = -Infinity;
      for (let ci = 0; ci < classes.length; ci++) {
        const s = v3dot(x, classes[ci].muHat);
        if (s > maxCos) { maxCos = s; nearCls = ci; }
      }
      return { score: "max cos", val: maxCos, thr: thr.cosPerClass[nearCls], lo: false, cls: nearCls, accepted: best >= 0, acceptCls: best };
    }
    case "msp":
      return { score: "max P", val: probs[pArgmax], thr: thr.mspGamma, lo: false, cls: pArgmax };
    case "mls": {
      const maxL = Math.max(...logits);
      return { score: "max logit", val: maxL, thr: thr.mlsThr, lo: false, cls: pArgmax };
    }
    case "ebo": {
      const e = logsumexp(logits);
      return { score: "energy", val: e, thr: thr.eboThr, lo: false, cls: pArgmax };
    }
    default: return null;
  }
}

// Compute near-OOD: inter-class boundary candidates + grid disagreement search
const nearOodMethods = ["msp", "cos", "mds", "rmds"];
const NEAR_MIN_SEP = 0.35;

// Generate inter-class boundary candidates along geodesics between class pairs
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

function scorePt(pt) {
  const decisions = [];
  for (const m of nearOodMethods) {
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
  const maxCos = Math.max(...classes.map(c => v3dot(pt, c.muHat)));
  return disagree * Math.max(0, maxCos);
}

const candidates = [];
for (let gi = 0; gi < grid4k.length; gi++) {
  const pt = grid4k[gi];
  if (!oodPoints.every(fp => (1 - v3dot(fp, pt)) > NEAR_MIN_SEP)) continue;
  const s = scorePt(pt);
  if (s > 0) candidates.push({ pt, score: s });
}
for (const pt of boundaryCandidates) {
  if (!oodPoints.every(fp => (1 - v3dot(fp, pt)) > NEAR_MIN_SEP)) continue;
  const s = scorePt(pt);
  if (s > 0) candidates.push({ pt, score: s });
}
candidates.sort((a, b) => b.score - a.score);

const nearOodSelected = [];
for (const cand of candidates) {
  if (nearOodSelected.every(sp => (1 - v3dot(sp, cand.pt)) > NEAR_MIN_SEP)) {
    nearOodSelected.push(cand.pt);
    if (nearOodSelected.length >= 4) break;
  }
}
const nearOodPoints = nearOodSelected;
console.log("  Near-OOD points: " + nearOodPoints.length);

// ================================================================
//  TEXTURE GENERATION
// ================================================================
const TEX_W = 2048, TEX_H = 1024;
const KNN_W = 1024, KNN_H = 512;

const texDir = path.join(__dirname, "tex");
if (!fs.existsSync(texDir)) fs.mkdirSync(texDir);

function generateTexture(name, w, h, evalFn) {
  const t1 = Date.now();
  process.stdout.write("  Generating " + name + " (" + w + "x" + h + ")...");
  const buf = Buffer.alloc(w * h * 4);
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const u = px / w;
      const v = py / h;
      const dir = dirFromUV(u, v);
      const result = evalFn(dir);
      if (result) {
        const idx = (py * w + px) * 4;
        const [r, g, b] = hexToRGB(result.color);
        buf[idx]   = r;
        buf[idx+1] = g;
        buf[idx+2] = b;
        buf[idx+3] = Math.round((result.opacity || 0.55) * 255);
      }
    }
  }
  const elapsed = ((Date.now() - t1) / 1000).toFixed(1);
  process.stdout.write(" done (" + elapsed + "s)\n");
  return buf;
}

// Helper for banded textures (vMF, Kent, FC-vMF, FC-Kent)
function generateBandedTexture(name, w, h, bandEvalFn) {
  const t1 = Date.now();
  process.stdout.write("  Generating " + name + " (" + w + "x" + h + ")...");
  const bandOpacities = [0.18, 0.35, 0.55];
  const buf = Buffer.alloc(w * h * 4);
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const u = px / w;
      const v = py / h;
      const dir = dirFromUV(u, v);
      const idx = (py * w + px) * 4;

      let bestColor = null;
      let bestOpacity = 0;

      for (let band = 0; band < 3; band++) {
        const result = bandEvalFn(dir, band);
        if (result) {
          bestColor = result.color;
          bestOpacity = bandOpacities[band];
        }
      }

      if (bestColor) {
        const [r, g, b] = hexToRGB(bestColor);
        buf[idx]   = r;
        buf[idx+1] = g;
        buf[idx+2] = b;
        buf[idx+3] = Math.round(bestOpacity * 255);
      }
    }
  }
  const elapsed = ((Date.now() - t1) / 1000).toFixed(1);
  process.stdout.write(" done (" + elapsed + "s)\n");
  return buf;
}

console.log("\nGenerating prototype textures...");

// MSP
const mspBuf = generateTexture("msp", TEX_W, TEX_H, (p) => {
  const logits = W.map(w => TAU * v3dot(p, w));
  const probs = softmax(logits);
  let argmax = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[argmax]) argmax = i;
  if (probs[argmax] >= gamma) return { color: classes[argmax].color, opacity: 0.55 };
  return null;
});
writePNG(path.join(texDir, "msp.png"), TEX_W, TEX_H, mspBuf);

// Cosine
const cosBuf = generateTexture("cos", TEX_W, TEX_H, (p) => {
  let best = null, bestScore = -Infinity;
  for (const c of classes) {
    const s = v3dot(p, c.muHat);
    if (s >= c.cosThreshold && s > bestScore) { best = c; bestScore = s; }
  }
  return best ? { color: best.color, opacity: 0.55 } : null;
});
writePNG(path.join(texDir, "cos.png"), TEX_W, TEX_H, cosBuf);

// vMF (banded)
const vmfBuf = generateBandedTexture("vmf", TEX_W, TEX_H, (dir, band) => {
  let best = null, bestScore = -Infinity;
  for (const c of classes) {
    const s = c.kappa * v3dot(dir, c.muHat);
    if (s >= c.vmBands[band] && s > bestScore) { best = c; bestScore = s; }
  }
  return best ? { color: best.color } : null;
});
writePNG(path.join(texDir, "vmf.png"), TEX_W, TEX_H, vmfBuf);

// Kent (banded)
const kentBuf = generateBandedTexture("kent", TEX_W, TEX_H, (dir, band) => {
  let best = null, bestScore = -Infinity;
  for (const c of classes) {
    const s = c.kent.kentScore(dir);
    if (s >= c.kent.bands[band] && s > bestScore) { best = c; bestScore = s; }
  }
  return best ? { color: best.color } : null;
});
writePNG(path.join(texDir, "kent.png"), TEX_W, TEX_H, kentBuf);

// MLS
const mlsBuf = generateTexture("mls", TEX_W, TEX_H, (p) => {
  const logits = W.map(w => TAU * v3dot(p, w));
  const maxL = Math.max(...logits);
  if (maxL >= mlsThreshold) {
    let argmax = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
    return { color: classes[argmax].color, opacity: 0.55 };
  }
  return null;
});
writePNG(path.join(texDir, "mls.png"), TEX_W, TEX_H, mlsBuf);

// EBO
const eboBuf = generateTexture("ebo", TEX_W, TEX_H, (p) => {
  const logits = W.map(w => TAU * v3dot(p, w));
  const energy = logsumexp(logits);
  if (energy >= eboThreshold) {
    let argmax = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
    return { color: classes[argmax].color, opacity: 0.55 };
  }
  return null;
});
writePNG(path.join(texDir, "ebo.png"), TEX_W, TEX_H, eboBuf);

// TempScale
const tsBuf = generateTexture("ts", TEX_W, TEX_H, (p) => {
  const logits = W.map(w => (TAU / bestT) * v3dot(p, w));
  const probs = softmax(logits);
  let argmax = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[argmax]) argmax = i;
  if (probs[argmax] >= tsGamma) return { color: classes[argmax].color, opacity: 0.55 };
  return null;
});
writePNG(path.join(texDir, "ts.png"), TEX_W, TEX_H, tsBuf);

// KNN (reduced resolution)
console.log("\nGenerating KNN texture (reduced resolution)...");
const knnBuf = generateTexture("knn", KNN_W, KNN_H, (p) => {
  const result = knnResult(p);
  if (result.kthDist <= knnThreshold) {
    return { color: classes[result.nearestLabel].color, opacity: 0.55 };
  }
  return null;
});
writePNG(path.join(texDir, "knn.png"), KNN_W, KNN_H, knnBuf);

// RMDS
const rmdsBuf = generateTexture("rmds", TEX_W, TEX_H, (p) => {
  const result = rmdsScore(p);
  if (result.score <= rmdsThreshold) {
    return { color: classes[result.bestClass].color, opacity: 0.55 };
  }
  return null;
});
writePNG(path.join(texDir, "rmds.png"), TEX_W, TEX_H, rmdsBuf);

// ================================================================
//  FC TEXTURES (both balanced and unbalanced)
// ================================================================
function generateFCTextures(variant, suffix) {
  console.log("\nGenerating FC textures (" + suffix + ")...");
  const fn = variant.logitsFn;

  // FC-MSP
  const buf1 = generateTexture("fc_msp_" + suffix, TEX_W, TEX_H, (p) => {
    const l = fn(p), pr = softmax(l); let a = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[a]) a = i;
    return pr[a] >= variant.gamma ? { color: classes[a].color, opacity: 0.55 } : null;
  });
  writePNG(path.join(texDir, "fc_msp_" + suffix + ".png"), TEX_W, TEX_H, buf1);

  // FC-MLS
  const buf2 = generateTexture("fc_mls_" + suffix, TEX_W, TEX_H, (p) => {
    const l = fn(p), mx = Math.max(...l);
    if (mx >= variant.mlsThr) { let a = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[a]) a = i;
      return { color: classes[a].color, opacity: 0.55 }; } return null;
  });
  writePNG(path.join(texDir, "fc_mls_" + suffix + ".png"), TEX_W, TEX_H, buf2);

  // FC-EBO
  const buf3 = generateTexture("fc_ebo_" + suffix, TEX_W, TEX_H, (p) => {
    const l = fn(p);
    if (logsumexp(l) >= variant.eboThr) { let a = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[a]) a = i;
      return { color: classes[a].color, opacity: 0.55 }; } return null;
  });
  writePNG(path.join(texDir, "fc_ebo_" + suffix + ".png"), TEX_W, TEX_H, buf3);

  // FC-TS
  const buf4 = generateTexture("fc_ts_" + suffix, TEX_W, TEX_H, (p) => {
    const l = fn(p), pr = softmax(l.map(v => v / variant.bestT)); let a = 0;
    for (let i = 1; i < pr.length; i++) if (pr[i] > pr[a]) a = i;
    return pr[a] >= variant.tsGamma ? { color: classes[a].color, opacity: 0.55 } : null;
  });
  writePNG(path.join(texDir, "fc_ts_" + suffix + ".png"), TEX_W, TEX_H, buf4);

  // FC-Cosine
  const buf5 = generateTexture("fc_cos_" + suffix, TEX_W, TEX_H, (p) => {
    let best = null, bestScore = -Infinity;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = v3dot(p, variant.wDirs[ci]);
      if (s >= variant.fcCosThresholds[ci] && s > bestScore) { best = classes[ci]; bestScore = s; }
    }
    return best ? { color: best.color, opacity: 0.55 } : null;
  });
  writePNG(path.join(texDir, "fc_cos_" + suffix + ".png"), TEX_W, TEX_H, buf5);

  // FC-vMF (banded)
  const buf6 = generateBandedTexture("fc_vmf_" + suffix, TEX_W, TEX_H, (dir, band) => {
    let best = null, bestScore = -Infinity;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = variant.fcKappas[ci] * v3dot(dir, variant.wDirs[ci]);
      if (s >= variant.fcVmBands[ci][band] && s > bestScore) { best = classes[ci]; bestScore = s; }
    }
    return best ? { color: best.color } : null;
  });
  writePNG(path.join(texDir, "fc_vmf_" + suffix + ".png"), TEX_W, TEX_H, buf6);

  // FC-Kent (banded)
  const buf7 = generateBandedTexture("fc_kent_" + suffix, TEX_W, TEX_H, (dir, band) => {
    let best = null, bestScore = -Infinity;
    for (let ci = 0; ci < classes.length; ci++) {
      const s = variant.fcKents[ci].kentScore(dir);
      if (s >= variant.fcKents[ci].bands[band] && s > bestScore) { best = classes[ci]; bestScore = s; }
    }
    return best ? { color: best.color } : null;
  });
  writePNG(path.join(texDir, "fc_kent_" + suffix + ".png"), TEX_W, TEX_H, buf7);
}

generateFCTextures(fcBal, "bal");
generateFCTextures(fcUnbal, "unbal");

// ================================================================
//  SAVE precomputed.js
// ================================================================
console.log("\nBuilding precomputed.js...");

// Rounding helpers
function r6(v) { return parseFloat(v.toFixed(6)); }
function r4(v) { return parseFloat(v.toFixed(4)); }
function rv3(a) { return a.map(v => r6(v)); }
function rmat(m) { return m.map(row => row.map(v => r6(v))); }

function serializeKent(kent) {
  return {
    kappa: r6(kent.kappa),
    beta: r6(kent.beta),
    gamma1: rv3(kent.gamma1),
    gamma2: rv3(kent.gamma2),
    gamma3: rv3(kent.gamma3),
    bands: kent.bands.map(b => r4(b))
  };
}

function serializeFC(variant, odinPerClass, odinGammaVal, vimData) {
  return {
    W: variant.W.map(w => rv3(w)),
    B: variant.B.map(b => r6(b)),
    wNorms: variant.wNorms.map(n => r6(n)),
    wDirs: variant.wDirs.map(d => rv3(d)),
    gamma: r6(variant.gamma),
    mlsThr: r6(variant.mlsThr),
    eboThr: r6(variant.eboThr),
    bestT: r6(variant.bestT),
    tsGamma: r6(variant.tsGamma),
    fcCosThresholds: variant.fcCosThresholds.map(t => r6(t)),
    fcKappas: variant.fcKappas.map(k => r6(k)),
    fcVmBands: variant.fcVmBands.map(b => b.map(v => r4(v))),
    fcKents: variant.fcKents.map(k => serializeKent(k)),
    odinPerClass: odinPerClass.map(xs => xs.map(v => r6(v))),
    odinGamma: r6(odinGammaVal),
    vimAlpha: r6(vimData.alpha),
    trainVimScores: vimData.scores.map(v => r6(v)),
    vimThreshold: r6(quantile(vimData.scores, Q))
  };
}

const data = {
  // Class data
  classes: classes.map(c => ({
    label: c.label,
    color: c.color,
    dir: rv3(classDefs.find(d => d.label === c.label).dir),
    generatingKappa: classDefs.find(d => d.label === c.label).kappa,
    n: c.n,
    aniso: c.aniso,
    stretch: c.stretch || null,
    points: c.points.map(p => rv3(p)),
    muBar: rv3(c.muBar),
    muHat: rv3(c.muHat),
    Rbar: r6(c.Rbar),
    kappa: r6(c.kappa),
    cosScores: c.cosScores.map(s => r4(s)),
    vmScores: c.vmScores.map(s => r4(s)),
    kentScores: c.kentScores.map(s => r4(s)),
    cosThreshold: r6(c.cosThreshold),
    vmBands: c.vmBands.map(b => r4(b)),
    cov: rmat(c.cov),
    invCov: rmat(c.invCov),
    kent: serializeKent(c.kent)
  })),

  // Pooled covariance
  poolCov: rmat(poolCov),
  poolInvCov: rmat(poolInvCov),

  // Prototypes
  W: W.map(w => rv3(w)),

  // Training scores
  trainMsp: trainMsp.map(xs => xs.map(v => r4(v))),
  gamma: r6(gamma),
  trainMLSscores: trainMLSscores.map(v => r4(v)),
  mlsThreshold: r6(mlsThreshold),
  trainEBOscores: trainEBOscores.map(v => r4(v)),
  eboThreshold: r6(eboThreshold),
  bestT: r6(bestT),
  trainTSMSP: trainTSMSP.map(xs => xs.map(v => r4(v))),
  tsGamma: r6(tsGamma),

  // KNN
  allTrainPts: allTrainPts.map(p => rv3(p)),
  allTrainLabels: allTrainLabels,
  kVal: kVal,
  trainKNNdists: trainKNNdists.map(d => r6(d)),
  knnThreshold: r6(knnThreshold),

  // RMDS
  muAll: rv3(muAll),
  trainRMDS: trainRMDS.map(s => r4(s)),
  rmdsThreshold: r6(rmdsThreshold),

  // Mahalanobis
  trainMahalPC: trainMahalPC.map(v => r4(v)),
  mahalPCThr: r6(mahalPCThr),
  trainMahalS: trainMahalS.map(v => r4(v)),
  mahalSThr: r6(mahalSThr),

  // ODIN
  odinT: 1000,
  odinEps: 0.002,
  trainOdinPerClass: trainOdinPerClass.map(xs => xs.map(v => r6(v))),
  odinGamma: r6(odinGamma),

  // ViM
  vimU: vimU.map(u => rv3(u)),
  vimAlpha: r6(vimAlpha),
  trainVimScores: trainVimScores.map(v => r6(v)),
  vimThreshold: r6(vimThreshold),

  // KDE
  kdeBandwidth: r6(kdeBandwidth),
  trainKDEscores: trainKDEscores.map(v => r6(v)),
  kdeThreshold: r6(kdeThreshold),

  // FC (both variants)
  fcBal: serializeFC(fcBal, fcBalOdinPerClass, fcBalOdinGamma, fcBalVim),
  fcUnbal: serializeFC(fcUnbal, fcUnbalOdinPerClass, fcUnbalOdinGamma, fcUnbalVim),

  // OOD points
  oodPoints: oodPoints.map(p => rv3(p)),
  nearOodPoints: nearOodPoints.map(p => rv3(p)),

  // Cached thresholds at q=0.05
  cachedThr: {
    mspGamma: r6(cachedThr.mspGamma),
    mlsThr: r6(cachedThr.mlsThr),
    eboThr: r6(cachedThr.eboThr),
    tsGamma: r6(cachedThr.tsGamma),
    cosPerClass: cachedThr.cosPerClass.map(v => r6(v)),
    vmfBands: cachedThr.vmfBands.map(b => b.map(v => r4(v))),
    kentBands: cachedThr.kentBands.map(b => b.map(v => r4(v))),
    mahalPCThr: r6(cachedThr.mahalPCThr),
    mahalSThr: r6(cachedThr.mahalSThr),
    knnThr: r6(cachedThr.knnThr),
    rmdsThr: r6(cachedThr.rmdsThr),
    fcMspGamma: r6(cachedThr.fcMspGamma),
    fcMlsThr: r6(cachedThr.fcMlsThr),
    fcEboThr: r6(cachedThr.fcEboThr),
    fcTsGamma: r6(cachedThr.fcTsGamma),
    fcCosPerClass: cachedThr.fcCosPerClass.map(v => r6(v)),
    fcVmfBands: cachedThr.fcVmfBands.map(b => b.map(v => r4(v))),
    fcKentBands: cachedThr.fcKentBands.map(b => b.map(v => r4(v))),
    odinGamma: r6(cachedThr.odinGamma),
    fcOdinGamma: r6(cachedThr.fcOdinGamma),
    vimThr: r6(cachedThr.vimThr),
    fcVimThr: r6(cachedThr.fcVimThr),
    kdeThr: r6(cachedThr.kdeThr)
  },

  // Texture manifest
  textures: {
    msp:  { file: "tex/msp.png",  w: TEX_W, h: TEX_H },
    mls:  { file: "tex/mls.png",  w: TEX_W, h: TEX_H },
    ebo:  { file: "tex/ebo.png",  w: TEX_W, h: TEX_H },
    ts:   { file: "tex/ts.png",   w: TEX_W, h: TEX_H },
    cos:  { file: "tex/cos.png",  w: TEX_W, h: TEX_H },
    vmf:  { file: "tex/vmf.png",  w: TEX_W, h: TEX_H },
    kent: { file: "tex/kent.png", w: TEX_W, h: TEX_H },
    knn:  { file: "tex/knn.png",  w: KNN_W, h: KNN_H },
    rmds: { file: "tex/rmds.png", w: TEX_W, h: TEX_H },
    "fc-msp-bal":  { file: "tex/fc_msp_bal.png",  w: TEX_W, h: TEX_H },
    "fc-mls-bal":  { file: "tex/fc_mls_bal.png",  w: TEX_W, h: TEX_H },
    "fc-ebo-bal":  { file: "tex/fc_ebo_bal.png",  w: TEX_W, h: TEX_H },
    "fc-ts-bal":   { file: "tex/fc_ts_bal.png",   w: TEX_W, h: TEX_H },
    "fc-cos-bal":  { file: "tex/fc_cos_bal.png",  w: TEX_W, h: TEX_H },
    "fc-vmf-bal":  { file: "tex/fc_vmf_bal.png",  w: TEX_W, h: TEX_H },
    "fc-kent-bal": { file: "tex/fc_kent_bal.png", w: TEX_W, h: TEX_H },
    "fc-msp-unbal":  { file: "tex/fc_msp_unbal.png",  w: TEX_W, h: TEX_H },
    "fc-mls-unbal":  { file: "tex/fc_mls_unbal.png",  w: TEX_W, h: TEX_H },
    "fc-ebo-unbal":  { file: "tex/fc_ebo_unbal.png",  w: TEX_W, h: TEX_H },
    "fc-ts-unbal":   { file: "tex/fc_ts_unbal.png",   w: TEX_W, h: TEX_H },
    "fc-cos-unbal":  { file: "tex/fc_cos_unbal.png",  w: TEX_W, h: TEX_H },
    "fc-vmf-unbal":  { file: "tex/fc_vmf_unbal.png",  w: TEX_W, h: TEX_H },
    "fc-kent-unbal": { file: "tex/fc_kent_unbal.png", w: TEX_W, h: TEX_H }
  },

  // MSP scores on 4k grid (needed for near-OOD fallback)
  mspScores4k: mspScores4k.map(s => r4(s))
};

const jsContent = "window.OOD_PRECOMPUTED = " + JSON.stringify(data) + ";\n";
const outPath = path.join(__dirname, "precomputed.js");
fs.writeFileSync(outPath, jsContent);
const fileSizeKB = (fs.statSync(outPath).size / 1024).toFixed(0);
console.log("\n  precomputed.js (" + fileSizeKB + " KB)");

// ── Summary ──
const totalTime = ((Date.now() - t0) / 1000).toFixed(1);
console.log("\n=== Done in " + totalTime + "s ===");
console.log("Files created:");
console.log("  precomputed.js (" + fileSizeKB + " KB)");
const texFiles = fs.readdirSync(texDir).filter(f => f.endsWith(".png"));
texFiles.forEach(f => {
  const sz = (fs.statSync(path.join(texDir, f)).size / 1024).toFixed(0);
  console.log("  tex/" + f + " (" + sz + " KB)");
});
