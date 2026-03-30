import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { PRE } from '../load-oracle.js';

import { v3dot, v3norm, v3scale, v3len, v3sub, v3add } from '../../src/math/vec3.js';
import { softmax, logsumexp, quantile } from '../../src/math/stats.js';
import { mahalDist2, inv3, covMatrix3 } from '../../src/math/linalg.js';
import { fitKent, makeKentScoreFn } from '../../src/math/fitting.js';
import { getMethod, allKeys } from '../../src/config/method-registry.js';
import {
  odinScore, vimResidual, vimScoreFn, rmdsScore, knnResult, kdeScore,
  makeFCLogitsFn, perClassDecide,
} from '../../src/methods/scoring.js';

const TAU = 5.0;
const Q_VAL = 0.05;
const ODIN_T = 1000;
const ODIN_EPS = 0.002;
const TOL = 1e-6;

// ── Reconstruct runtime context from oracle ──

const classes = PRE.classes.map((pc) => {
  const kentObj = {
    kappa: pc.kent.kappa, beta: pc.kent.beta,
    gamma1: pc.kent.gamma1, gamma2: pc.kent.gamma2, gamma3: pc.kent.gamma3,
    bands: pc.kent.bands,
    kentScore: makeKentScoreFn(pc.kent),
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
    kent: kentObj, kentScores: pc.kentScores,
  };
});

const W = PRE.W;
const poolInvCov = PRE.poolInvCov;
const muAll = PRE.muAll;
const allTrainPts = PRE.allTrainPts;
const allTrainLabels = PRE.allTrainLabels;
const kVal = PRE.kVal;
const kdeBandwidth = quantile(classes.map(c => c.kappa), 0.5);
const bestT = PRE.bestT;
const vimU = PRE.vimU;
const vimAlpha = PRE.vimAlpha;

// Reconstruct FC variant
const fcBal = (() => {
  const pf = PRE.fcBal;
  const logitsFn = makeFCLogitsFn(pf.W, pf.B);
  const wDirs = pf.W.map(w => v3norm(w));
  const fcKappas = pf.W.map(w => v3len(w));
  const fcKents = classes.map((c, ci) => fitKent(c.points, wDirs[ci]));
  return {
    W: pf.W, B: pf.B, logitsFn, wDirs, fcKappas, fcKents,
    bestT: pf.bestT, gamma: pf.gamma,
    fcCosThresholds: pf.fcCosThresholds,
    fcVmBands: pf.fcVmBands,
  };
})();
const fcVimAlpha = PRE.fcBal.vimAlpha;

// Compute thresholds (simplified — use oracle values)
function buildThresholds() {
  const trainMsp = PRE.trainMsp;
  const perClassGamma = trainMsp.map(xs => quantile(xs, Q_VAL));
  const mspGamma = perClassGamma.reduce((a, b) => a + b, 0) / perClassGamma.length;

  return {
    mspGamma,
    mlsThr: PRE.mlsThreshold,
    eboThr: PRE.eboThreshold,
    tsGamma: PRE.tsGamma,
    odinGamma: PRE.trainOdinPerClass
      ? PRE.trainOdinPerClass.map(xs => quantile(xs, Q_VAL)).reduce((a, b) => a + b, 0) / classes.length
      : 0,
    vimThr: PRE.vimThreshold,
    cosPerClass: classes.map(c => c.cosThreshold),
    vmfBands: classes.map(c => c.vmBands),
    kentBands: classes.map(c => c.kent.bands),
    mahalPCThr: PRE.mahalPCThr || 0,
    mahalSThr: PRE.mahalSThr || 0,
    knnThr: PRE.knnThreshold,
    rmdsThr: PRE.rmdsThreshold,
    kdeThr: quantile(allTrainPts.map(p => kdeScore(p, classes, kdeBandwidth).density), Q_VAL),
    fcMspGamma: PRE.fcBal.gamma,
    fcMlsThr: PRE.fcBal.mlsThr,
    fcEboThr: PRE.fcBal.eboThr,
    fcTsGamma: PRE.fcBal.tsGamma,
    fcOdinGamma: PRE.fcBal.odinGamma,
    fcVimThr: PRE.fcBal.vimThreshold,
    fcCosPerClass: PRE.fcBal.fcCosThresholds,
    fcVmfBands: PRE.fcBal.fcVmBands,
    fcKentBands: classes.map((c, ci) => fcBal.fcKents[ci].bands),
  };
}

const thr = buildThresholds();

const ctx = {
  W, TAU, classes, poolInvCov, muAll,
  allTrainPts, allTrainLabels, kVal, kdeBandwidth,
  bestT, vimU, vimAlpha,
  fc: fcBal, fcVimAlpha,
  thr, odinT: ODIN_T, odinEps: ODIN_EPS,
};

// Test points: class centers + OOD points
const testPoints = [
  ...classes.map(c => c.muHat),
  ...PRE.oodPoints,
];

// ── Method keys that should have decideFn ──
const decideKeys = [
  "msp", "mls", "ebo", "ts", "odin", "vim",
  "fc-msp", "fc-mls", "fc-ebo", "fc-ts", "fc-odin", "fc-vim",
  "cos", "vmf", "kent", "fc-cos", "fc-vmf", "fc-kent",
  "mds", "mds-s", "rmds", "knn", "kde",
];

// Keys with classScoresFn
const classScoreKeys = [
  "msp", "ts", "odin", "fc-msp", "fc-ts", "fc-odin",
  "vim", "fc-vim", "mls", "fc-mls",
  "cos", "fc-cos", "vmf", "fc-vmf", "kent", "fc-kent",
  "kde", "mds", "mds-s",
];

// Keys with scoreFn
const scoreKeys = [
  "msp", "mls", "ebo", "ts", "odin", "vim",
  "fc-msp", "fc-mls", "fc-ebo", "fc-ts", "fc-odin", "fc-vim",
  "knn", "kde",
];

describe('Registry dispatch: decideFn', () => {
  it('every decide key has a decideFn on the registry entry', () => {
    for (const key of decideKeys) {
      const m = getMethod(key);
      assert.ok(m, `method "${key}" should be in registry`);
      assert.ok(typeof m.decideFn === 'function', `"${key}" should have decideFn`);
    }
  });

  for (const key of decideKeys) {
    it(`"${key}" decideFn returns valid structure`, () => {
      const m = getMethod(key);
      for (const pt of testPoints) {
        const d = m.decideFn(pt, ctx);
        assert.ok(d, `decideFn for "${key}" should return non-null`);
        assert.ok(typeof d.score === 'string', `"${key}" .score should be string`);
        assert.ok(typeof d.val === 'number' && !isNaN(d.val), `"${key}" .val should be a number, got ${d.val}`);
        assert.ok(typeof d.thr === 'number', `"${key}" .thr should be a number`);
        assert.ok(typeof d.lo === 'boolean', `"${key}" .lo should be boolean`);
        assert.ok(typeof d.cls === 'number' && d.cls >= 0, `"${key}" .cls should be non-negative integer`);
      }
    });
  }

  it('MSP on class A center gives high probability', () => {
    const d = getMethod("msp").decideFn(classes[0].muHat, ctx);
    assert.ok(d.val > 0.5, `MSP on class A center should be >0.5, got ${d.val}`);
    assert.equal(d.cls, 0, 'MSP on class A center should predict class 0');
  });

  it('per-class methods return accepted/acceptCls fields', () => {
    for (const key of ["cos", "vmf", "kent", "fc-cos", "fc-vmf", "fc-kent"]) {
      const d = getMethod(key).decideFn(classes[0].muHat, ctx);
      assert.ok('accepted' in d, `"${key}" should have .accepted field`);
      assert.ok('acceptCls' in d, `"${key}" should have .acceptCls field`);
    }
  });

  it('MDS returns lo=true (lower is more in-distribution)', () => {
    const d = getMethod("mds").decideFn(classes[0].muHat, ctx);
    assert.equal(d.lo, true);
  });

  it('KDE returns lo=false (higher density is more in-distribution)', () => {
    const d = getMethod("kde").decideFn(classes[0].muHat, ctx);
    assert.equal(d.lo, false);
  });
});

describe('Registry dispatch: classScoresFn', () => {
  it('every classScore key has a classScoresFn', () => {
    for (const key of classScoreKeys) {
      const m = getMethod(key);
      assert.ok(m, `method "${key}" should be in registry`);
      assert.ok(typeof m.classScoresFn === 'function', `"${key}" should have classScoresFn`);
    }
  });

  for (const key of classScoreKeys) {
    it(`"${key}" classScoresFn returns valid structure`, () => {
      const m = getMethod(key);
      const pt = classes[0].muHat;
      const cs = m.classScoresFn(pt, ctx);
      assert.ok(cs, `classScoresFn for "${key}" should return non-null`);
      assert.ok(["prob", "logit", "ratio", "dist"].includes(cs.type),
        `"${key}" .type should be prob/logit/ratio/dist, got ${cs.type}`);
      assert.equal(cs.items.length, 5, `"${key}" should have 5 items (one per class)`);
      for (const item of cs.items) {
        assert.ok(typeof item.l === 'string', 'item.l should be string');
        assert.ok(typeof item.c === 'string', 'item.c should be string (color)');
        assert.ok(typeof item.v === 'number' && !isNaN(item.v), `item.v should be number, got ${item.v}`);
      }
    });
  }

  it('MSP classScoresFn probabilities sum to ~1', () => {
    const cs = getMethod("msp").classScoresFn(classes[0].muHat, ctx);
    const sum = cs.items.reduce((s, i) => s + i.v, 0);
    assert.ok(Math.abs(sum - 1.0) < 0.01, `MSP probs should sum to ~1, got ${sum}`);
  });

  it('methods without classScoresFn have null', () => {
    for (const key of ["ebo", "fc-ebo", "rmds", "knn"]) {
      const m = getMethod(key);
      assert.ok(m, `"${key}" should be in registry`);
      assert.ok(!m.classScoresFn || m.classScoresFn === null,
        `"${key}" should not have classScoresFn`);
    }
  });
});

describe('Registry dispatch: scoreFn', () => {
  it('every score key has a scoreFn', () => {
    for (const key of scoreKeys) {
      const m = getMethod(key);
      assert.ok(m, `method "${key}" should be in registry`);
      assert.ok(typeof m.scoreFn === 'function', `"${key}" should have scoreFn`);
    }
  });

  for (const key of scoreKeys) {
    it(`"${key}" scoreFn returns {classIdx, score}`, () => {
      const m = getMethod(key);
      const pt = classes[0].muHat;
      const s = m.scoreFn(pt, ctx);
      assert.ok(typeof s.classIdx === 'number', `"${key}" .classIdx should be number`);
      assert.ok(s.classIdx >= 0 && s.classIdx < 5, `"${key}" .classIdx should be 0-4`);
      assert.ok(typeof s.score === 'number' && !isNaN(s.score), `"${key}" .score should be number`);
    });
  }
});

describe('Scoring helpers: pure function parity', () => {
  it('odinScore matches expected signature', () => {
    const getLogits = (x) => W.map(w => TAU * v3dot(x, w));
    const weights = W.map(w => v3scale(w, TAU));
    const r = odinScore(classes[0].muHat, getLogits, weights, ODIN_T, ODIN_EPS);
    assert.ok(typeof r.score === 'number' && r.score > 0);
    assert.ok(typeof r.argmax === 'number' && r.argmax >= 0);
  });

  it('vimResidual is orthogonal to principal subspace', () => {
    const x = classes[0].muHat;
    const res = vimResidual(x, vimU, muAll);
    for (const u of vimU) {
      const dot = v3dot(res, u);
      assert.ok(Math.abs(dot) < TOL, `residual should be orthogonal to vimU, dot=${dot}`);
    }
  });

  it('knnResult returns kthDist and nearestLabel', () => {
    const r = knnResult(classes[0].muHat, allTrainPts, allTrainLabels, kVal);
    assert.ok(typeof r.kthDist === 'number' && r.kthDist >= 0);
    assert.ok(typeof r.nearestLabel === 'number');
    assert.equal(r.nearestLabel, 0, 'nearest to class A center should be class 0');
  });

  it('kdeScore returns density and bestClass', () => {
    const r = kdeScore(classes[0].muHat, classes, kdeBandwidth);
    assert.ok(typeof r.density === 'number');
    assert.equal(r.bestClass, 0, 'highest density at class A center should be class 0');
  });

  it('perClassDecide returns accepted/acceptCls structure', () => {
    const scoreFn = (x, ci) => v3dot(x, classes[ci].muHat);
    const thresholds = classes.map(c => c.cosThreshold);
    const d = perClassDecide(classes[0].muHat, 5, scoreFn, thresholds, "test");
    assert.ok(typeof d.accepted === 'boolean');
    assert.ok(typeof d.acceptCls === 'number');
    assert.ok(typeof d.cls === 'number');
    assert.ok(typeof d.val === 'number');
  });
});

describe('Registry coverage', () => {
  it('all registry keys are accounted for in tests', () => {
    const allRegistryKeys = allKeys();
    const testedKeys = new Set([...decideKeys, "none", "cp-proto", "cp-fc"]);
    for (const key of allRegistryKeys) {
      assert.ok(testedKeys.has(key), `registry key "${key}" not covered by tests`);
    }
  });

  it('none/conformal methods have no decideFn', () => {
    for (const key of ["none", "cp-proto", "cp-fc"]) {
      const m = getMethod(key);
      assert.ok(m, `"${key}" should be in registry`);
      assert.ok(!m.decideFn, `"${key}" should not have decideFn`);
    }
  });
});
