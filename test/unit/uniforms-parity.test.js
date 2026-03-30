import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { PRE } from '../load-oracle.js';

import { v3dot, v3norm, v3len } from '../../src/math/vec3.js';
import { quantile, softmax } from '../../src/math/stats.js';
import { fitKent, makeKentScoreFn } from '../../src/math/fitting.js';
import { covMatrix3, inv3 } from '../../src/math/linalg.js';
import { getMethod, allKeys } from '../../src/config/method-registry.js';
import { makeFCLogitsFn } from '../../src/methods/scoring.js';

const TAU = 5.0;
const Q_VAL = 0.05;

// Reconstruct classes from oracle
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
const fcBal = (() => {
  const pf = PRE.fcBal;
  return {
    W: pf.W, B: pf.B,
    logitsFn: makeFCLogitsFn(pf.W, pf.B),
    wDirs: pf.W.map(w => v3norm(w)),
    fcKappas: pf.W.map(w => v3len(w)),
    fcKents: classes.map((c, ci) => fitKent(c.points, v3norm(pf.W[ci]))),
    bestT: pf.bestT,
  };
})();

// Build minimal threshold object and sorted arrays
const sortedMLSscores = PRE.trainMLSscores.slice().sort((a, b) => a - b);
const sortedEBOscores = PRE.trainEBOscores.slice().sort((a, b) => a - b);
const sortedTSMSP = PRE.trainTSMSP.map(xs => xs.slice().sort((a, b) => a - b));
const sortedRMDS = PRE.trainRMDS.slice().sort((a, b) => a - b);
const sortedVimScores = PRE.trainVimScores.slice().sort((a, b) => a - b);
const sortedOdinPerClass = PRE.trainOdinPerClass.map(xs => xs.slice().sort((a, b) => a - b));

const thr = {
  mspGamma: PRE.gamma, mlsThr: PRE.mlsThreshold, eboThr: PRE.eboThreshold,
  tsGamma: PRE.tsGamma, odinGamma: PRE.trainOdinPerClass.map(xs => quantile(xs, Q_VAL)).reduce((a, b) => a + b, 0) / 5,
  vimThr: PRE.vimThreshold,
  cosPerClass: classes.map(c => c.cosThreshold),
  vmfBands: classes.map(c => c.vmBands),
  kentBands: classes.map(c => c.kent.bands),
  rmdsThr: PRE.rmdsThreshold,
  fcMspGamma: PRE.fcBal.gamma, fcMlsThr: PRE.fcBal.mlsThr, fcEboThr: PRE.fcBal.eboThr,
  fcTsGamma: PRE.fcBal.tsGamma, fcOdinGamma: PRE.fcBal.odinGamma, fcVimThr: PRE.fcBal.vimThreshold,
  fcCosPerClass: PRE.fcBal.fcCosThresholds,
  fcVmfBands: PRE.fcBal.fcVmBands,
  fcKentBands: classes.map((c, ci) => fcBal.fcKents[ci].bands),
};

// Sorted FC scores
const fcMspPerClass = classes.map((c, i) =>
  c.points.map(p => softmax(fcBal.logitsFn(p))[i]).sort((a, b) => a - b));
const fcMlsScores = PRE.allTrainPts.map(p => Math.max(...fcBal.logitsFn(p))).sort((a, b) => a - b);
const fcEboScores = PRE.allTrainPts.map(p => {
  const l = fcBal.logitsFn(p);
  const mx = Math.max(...l);
  return mx + Math.log(l.reduce((s, v) => s + Math.exp(v - mx), 0));
}).sort((a, b) => a - b);
const fcTsPerClass = classes.map((c, i) =>
  c.points.map(p => softmax(fcBal.logitsFn(p).map(l => l / fcBal.bestT))[i]).sort((a, b) => a - b));
const fcOdinSorted = (PRE.fcBal.odinPerClass || []).map(xs => xs.slice().sort((a, b) => a - b));
const fcVimSorted = (PRE.fcBal.trainVimScores || []).slice().sort((a, b) => a - b);

const sortedFC = {
  fcMspPerClass, fcMlsScores, fcEboScores, fcTsPerClass,
  fcOdinSorted, fcVimSorted,
};

// Test ctx (no THREE — uses plain array fallback)
const ctx = {
  vec3: (x, y, z) => [x, y, z],
  mat3Set: function() { return Array.from(arguments); },
  classColors: classes.map(() => [0, 0, 0]),
  W, TAU, classes, poolInvCov: PRE.poolInvCov, muAll: PRE.muAll, bestT: PRE.bestT,
  vimU: PRE.vimU, vimAlpha: PRE.vimAlpha, fcVimAlpha: PRE.fcBal.vimAlpha,
  odinT: 1000, odinEps: 0.002,
  fc: fcBal, thr,
  trainMsp: PRE.trainMsp,
  sortedMLSscores, sortedEBOscores, sortedTSMSP,
  sortedOdinPerClass, sortedVimScores, sortedRMDS,
  sortedFC,
};

// Methods with buildUniforms (procedural render kind)
const proceduralKeys = [
  "msp", "mls", "ebo", "ts", "odin", "vim",
  "fc-msp", "fc-mls", "fc-ebo", "fc-ts", "fc-odin", "fc-vim",
  "cos", "vmf", "kent", "fc-cos", "fc-vmf", "fc-kent",
  "rmds",
];

describe('Registry buildUniforms dispatch', () => {
  it('every procedural method has a buildUniforms function', () => {
    for (const key of proceduralKeys) {
      const m = getMethod(key);
      assert.ok(m, `"${key}" should be in registry`);
      assert.ok(typeof m.buildUniforms === 'function', `"${key}" should have buildUniforms`);
    }
  });

  for (const key of proceduralKeys) {
    it(`"${key}" buildUniforms returns valid uniform object`, () => {
      const m = getMethod(key);
      const u = m.buildUniforms(ctx);
      assert.ok(u && typeof u === 'object', `"${key}" should return an object`);
      // Every uniform should have a .value property
      for (const [uName, uVal] of Object.entries(u)) {
        assert.ok(uVal && 'value' in uVal,
          `"${key}" uniform "${uName}" should have .value, got ${JSON.stringify(uVal).slice(0, 80)}`);
      }
      // Must have classColors
      assert.ok('classColors' in u, `"${key}" should include classColors uniform`);
    });
  }

  it('logit methods have scoreMode', () => {
    for (const key of ["msp", "mls", "ebo", "ts", "fc-msp", "fc-mls", "fc-ebo", "fc-ts"]) {
      const u = getMethod(key).buildUniforms(ctx);
      assert.ok('scoreMode' in u, `"${key}" should have scoreMode`);
    }
  });

  it('directional methods have perClassThr', () => {
    for (const key of ["cos", "vmf", "kent", "fc-cos", "fc-vmf", "fc-kent"]) {
      const u = getMethod(key).buildUniforms(ctx);
      assert.ok('perClassThr' in u, `"${key}" should have perClassThr`);
      assert.equal(u.perClassThr.value.length, 5, `"${key}" perClassThr should have 5 entries`);
    }
  });

  it('odin methods have odinT and odinEps', () => {
    for (const key of ["odin", "fc-odin"]) {
      const u = getMethod(key).buildUniforms(ctx);
      assert.ok('odinT' in u, `"${key}" should have odinT`);
      assert.ok('odinEps' in u, `"${key}" should have odinEps`);
    }
  });

  it('vim methods have vimU0, vimU1, muAll', () => {
    for (const key of ["vim", "fc-vim"]) {
      const u = getMethod(key).buildUniforms(ctx);
      assert.ok('vimU0' in u && 'vimU1' in u && 'muAll' in u,
        `"${key}" should have vimU0, vimU1, muAll`);
    }
  });

  it('rmds has invCov, classMeans, globalMean', () => {
    const u = getMethod("rmds").buildUniforms(ctx);
    assert.ok('invCov' in u, 'rmds should have invCov');
    assert.ok('classMeans' in u, 'rmds should have classMeans');
    assert.ok('globalMean' in u, 'rmds should have globalMean');
  });

  it('non-procedural methods have no buildUniforms', () => {
    for (const key of ["knn", "kde", "mds", "mds-s", "none", "cp-proto", "cp-fc"]) {
      const m = getMethod(key);
      assert.ok(!m.buildUniforms, `"${key}" should NOT have buildUniforms`);
    }
  });
});
