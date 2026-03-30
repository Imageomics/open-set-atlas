// ═══════════════════════════════════════════════════════════════════
//  Per-method GPU shader uniform builders.
//  Each buildUniforms(ctx) returns a complete uniforms object for
//  THREE.ShaderMaterial.  ctx.vec3(x,y,z) creates Vector3 in
//  production (THREE) or a plain array in tests.
//
//  ctx shape:
//    { vec3, mat3Set, classColors,
//      W, TAU, classes, poolInvCov, muAll, bestT,
//      vimU, vimAlpha, fcVimAlpha, odinT, odinEps,
//      fc, thr,
//      trainMsp, sortedMLSscores, sortedEBOscores, sortedTSMSP,
//      sortedOdinPerClass, sortedVimScores, sortedRMDS,
//      sortedFC }
// ═══════════════════════════════════════════════════════════════════

import { v3dot } from '../math/vec3.js';

function v3f(ctx, a) { return ctx.vec3(a[0], a[1], a[2]); }

function protoW(ctx) {
  return ctx.W.map(w => ctx.vec3(w[0]*ctx.TAU, w[1]*ctx.TAU, w[2]*ctx.TAU));
}
function protoB() { return [0, 0, 0, 0, 0]; }
function fcW(ctx) { return ctx.fc.W.map(w => v3f(ctx, w)); }
function fcB(ctx) { return ctx.fc.B.slice(); }

function maxSorted(arr) { return arr[arr.length - 1]; }
function maxPerClass(perClassArr) {
  return Math.max(...perClassArr.map(xs => xs[xs.length - 1]));
}

export const methodUniforms = {

  // ── MSP ──
  "msp": {
    buildUniforms(ctx) {
      return {
        classW: { value: protoW(ctx) },
        classB: { value: protoB() },
        threshold: { value: ctx.thr.mspGamma },
        maxTrainScore: { value: Math.max(...ctx.trainMsp.map(xs => Math.max(...xs))) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 0 },
      };
    },
  },
  "fc-msp": {
    buildUniforms(ctx) {
      return {
        classW: { value: fcW(ctx) },
        classB: { value: fcB(ctx) },
        threshold: { value: ctx.thr.fcMspGamma },
        maxTrainScore: { value: maxPerClass(ctx.sortedFC.fcMspPerClass) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 0 },
      };
    },
  },

  // ── MLS ──
  "mls": {
    buildUniforms(ctx) {
      return {
        classW: { value: protoW(ctx) },
        classB: { value: protoB() },
        threshold: { value: ctx.thr.mlsThr },
        maxTrainScore: { value: maxSorted(ctx.sortedMLSscores) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 1 },
      };
    },
  },
  "fc-mls": {
    buildUniforms(ctx) {
      return {
        classW: { value: fcW(ctx) },
        classB: { value: fcB(ctx) },
        threshold: { value: ctx.thr.fcMlsThr },
        maxTrainScore: { value: maxSorted(ctx.sortedFC.fcMlsScores) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 1 },
      };
    },
  },

  // ── EBO ──
  "ebo": {
    buildUniforms(ctx) {
      return {
        classW: { value: protoW(ctx) },
        classB: { value: protoB() },
        threshold: { value: ctx.thr.eboThr },
        maxTrainScore: { value: maxSorted(ctx.sortedEBOscores) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 2 },
      };
    },
  },
  "fc-ebo": {
    buildUniforms(ctx) {
      return {
        classW: { value: fcW(ctx) },
        classB: { value: fcB(ctx) },
        threshold: { value: ctx.thr.fcEboThr },
        maxTrainScore: { value: maxSorted(ctx.sortedFC.fcEboScores) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 2 },
      };
    },
  },

  // ── TempScale ──
  "ts": {
    buildUniforms(ctx) {
      const T = ctx.bestT;
      return {
        classW: { value: ctx.W.map(w => ctx.vec3(w[0]*ctx.TAU/T, w[1]*ctx.TAU/T, w[2]*ctx.TAU/T)) },
        classB: { value: protoB() },
        threshold: { value: ctx.thr.tsGamma },
        maxTrainScore: { value: maxPerClass(ctx.sortedTSMSP) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 0 },
      };
    },
  },
  "fc-ts": {
    buildUniforms(ctx) {
      const T = ctx.fc.bestT;
      return {
        classW: { value: ctx.fc.W.map(w => ctx.vec3(w[0]/T, w[1]/T, w[2]/T)) },
        classB: { value: ctx.fc.B.map(b => b / T) },
        threshold: { value: ctx.thr.fcTsGamma },
        maxTrainScore: { value: maxPerClass(ctx.sortedFC.fcTsPerClass) },
        classColors: { value: ctx.classColors },
        scoreMode: { value: 0 },
      };
    },
  },

  // ── Cosine ──
  "cos": {
    buildUniforms(ctx) {
      return {
        classDir: { value: ctx.classes.map(c => v3f(ctx, c.muHat)) },
        classKappa: { value: [1, 1, 1, 1, 1] },
        perClassThr: { value: ctx.thr.cosPerClass.slice() },
        classColors: { value: ctx.classColors },
        isGradient: { value: 1.0 },
        maxTrainScores: { value: ctx.classes.map(c => Math.max(...c.cosScores)) },
      };
    },
  },
  "fc-cos": {
    buildUniforms(ctx) {
      return {
        classDir: { value: ctx.fc.wDirs.map(d => v3f(ctx, d)) },
        classKappa: { value: [1, 1, 1, 1, 1] },
        perClassThr: { value: ctx.thr.fcCosPerClass.slice() },
        classColors: { value: ctx.classColors },
        isGradient: { value: 1.0 },
        maxTrainScores: { value: ctx.classes.map(c => Math.max(...c.cosScores)) },
      };
    },
  },

  // ── vMF ──
  "vmf": {
    buildUniforms(ctx) {
      return {
        classDir: { value: ctx.classes.map(c => v3f(ctx, c.muHat)) },
        classKappa: { value: ctx.classes.map(c => c.kappa) },
        perClassThr: { value: ctx.thr.vmfBands.map(b => b[0]) },
        classColors: { value: ctx.classColors },
        isGradient: { value: 1.0 },
        maxTrainScores: { value: ctx.classes.map(c => Math.max(...c.vmScores)) },
      };
    },
  },
  "fc-vmf": {
    buildUniforms(ctx) {
      return {
        classDir: { value: ctx.fc.wDirs.map(d => v3f(ctx, d)) },
        classKappa: { value: ctx.fc.fcKappas.slice() },
        perClassThr: { value: ctx.thr.fcVmfBands.map(b => b[0]) },
        classColors: { value: ctx.classColors },
        isGradient: { value: 1.0 },
        maxTrainScores: { value: ctx.classes.map((c, ci) =>
          Math.max(...c.points.map(p => ctx.fc.fcKappas[ci] * v3dot(p, ctx.fc.wDirs[ci])))
        ) },
      };
    },
  },

  // ── Kent ──
  "kent": {
    buildUniforms(ctx) {
      return {
        kentGamma1: { value: ctx.classes.map(c => v3f(ctx, c.kent.gamma1)) },
        kentGamma2: { value: ctx.classes.map(c => v3f(ctx, c.kent.gamma2)) },
        kentKappa: { value: ctx.classes.map(c => c.kent.kappa) },
        kentBeta: { value: ctx.classes.map(c => c.kent.beta) },
        perClassThr: { value: ctx.thr.kentBands.map(b => b[0]) },
        classColors: { value: ctx.classColors },
        isGradient: { value: 1.0 },
        maxTrainScores: { value: ctx.classes.map(c => Math.max(...c.kentScores)) },
      };
    },
  },
  "fc-kent": {
    buildUniforms(ctx) {
      return {
        kentGamma1: { value: ctx.fc.fcKents.map(k => v3f(ctx, k.gamma1)) },
        kentGamma2: { value: ctx.fc.fcKents.map(k => v3f(ctx, k.gamma2)) },
        kentKappa: { value: ctx.fc.fcKents.map(k => k.kappa) },
        kentBeta: { value: ctx.fc.fcKents.map(k => k.beta) },
        perClassThr: { value: ctx.thr.fcKentBands.map(b => b[0]) },
        classColors: { value: ctx.classColors },
        isGradient: { value: 1.0 },
        maxTrainScores: { value: ctx.classes.map((c, ci) =>
          Math.max(...c.points.map(p => ctx.fc.fcKents[ci].kentScore(p)))
        ) },
      };
    },
  },

  // ── ODIN ──
  "odin": {
    buildUniforms(ctx) {
      return {
        classW: { value: protoW(ctx) },
        classB: { value: protoB() },
        odinT: { value: ctx.odinT },
        odinEps: { value: ctx.odinEps },
        threshold: { value: ctx.thr.odinGamma },
        maxTrainScore: { value: maxPerClass(ctx.sortedOdinPerClass) },
        classColors: { value: ctx.classColors },
      };
    },
  },
  "fc-odin": {
    buildUniforms(ctx) {
      return {
        classW: { value: fcW(ctx) },
        classB: { value: fcB(ctx) },
        odinT: { value: ctx.odinT },
        odinEps: { value: ctx.odinEps },
        threshold: { value: ctx.thr.fcOdinGamma },
        maxTrainScore: { value: maxPerClass(ctx.sortedFC.fcOdinSorted) },
        classColors: { value: ctx.classColors },
      };
    },
  },

  // ── ViM ──
  "vim": {
    buildUniforms(ctx) {
      return {
        classW: { value: protoW(ctx) },
        classB: { value: protoB() },
        vimU0: { value: v3f(ctx, ctx.vimU[0]) },
        vimU1: { value: v3f(ctx, ctx.vimU[1]) },
        muAll: { value: v3f(ctx, ctx.muAll) },
        vimAlphaU: { value: ctx.vimAlpha },
        threshold: { value: ctx.thr.vimThr },
        maxTrainScore: { value: maxSorted(ctx.sortedVimScores) },
        classColors: { value: ctx.classColors },
      };
    },
  },
  "fc-vim": {
    buildUniforms(ctx) {
      return {
        classW: { value: fcW(ctx) },
        classB: { value: fcB(ctx) },
        vimU0: { value: v3f(ctx, ctx.vimU[0]) },
        vimU1: { value: v3f(ctx, ctx.vimU[1]) },
        muAll: { value: v3f(ctx, ctx.muAll) },
        vimAlphaU: { value: ctx.fcVimAlpha },
        threshold: { value: ctx.thr.fcVimThr },
        maxTrainScore: { value: maxSorted(ctx.sortedFC.fcVimSorted) },
        classColors: { value: ctx.classColors },
      };
    },
  },

  // ── RMDS ──
  "rmds": {
    buildUniforms(ctx) {
      const ic = ctx.poolInvCov;
      return {
        invCov: { value: ctx.mat3Set(
          ic[0][0], ic[0][1], ic[0][2],
          ic[1][0], ic[1][1], ic[1][2],
          ic[2][0], ic[2][1], ic[2][2]
        ) },
        classMeans: { value: ctx.classes.map(c => v3f(ctx, c.muBar)) },
        globalMean: { value: v3f(ctx, ctx.muAll) },
        threshold: { value: ctx.thr.rmdsThr },
        minTrainScore: { value: ctx.sortedRMDS[0] },
        classColors: { value: ctx.classColors },
      };
    },
  },
};
