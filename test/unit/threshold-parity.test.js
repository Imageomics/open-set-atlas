import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { PRE } from '../load-oracle.js';

import { v3norm, v3dot } from '../../src/math/vec3.js';
import { computeMeanDir, kappaFromRbar3D } from '../../src/math/spherical.js';
import { resetSeed, sampleVMF, sampleAniso } from '../../src/math/sampling.js';
import { covMatrix3, pooledCov, inv3 } from '../../src/math/linalg.js';
import { quantile, softmax } from '../../src/math/stats.js';
import { fitKent, makeKentScoreFn } from '../../src/math/fitting.js';

const TOL = 1e-4;

function assertNear(actual, expected, label, tol = TOL) {
  assert.ok(Math.abs(actual - expected) < tol,
    `${label}: expected ${expected}, got ${actual} (diff ${Math.abs(actual - expected)})`);
}

// Reconstruct classes from oracle (same as parity test)
const classDefs = [
  { label:"A", color:"#534AB7", dir:v3norm([0.75,0.55,0.35]), kappa:80, n:200, aniso:false },
  { label:"B", color:"#1D9E75", dir:v3norm([-0.35,0.8,0.5]), kappa:30, n:80, aniso:false },
  { label:"C", color:"#D85A30", dir:v3norm([0.5,-0.65,0.55]), kappa:80, n:15, aniso:false },
  { label:"D", color:"#D4537E", dir:v3norm([-0.55,-0.35,0.75]), kappa:30, n:80, aniso:true, stretch:3.0 },
  { label:"E", color:"#378ADD", dir:v3norm([0.1,0.6,0.8]), kappa:30, n:60, aniso:false },
];

describe('Threshold and covariance parity against oracle', () => {

  let classes;

  it('setup: sample classes', () => {
    resetSeed();
    classes = classDefs.map(def => {
      const points = def.aniso
        ? sampleAniso(def.dir, def.kappa, def.stretch, def.n)
        : sampleVMF(def.dir, def.kappa, def.n);
      const stats = computeMeanDir(points);
      const kappa = kappaFromRbar3D(stats.Rbar);
      const cov = covMatrix3(points, stats.muBar);
      const invCov = inv3(cov);
      const cosScores = points.map(p => v3dot(p, stats.muHat));
      const vmScores = points.map(p => kappa * v3dot(p, stats.muHat));
      const kent = fitKent(points, stats.muHat);
      const kentScores = points.map(p => kent.kentScore(p));
      return { ...def, points, ...stats, kappa, cov, invCov, cosScores, vmScores, kent, kentScores };
    });
  });

  it('pooledCov matches oracle', () => {
    assert.ok(PRE.poolCov, 'oracle should have poolCov');
    const pCov = pooledCov(classes);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        assertNear(pCov[i][j], PRE.poolCov[i][j],
          `poolCov[${i}][${j}]`);
      }
    }
  });

  it('poolInvCov matches oracle', () => {
    assert.ok(PRE.poolInvCov, 'oracle should have poolInvCov');
    const pCov = pooledCov(classes);
    const pInv = inv3(pCov);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        assertNear(pInv[i][j], PRE.poolInvCov[i][j],
          `poolInvCov[${i}][${j}]`, 0.01);
      }
    }
  });

  it('per-class cosine thresholds match oracle', () => {
    const Q = 0.05;
    for (let ci = 0; ci < 5; ci++) {
      if (!PRE.classes[ci].cosThreshold) continue;
      const thr = quantile(classes[ci].cosScores, Q);
      assertNear(thr, PRE.classes[ci].cosThreshold,
        `Class ${classDefs[ci].label} cosThreshold`);
    }
  });

  it('per-class vMF bands match oracle', () => {
    for (let ci = 0; ci < 5; ci++) {
      if (!PRE.classes[ci].vmBands) continue;
      const bands = [0.05, 0.20, 0.50].map(q => quantile(classes[ci].vmScores, q));
      for (let bi = 0; bi < 3; bi++) {
        assertNear(bands[bi], PRE.classes[ci].vmBands[bi],
          `Class ${classDefs[ci].label} vmBand[${bi}]`);
      }
    }
  });

  it('per-class Kent bands match oracle', () => {
    for (let ci = 0; ci < 5; ci++) {
      if (!PRE.classes[ci].kent || !PRE.classes[ci].kent.bands) continue;
      const kentScores = classes[ci].points.map(p => classes[ci].kent.kentScore(p));
      const bands = [0.05, 0.20, 0.50].map(q => quantile(kentScores, q));
      for (let bi = 0; bi < 3; bi++) {
        assertNear(bands[bi], PRE.classes[ci].kent.bands[bi],
          `Class ${classDefs[ci].label} kentBand[${bi}]`, 0.01);
      }
    }
  });

  it('MSP gamma threshold matches oracle', () => {
    assert.ok(PRE.gamma, 'oracle should have gamma');
    const W = classes.map(c => c.muHat);
    const TAU = 5.0;
    const Q = 0.05;
    const trainMsp = classes.map((c, i) =>
      c.points.map(p => {
        const logits = W.map(w => TAU * v3dot(p, w));
        return softmax(logits)[i];
      })
    );
    const perClassGamma = trainMsp.map(xs => quantile(xs, Q));
    const gamma = perClassGamma.reduce((a, b) => a + b, 0) / perClassGamma.length;
    assertNear(gamma, PRE.gamma, 'MSP gamma');
  });

  it('makeKentScoreFn roundtrip matches fitKent kentScore', () => {
    for (let ci = 0; ci < 5; ci++) {
      const kent = classes[ci].kent;
      const reconstructed = makeKentScoreFn({
        kappa: kent.kappa,
        beta: kent.beta,
        gamma1: kent.gamma1,
        gamma2: kent.gamma2,
        gamma3: kent.gamma3,
      });
      // Test on 10 training points
      for (let pi = 0; pi < Math.min(10, classes[ci].points.length); pi++) {
        const p = classes[ci].points[pi];
        assertNear(reconstructed(p), kent.kentScore(p),
          `Class ${classDefs[ci].label} kentScore roundtrip pt ${pi}`, 1e-10);
      }
    }
  });
});
