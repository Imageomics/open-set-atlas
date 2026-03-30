import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { PRE } from '../load-oracle.js';

import { v3norm, v3dot } from '../../src/math/vec3.js';
import { computeMeanDir, kappaFromRbar3D } from '../../src/math/spherical.js';
import { resetSeed, sampleVMF, sampleAniso } from '../../src/math/sampling.js';
import { covMatrix3, inv3 } from '../../src/math/linalg.js';
import { quantile, softmax } from '../../src/math/stats.js';
import { fitKent } from '../../src/math/fitting.js';

const TOL = 1e-6;

function near(a, b, tol = TOL) {
  return Math.abs(a - b) < tol;
}

function assertNear(actual, expected, label, tol = TOL) {
  assert.ok(near(actual, expected, tol),
    `${label}: expected ${expected}, got ${actual} (diff ${Math.abs(actual - expected)})`);
}

function assertVec3Near(actual, expected, label, tol = TOL) {
  for (let i = 0; i < 3; i++) {
    assertNear(actual[i], expected[i], `${label}[${i}]`, tol);
  }
}

// Class definitions — must match monolith exactly
const classDefs = [
  { label:"A", color:"#534AB7", dir:v3norm([0.75,0.55,0.35]), kappa:80, n:200, aniso:false },
  { label:"B", color:"#1D9E75", dir:v3norm([-0.35,0.8,0.5]), kappa:30, n:80, aniso:false },
  { label:"C", color:"#D85A30", dir:v3norm([0.5,-0.65,0.55]), kappa:80, n:15, aniso:false },
  { label:"D", color:"#D4537E", dir:v3norm([-0.55,-0.35,0.75]), kappa:30, n:80, aniso:true, stretch:3.0 },
  { label:"E", color:"#378ADD", dir:v3norm([0.1,0.6,0.8]), kappa:30, n:60, aniso:false },
];

describe('Math parity against precomputed.js oracle', () => {

  it('oracle loaded successfully', () => {
    assert.ok(PRE, 'OOD_PRECOMPUTED should be defined');
    assert.ok(PRE.classes, 'classes array should exist');
    assert.equal(PRE.classes.length, 5, 'should have 5 classes');
  });

  describe('Sampled points', () => {
    // The monolith samples all 5 classes in sequence with seed 42.
    // We do the same and compare point-by-point.

    let sampledClasses;

    it('sampling pipeline produces identical points', () => {
      resetSeed();
      sampledClasses = classDefs.map(def => {
        const points = def.aniso
          ? sampleAniso(def.dir, def.kappa, def.stretch, def.n)
          : sampleVMF(def.dir, def.kappa, def.n);
        return { ...def, points };
      });

      for (let ci = 0; ci < 5; ci++) {
        const ours = sampledClasses[ci].points;
        const oracle = PRE.classes[ci].points;
        assert.equal(ours.length, oracle.length,
          `Class ${classDefs[ci].label}: point count mismatch`);
        for (let pi = 0; pi < ours.length; pi++) {
          assertVec3Near(ours[pi], oracle[pi],
            `Class ${classDefs[ci].label} point ${pi}`);
        }
      }
    });

    it('computeMeanDir matches oracle', () => {
      for (let ci = 0; ci < 5; ci++) {
        const stats = computeMeanDir(sampledClasses[ci].points);
        const oracle = PRE.classes[ci];
        assertVec3Near(stats.muBar, oracle.muBar,
          `Class ${classDefs[ci].label} muBar`);
        assertVec3Near(stats.muHat, oracle.muHat,
          `Class ${classDefs[ci].label} muHat`);
        assertNear(stats.Rbar, oracle.Rbar,
          `Class ${classDefs[ci].label} Rbar`);
      }
    });

    it('kappaFromRbar3D matches oracle', () => {
      for (let ci = 0; ci < 5; ci++) {
        const stats = computeMeanDir(sampledClasses[ci].points);
        const kappa = kappaFromRbar3D(stats.Rbar);
        assertNear(kappa, PRE.classes[ci].kappa,
          `Class ${classDefs[ci].label} kappa`, 0.01);
      }
    });

    it('covMatrix3 matches oracle', () => {
      for (let ci = 0; ci < 5; ci++) {
        if (!PRE.classes[ci].cov) continue;
        const stats = computeMeanDir(sampledClasses[ci].points);
        const cov = covMatrix3(sampledClasses[ci].points, stats.muBar);
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            assertNear(cov[i][j], PRE.classes[ci].cov[i][j],
              `Class ${classDefs[ci].label} cov[${i}][${j}]`, 1e-4);
          }
        }
      }
    });

    it('fitKent parameters match oracle', () => {
      for (let ci = 0; ci < 5; ci++) {
        const stats = computeMeanDir(sampledClasses[ci].points);
        const kent = fitKent(sampledClasses[ci].points, stats.muHat);
        const oracle = PRE.classes[ci].kent;

        assertNear(kent.kappa, oracle.kappa,
          `Class ${classDefs[ci].label} kent.kappa`, 0.01);
        assertNear(kent.beta, oracle.beta,
          `Class ${classDefs[ci].label} kent.beta`, 0.01);
        assertVec3Near(kent.gamma1, oracle.gamma1,
          `Class ${classDefs[ci].label} kent.gamma1`, 1e-4);
        assertVec3Near(kent.gamma2, oracle.gamma2,
          `Class ${classDefs[ci].label} kent.gamma2`, 1e-3);
      }
    });

    it('cosine training scores match oracle', () => {
      for (let ci = 0; ci < 5; ci++) {
        const stats = computeMeanDir(sampledClasses[ci].points);
        const cosScores = sampledClasses[ci].points.map(p => v3dot(p, stats.muHat));
        const oracle = PRE.classes[ci].cosScores;
        if (!oracle) continue;
        assert.equal(cosScores.length, oracle.length,
          `Class ${classDefs[ci].label} cosScores length`);
        for (let pi = 0; pi < cosScores.length; pi++) {
          assertNear(cosScores[pi], oracle[pi],
            `Class ${classDefs[ci].label} cosScore[${pi}]`, 1e-4);
        }
      }
    });

    it('softmax on prototype logits matches oracle MSP scores', () => {
      // Build W (prototype weight directions) same as monolith
      const W = [];
      for (let ci = 0; ci < 5; ci++) {
        W.push(computeMeanDir(sampledClasses[ci].points).muHat);
      }
      const TAU = 5.0;

      // Compute per-class MSP training scores
      const trainMsp = [];
      for (let ci = 0; ci < 5; ci++) {
        const classScores = sampledClasses[ci].points.map(p => {
          const logits = W.map(w => TAU * v3dot(p, w));
          return softmax(logits)[ci];
        });
        trainMsp.push(classScores);
      }

      // Compare against oracle
      if (PRE.trainMsp) {
        for (let ci = 0; ci < 5; ci++) {
          assert.equal(trainMsp[ci].length, PRE.trainMsp[ci].length,
            `Class ${classDefs[ci].label} trainMsp length`);
          for (let pi = 0; pi < trainMsp[ci].length; pi++) {
            assertNear(trainMsp[ci][pi], PRE.trainMsp[ci][pi],
              `Class ${classDefs[ci].label} trainMsp[${pi}]`, 1e-4);
          }
        }
      }
    });
  });
});
