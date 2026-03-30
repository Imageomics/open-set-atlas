import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { v3dot, v3norm, v3len } from '../../src/math/vec3.js';
import { computeMeanDir, kappaFromRbar3D } from '../../src/math/spherical.js';
import { resetSeed, sampleVMF } from '../../src/math/sampling.js';
import { covMatrix3, pooledCov, inv3, eig3sym } from '../../src/math/linalg.js';
import { quantile, quantileSorted, softmax } from '../../src/math/stats.js';
import { fitKent, makeKentScoreFn } from '../../src/math/fitting.js';

const TOL = 1e-10;
function near(a, b, tol = TOL) { return Math.abs(a - b) < tol; }

describe('Edge cases', () => {

  describe('softmax edge cases', () => {
    it('single element returns [1]', () => {
      assert.deepStrictEqual(softmax([42]), [1]);
    });

    it('equal inputs return uniform distribution', () => {
      const p = softmax([0, 0, 0, 0]);
      for (const v of p) assert.ok(near(v, 0.25, 1e-10));
    });

    it('very negative inputs do not produce NaN', () => {
      const p = softmax([-1000, -1001, -999]);
      assert.ok(!p.some(v => isNaN(v)));
      assert.ok(near(p.reduce((a, b) => a + b), 1.0));
    });
  });

  describe('quantile edge cases', () => {
    it('q=0 returns minimum', () => {
      assert.equal(quantile([5, 3, 1, 4, 2], 0), 1);
    });

    it('q=1 returns maximum', () => {
      assert.equal(quantile([5, 3, 1, 4, 2], 1), 5);
    });

    it('two-element array interpolates', () => {
      assert.ok(near(quantile([0, 100], 0.3), 30));
    });
  });

  describe('inv3 edge cases', () => {
    it('identity matrix inverts to itself', () => {
      const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      const result = inv3(I);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          assert.ok(near(result[i][j], I[i][j]));
    });

    it('near-singular matrix with tiny determinant still inverts', () => {
      // det ≈ 1e-10, should still work (threshold is 1e-20)
      const M = [[1, 0, 0], [0, 1, 0], [0, 0, 1e-10]];
      const result = inv3(M);
      assert.ok(result, 'should not return null');
      assert.ok(near(result[2][2], 1e10, 1e5));
    });
  });

  describe('eig3sym edge cases', () => {
    it('identity matrix has all eigenvalues = 1', () => {
      const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      const { values } = eig3sym(I);
      for (const v of values) assert.ok(near(v, 1.0, 1e-6));
    });

    it('eigenvalues sum equals trace', () => {
      const M = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]];
      const { values } = eig3sym(M);
      const trace = M[0][0] + M[1][1] + M[2][2];
      const eigSum = values.reduce((a, b) => a + b);
      assert.ok(near(eigSum, trace, 1e-6),
        `eigenvalue sum ${eigSum} should equal trace ${trace}`);
    });
  });

  describe('covMatrix3', () => {
    it('single-point class has ridge-only covariance', () => {
      const pts = [[1, 0, 0]];
      const mu = [1, 0, 0];
      const C = covMatrix3(pts, mu);
      // With 1 point, deviations are zero, so cov = ridge * I
      const ridge = 1e-5;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const expected = i === j ? ridge : 0;
          assert.ok(near(C[i][j], expected, 1e-8),
            `cov[${i}][${j}] = ${C[i][j]}, expected ${expected}`);
        }
      }
    });

    it('covariance eigenvalues are non-negative', () => {
      resetSeed();
      const pts = sampleVMF([0, 0, 1], 20, 50);
      const { muBar } = computeMeanDir(pts);
      const C = covMatrix3(pts, muBar);
      const { values } = eig3sym(C);
      for (const v of values) {
        assert.ok(v >= -1e-10, `eigenvalue ${v} should be non-negative`);
      }
    });
  });

  describe('pooledCov', () => {
    it('is symmetric', () => {
      resetSeed();
      const cls = [
        { points: sampleVMF([0, 0, 1], 50, 30), muBar: computeMeanDir(sampleVMF([0, 0, 1], 50, 30)).muBar },
        { points: sampleVMF([1, 0, 0], 50, 30), muBar: computeMeanDir(sampleVMF([1, 0, 0], 50, 30)).muBar },
      ];
      // Recompute muBar consistently
      resetSeed();
      const pts1 = sampleVMF([0, 0, 1], 50, 30);
      const pts2 = sampleVMF([1, 0, 0], 50, 30);
      const classes2 = [
        { points: pts1, muBar: computeMeanDir(pts1).muBar },
        { points: pts2, muBar: computeMeanDir(pts2).muBar },
      ];
      const P = pooledCov(classes2);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          assert.ok(near(P[i][j], P[j][i]),
            `pooledCov[${i}][${j}] should equal [${j}][${i}]`);
        }
      }
    });

    it('eigenvalues are positive', () => {
      resetSeed();
      const pts1 = sampleVMF([0, 0, 1], 50, 40);
      const pts2 = sampleVMF([1, 0, 0], 50, 40);
      const classes2 = [
        { points: pts1, muBar: computeMeanDir(pts1).muBar },
        { points: pts2, muBar: computeMeanDir(pts2).muBar },
      ];
      const P = pooledCov(classes2);
      const { values } = eig3sym(P);
      for (const v of values) {
        assert.ok(v > 0, `eigenvalue ${v} should be positive`);
      }
    });
  });

  describe('makeKentScoreFn', () => {
    it('matches fitKent inline kentScore on arbitrary point', () => {
      resetSeed();
      const mu = v3norm([0.3, 0.7, 0.5]);
      const pts = sampleVMF(mu, 80, 100);
      const kent = fitKent(pts, mu);
      const reconstructed = makeKentScoreFn({
        kappa: kent.kappa, beta: kent.beta,
        gamma1: kent.gamma1, gamma2: kent.gamma2, gamma3: kent.gamma3,
      });
      const testPt = v3norm([0.5, 0.5, 0.5]);
      assert.ok(near(reconstructed(testPt), kent.kentScore(testPt), 1e-12));
    });
  });

  describe('kappaFromRbar3D', () => {
    it('high Rbar gives high kappa', () => {
      assert.ok(kappaFromRbar3D(0.99) > 100);
    });

    it('moderate Rbar gives moderate kappa', () => {
      const k = kappaFromRbar3D(0.5);
      assert.ok(k > 1 && k < 20, `kappa=${k} should be moderate`);
    });
  });
});
