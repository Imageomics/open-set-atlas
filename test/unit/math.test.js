import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { v3add, v3sub, v3scale, v3dot, v3cross, v3len, v3norm } from '../../src/math/vec3.js';
import { rotateToDir, computeMeanDir, kappaFromRbar3D } from '../../src/math/spherical.js';
import { resetSeed, rand, sampleVMF, sampleAniso, fibGrid } from '../../src/math/sampling.js';
import { covMatrix3, pooledCov, inv3, mahalDist2, eig3sym } from '../../src/math/linalg.js';
import { quantile, quantileSorted, softmax, logsumexp } from '../../src/math/stats.js';
import { fitKent } from '../../src/math/fitting.js';

const TOL = 1e-10;

function near(a, b, tol = TOL) {
  return Math.abs(a - b) < tol;
}

// ═══════════════════════════════════════════════════════════════
//  vec3
// ═══════════════════════════════════════════════════════════════

describe('vec3', () => {
  it('v3add', () => {
    const r = v3add([1, 2, 3], [4, 5, 6]);
    assert.deepStrictEqual(r, [5, 7, 9]);
  });

  it('v3sub', () => {
    const r = v3sub([4, 5, 6], [1, 2, 3]);
    assert.deepStrictEqual(r, [3, 3, 3]);
  });

  it('v3scale', () => {
    assert.deepStrictEqual(v3scale([1, 2, 3], 2), [2, 4, 6]);
  });

  it('v3dot', () => {
    assert.equal(v3dot([1, 0, 0], [0, 1, 0]), 0);
    assert.equal(v3dot([1, 2, 3], [1, 2, 3]), 14);
  });

  it('v3cross produces orthogonal vector', () => {
    const a = [1, 0, 0], b = [0, 1, 0];
    const c = v3cross(a, b);
    assert.ok(near(v3dot(c, a), 0));
    assert.ok(near(v3dot(c, b), 0));
    assert.deepStrictEqual(c, [0, 0, 1]);
  });

  it('v3len', () => {
    assert.ok(near(v3len([3, 4, 0]), 5));
  });

  it('v3norm returns unit vector', () => {
    const n = v3norm([3, 4, 0]);
    assert.ok(near(v3len(n), 1.0));
    assert.ok(near(n[0], 0.6));
    assert.ok(near(n[1], 0.8));
  });

  it('v3norm of zero vector returns zero', () => {
    assert.deepStrictEqual(v3norm([0, 0, 0]), [0, 0, 0]);
  });
});

// ═══════════════════════════════════════════════════════════════
//  spherical
// ═══════════════════════════════════════════════════════════════

describe('spherical', () => {
  it('rotateToDir identity when dir = [0,0,1]', () => {
    const v = [0.5, 0.3, 0.8];
    const r = rotateToDir(v, [0, 0, 1]);
    for (let i = 0; i < 3; i++) assert.ok(near(r[i], v[i]));
  });

  it('rotateToDir maps [0,0,1] to dir', () => {
    const dir = v3norm([1, 1, 1]);
    const r = rotateToDir([0, 0, 1], dir);
    for (let i = 0; i < 3; i++) assert.ok(near(r[i], dir[i], 1e-6));
  });

  it('rotateToDir preserves vector length', () => {
    const v = [0.3, 0.5, 0.7];
    const dir = v3norm([-0.5, 0.8, 0.2]);
    const r = rotateToDir(v, dir);
    assert.ok(near(v3len(r), v3len(v), 1e-6));
  });

  it('computeMeanDir of identical points returns that point', () => {
    const p = v3norm([1, 1, 1]);
    const pts = [p, p, p, p];
    const { muHat, Rbar } = computeMeanDir(pts);
    for (let i = 0; i < 3; i++) assert.ok(near(muHat[i], p[i], 1e-6));
    assert.ok(near(Rbar, 1.0, 1e-6));
  });

  it('kappaFromRbar3D returns 0 for Rbar near 0', () => {
    assert.equal(kappaFromRbar3D(0), 0);
    assert.equal(kappaFromRbar3D(1e-10), 0);
  });

  it('kappaFromRbar3D increases with Rbar', () => {
    assert.ok(kappaFromRbar3D(0.5) < kappaFromRbar3D(0.9));
  });
});

// ═══════════════════════════════════════════════════════════════
//  sampling
// ═══════════════════════════════════════════════════════════════

describe('sampling', () => {
  it('seeded RNG is deterministic', () => {
    resetSeed();
    const a = [rand(), rand(), rand()];
    resetSeed();
    const b = [rand(), rand(), rand()];
    assert.deepStrictEqual(a, b);
  });

  it('sampleVMF produces points on S²', () => {
    resetSeed();
    const pts = sampleVMF([0, 0, 1], 50, 100);
    assert.equal(pts.length, 100);
    for (const p of pts) {
      assert.ok(near(v3len(p), 1.0, 1e-6), 'point should be on unit sphere');
    }
  });

  it('sampleVMF concentrates around mu for high kappa', () => {
    resetSeed();
    const mu = [0, 0, 1];
    const pts = sampleVMF(mu, 500, 200);
    const meanCos = pts.reduce((s, p) => s + v3dot(p, mu), 0) / pts.length;
    assert.ok(meanCos > 0.99, `high kappa should concentrate tightly, got meanCos=${meanCos}`);
  });

  it('sampleAniso produces points on S²', () => {
    resetSeed();
    const pts = sampleAniso(v3norm([1, 1, 1]), 30, 3.0, 50);
    assert.equal(pts.length, 50);
    for (const p of pts) {
      assert.ok(near(v3len(p), 1.0, 1e-6));
    }
  });

  it('fibGrid produces n unit vectors', () => {
    const pts = fibGrid(100);
    assert.equal(pts.length, 100);
    for (const p of pts) {
      assert.ok(near(v3len(p), 1.0, 1e-6));
    }
  });

  it('fibGrid covers the sphere roughly uniformly', () => {
    const pts = fibGrid(1000);
    let posY = 0, negY = 0;
    for (const p of pts) {
      if (p[1] > 0) posY++; else negY++;
    }
    // Should be roughly 50/50 hemisphere split
    assert.ok(Math.abs(posY - negY) < 100,
      `hemisphere imbalance: ${posY} vs ${negY}`);
  });
});

// ═══════════════════════════════════════════════════════════════
//  stats
// ═══════════════════════════════════════════════════════════════

describe('stats', () => {
  it('quantile of single element returns that element', () => {
    assert.equal(quantile([5], 0.5), 5);
  });

  it('quantile interpolates correctly', () => {
    assert.ok(near(quantile([0, 10], 0.5), 5));
    assert.ok(near(quantile([0, 10], 0.25), 2.5));
  });

  it('quantileSorted matches quantile on sorted input', () => {
    const arr = [3, 1, 4, 1, 5, 9, 2, 6];
    const sorted = [...arr].sort((a, b) => a - b);
    for (const q of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      assert.ok(near(quantileSorted(sorted, q), quantile(arr, q), 1e-10),
        `q=${q}`);
    }
  });

  it('softmax sums to 1', () => {
    const p = softmax([1, 2, 3, 4, 5]);
    const sum = p.reduce((a, b) => a + b, 0);
    assert.ok(near(sum, 1.0, 1e-10));
  });

  it('softmax is stable with large inputs', () => {
    const p = softmax([1000, 1001, 1002]);
    const sum = p.reduce((a, b) => a + b, 0);
    assert.ok(near(sum, 1.0, 1e-10));
    assert.ok(!p.some(v => isNaN(v)), 'no NaN from large inputs');
  });

  it('softmax argmax is correct', () => {
    const p = softmax([1, 5, 2]);
    assert.ok(p[1] > p[0] && p[1] > p[2]);
  });

  it('logsumexp matches log(sum(exp(...)))', () => {
    const vals = [1, 2, 3];
    const naive = Math.log(vals.reduce((s, v) => s + Math.exp(v), 0));
    assert.ok(near(logsumexp(vals), naive));
  });

  it('logsumexp is stable with large inputs', () => {
    const result = logsumexp([1000, 1001, 1002]);
    assert.ok(!isNaN(result), 'no NaN from large inputs');
    assert.ok(isFinite(result), 'should be finite');
  });
});

// ═══════════════════════════════════════════════════════════════
//  linalg
// ═══════════════════════════════════════════════════════════════

describe('linalg', () => {
  it('inv3 roundtrip: M * inv(M) ≈ I', () => {
    const M = [[2, 1, 0], [1, 3, 1], [0, 1, 2]];
    const Mi = inv3(M);
    assert.ok(Mi, 'should be invertible');
    // Multiply M * Mi, check identity
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let sum = 0;
        for (let k = 0; k < 3; k++) sum += M[i][k] * Mi[k][j];
        const expected = i === j ? 1 : 0;
        assert.ok(near(sum, expected, 1e-8),
          `(M*M^-1)[${i}][${j}] = ${sum}, expected ${expected}`);
      }
    }
  });

  it('inv3 returns null for singular matrix', () => {
    const M = [[1, 0, 0], [0, 0, 0], [0, 0, 1]];
    assert.equal(inv3(M), null);
  });

  it('mahalDist2 is zero at the mean', () => {
    const mu = [1, 2, 3];
    const M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    assert.ok(near(mahalDist2(mu, mu, M), 0));
  });

  it('mahalDist2 with identity covariance equals squared Euclidean', () => {
    const x = [1, 0, 0], mu = [0, 0, 0];
    const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    assert.ok(near(mahalDist2(x, mu, I), 1.0));
  });

  it('eig3sym on diagonal matrix returns diagonal entries', () => {
    const D = [[5, 0, 0], [0, 3, 0], [0, 0, 1]];
    const { values } = eig3sym(D);
    const sorted = [...values].sort((a, b) => b - a);
    assert.ok(near(sorted[0], 5));
    assert.ok(near(sorted[1], 3));
    assert.ok(near(sorted[2], 1));
  });

  it('eig3sym eigenvectors are orthonormal', () => {
    const M = [[2, 1, 0], [1, 3, 1], [0, 1, 2]];
    const { vectors } = eig3sym(M);
    for (let i = 0; i < 3; i++) {
      assert.ok(near(v3len(vectors[i]), 1.0, 1e-6),
        `eigenvector ${i} should be unit length`);
      for (let j = i + 1; j < 3; j++) {
        assert.ok(near(Math.abs(v3dot(vectors[i], vectors[j])), 0, 1e-6),
          `eigenvectors ${i},${j} should be orthogonal`);
      }
    }
  });

  it('covMatrix3 is symmetric', () => {
    resetSeed();
    const pts = sampleVMF([0, 0, 1], 30, 50);
    const { muBar } = computeMeanDir(pts);
    const C = covMatrix3(pts, muBar);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        assert.ok(near(C[i][j], C[j][i], 1e-10),
          `cov[${i}][${j}] should equal cov[${j}][${i}]`);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════
//  fitting
// ═══════════════════════════════════════════════════════════════

describe('fitting', () => {
  it('fitKent on isotropic data has small beta', () => {
    resetSeed();
    const mu = v3norm([1, 1, 1]);
    // High kappa, isotropic → beta should be near 0
    const pts = sampleVMF(mu, 200, 500);
    const kent = fitKent(pts, mu);
    assert.ok(kent.beta < kent.kappa * 0.1,
      `beta=${kent.beta} should be small relative to kappa=${kent.kappa}`);
  });

  it('fitKent gamma1 aligns with anchor direction', () => {
    resetSeed();
    const mu = v3norm([0.5, 0.8, 0.3]);
    const pts = sampleVMF(mu, 100, 200);
    const kent = fitKent(pts, mu);
    for (let i = 0; i < 3; i++) {
      assert.ok(near(kent.gamma1[i], mu[i], 1e-10),
        `gamma1 should equal anchor dir`);
    }
  });

  it('fitKent produces orthogonal axes', () => {
    resetSeed();
    const mu = v3norm([0.5, 0.8, 0.3]);
    const pts = sampleVMF(mu, 50, 200);
    const kent = fitKent(pts, mu);
    assert.ok(near(v3dot(kent.gamma1, kent.gamma2), 0, 1e-6),
      'gamma1 ⊥ gamma2');
    assert.ok(near(v3dot(kent.gamma1, kent.gamma3), 0, 1e-6),
      'gamma1 ⊥ gamma3');
    assert.ok(near(v3dot(kent.gamma2, kent.gamma3), 0, 1e-6),
      'gamma2 ⊥ gamma3');
  });

  it('fitKent bands are monotonically increasing', () => {
    resetSeed();
    const pts = sampleVMF([0, 0, 1], 80, 200);
    const kent = fitKent(pts, v3norm([0, 0, 1]));
    assert.ok(kent.bands[0] <= kent.bands[1], 'band[0] <= band[1]');
    assert.ok(kent.bands[1] <= kent.bands[2], 'band[1] <= band[2]');
  });

  it('fitKent on anisotropic data has larger beta', () => {
    resetSeed();
    const mu = v3norm([-0.55, -0.35, 0.75]);
    const iso = sampleVMF(mu, 30, 200);
    const kentIso = fitKent(iso, mu);

    resetSeed();
    const aniso = sampleAniso(mu, 30, 3.0, 200);
    const kentAniso = fitKent(aniso, computeMeanDir(aniso).muHat);

    assert.ok(kentAniso.beta > kentIso.beta,
      `aniso beta=${kentAniso.beta} should exceed iso beta=${kentIso.beta}`);
  });
});
