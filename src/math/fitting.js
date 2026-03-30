"use strict";

import { v3dot, v3cross, v3norm, v3add, v3sub, v3scale, v3len } from './vec3.js';
import { kappaFromRbar3D } from './spherical.js';
import { quantile } from './stats.js';

// Kent distribution fitting: returns { kappa, beta, gamma1 (mean dir), gamma2, gamma3 (tangent axes), kentScore function, bands }
export function fitKent(pts, anchorDir) {
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
  // beta from eigenvalue ratio: beta ≈ kappa * (l1 - l2) / (2 * (l1 + l2))
  const beta = (l1 + l2 > 1e-12) ? kappa * (l1 - l2) / (2 * (l1 + l2)) : 0;
  // Kent log-density score (up to constant): kappa * mu^T x + beta * [(gamma2^T x)^2 - (gamma3^T x)^2]
  function kentScore(x) {
    return kappa * v3dot(mu, x) + beta * (v3dot(ev1, x) * v3dot(ev1, x) - v3dot(ev2, x) * v3dot(ev2, x));
  }
  // Compute training score quantile bands
  const trainScores = pts.map(p => kentScore(p));
  const bands = [0.05, 0.20, 0.50].map(q => quantile(trainScores, q));
  return { kappa, beta, gamma1: mu, gamma2: ev1, gamma3: ev2, kentScore, bands };
}

// Reconstruct a kentScore function from serialized Kent parameters.
// Accepts an object { kappa, beta, gamma1, gamma2, gamma3 } to match
// the monolith's calling convention (e.g. makeKentScoreFn(pc.kent)).
export function makeKentScoreFn(params) {
  const { kappa, beta, gamma1, gamma2, gamma3 } = params;
  return function kentScore(x) {
    return kappa * v3dot(gamma1, x) + beta * (v3dot(gamma2, x) * v3dot(gamma2, x) - v3dot(gamma3, x) * v3dot(gamma3, x));
  };
}
