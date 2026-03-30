"use strict";

import { v3norm, v3add, v3sub, v3scale, v3dot, v3cross } from './vec3.js';
import { rotateToDir } from './spherical.js';

// ── seeded RNG ──
let _seed = 42;
export function rand() { _seed = (_seed * 16807 + 0) % 2147483647; return _seed / 2147483647; }
export function resetSeed() { _seed = 42; }

// ── vMF sampling on S² ──
export function sampleVMF(mu, kappa, n) {
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
export function sampleAniso(mu, kappa, stretchFactor, n) {
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

// ── fibonacci sphere grid ──
export function fibGrid(n) {
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
