"use strict";

import { v3sub, v3cross, v3len, v3norm } from './vec3.js';

export function covMatrix3(pts, mu) {
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

export function pooledCov(classes) {
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

export function inv3(M) {
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

export function mahalDist2(x, mu, invCov) {
  const d = v3sub(x, mu);
  let result = 0;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      result += d[i] * invCov[i][j] * d[j];
  return result;
}

// ── 3x3 symmetric eigendecomposition (Cardano) ──
export function eig3sym(A) {
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
