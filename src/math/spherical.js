"use strict";

import { v3dot, v3cross, v3norm, v3add, v3scale, v3sub, v3len } from './vec3.js';

// ── rotation: rotate vector v so that (0,0,1) maps to dir ──
export function rotateToDir(v, dir) {
  const z = [0,0,1];
  const d = v3dot(z, dir);
  if (d > 0.9999) return v.slice();
  if (d < -0.9999) return [-v[0], -v[1], -v[2]];
  const axis = v3norm(v3cross(z, dir));
  const angle = Math.acos(Math.max(-1, Math.min(1, d)));
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const [x,y,zz] = axis;
  // Rodrigues
  return [
    (t*x*x+c)*v[0] + (t*x*y-s*zz)*v[1] + (t*x*zz+s*y)*v[2],
    (t*x*y+s*zz)*v[0] + (t*y*y+c)*v[1] + (t*y*zz-s*x)*v[2],
    (t*x*zz-s*y)*v[0] + (t*y*zz+s*x)*v[1] + (t*zz*zz+c)*v[2]
  ];
}

export function computeMeanDir(pts) {
  let m = [0,0,0];
  for (const p of pts) m = v3add(m, p);
  m = v3scale(m, 1/pts.length);
  return { muBar: m, muHat: v3norm(m), Rbar: v3len(m) };
}

export function kappaFromRbar3D(Rbar) {
  if (Rbar < 1e-8) return 0;
  return Rbar * (3 - Rbar*Rbar) / (1 - Rbar*Rbar);
}

export function hexToRGB(hex) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return [r, g, b];
}

export function dirFromUV(u, v) {
  // Match Three.js SphereGeometry UV convention:
  // u: 0..1 maps to phi: 0..2pi (azimuthal)
  // v: 0..1 maps to theta: 0..pi (colatitude, 0=north pole)
  // Three.js: x = -cos(phi)*sin(theta), y = cos(theta), z = sin(phi)*sin(theta)
  const phi = u * 2 * Math.PI;
  const theta = v * Math.PI;
  const sinT = Math.sin(theta);
  return [-sinT * Math.cos(phi), Math.cos(theta), sinT * Math.sin(phi)];
}
