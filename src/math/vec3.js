"use strict";

// ── vector math (3D arrays) ──
export function v3add(a,b){ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
export function v3sub(a,b){ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
export function v3scale(a,s){ return [a[0]*s, a[1]*s, a[2]*s]; }
export function v3dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
export function v3cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
export function v3len(a){ return Math.sqrt(v3dot(a,a)); }
export function v3norm(a){ const l=v3len(a); return l<1e-12?[0,0,0]:[a[0]/l,a[1]/l,a[2]/l]; }
