// ═══════════════════════════════════════════════════════════════════
//  Mahalanobis ellipsoid construction for MDS/MDS-S methods.
// ═══════════════════════════════════════════════════════════════════

import { v3scale } from '../math/vec3.js';
import { eig3sym } from '../math/linalg.js';

export function addEllipsoids(THREE, group, covMatrix, classes) {
  const eig = eig3sym(covMatrix);
  const { values, vectors } = eig;
  const sigmas = [1, 2, 3];
  const opacities = [0.18, 0.10, 0.05];

  for (const c of classes) {
    for (let si = sigmas.length - 1; si >= 0; si--) {
      const s = sigmas[si];
      const geo = new THREE.SphereGeometry(1, 24, 16);
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(c.color),
        transparent: true, opacity: opacities[si],
        side: THREE.DoubleSide, depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(c.muBar[0], c.muBar[1], c.muBar[2]);
      const m4 = new THREE.Matrix4();
      const col0 = v3scale(vectors[0], s * Math.sqrt(Math.max(0, values[0])));
      const col1 = v3scale(vectors[1], s * Math.sqrt(Math.max(0, values[1])));
      const col2 = v3scale(vectors[2], s * Math.sqrt(Math.max(0, values[2])));
      m4.set(
        col0[0], col1[0], col2[0], 0,
        col0[1], col1[1], col2[1], 0,
        col0[2], col1[2], col2[2], 0,
        0, 0, 0, 1,
      );
      mesh.geometry.applyMatrix4(m4);
      group.add(mesh);
    }
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 8, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(c.color) }),
    );
    dot.position.set(c.muBar[0], c.muBar[1], c.muBar[2]);
    group.add(dot);
  }
}

export function buildPerClassEllipsoids(THREE, layer, classes) {
  for (const c of classes) {
    const eig = eig3sym(c.cov);
    const { values, vectors } = eig;
    const sigmas = [1, 2, 3];
    const opacities = [0.18, 0.10, 0.05];
    for (let si = sigmas.length - 1; si >= 0; si--) {
      const s = sigmas[si];
      const geo = new THREE.SphereGeometry(1, 24, 16);
      const posAttr = geo.attributes.position;
      for (let vi = 0; vi < posAttr.count; vi++) {
        const vx = posAttr.getX(vi), vy = posAttr.getY(vi), vz = posAttr.getZ(vi);
        const s0 = s * Math.sqrt(Math.max(0, values[0]));
        const s1 = s * Math.sqrt(Math.max(0, values[1]));
        const s2 = s * Math.sqrt(Math.max(0, values[2]));
        posAttr.setXYZ(vi,
          vectors[0][0] * s0 * vx + vectors[1][0] * s1 * vy + vectors[2][0] * s2 * vz,
          vectors[0][1] * s0 * vx + vectors[1][1] * s1 * vy + vectors[2][1] * s2 * vz,
          vectors[0][2] * s0 * vx + vectors[1][2] * s1 * vy + vectors[2][2] * s2 * vz,
        );
      }
      posAttr.needsUpdate = true;
      geo.computeVertexNormals();
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(c.color),
        transparent: true, opacity: opacities[si],
        side: THREE.DoubleSide, depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(c.muBar[0], c.muBar[1], c.muBar[2]);
      layer.add(mesh);
    }
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 8, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(c.color) }),
    );
    dot.position.set(c.muBar[0], c.muBar[1], c.muBar[2]);
    layer.add(dot);
  }
}
