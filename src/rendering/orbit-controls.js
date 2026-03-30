// ═══════════════════════════════════════════════════════════════════
//  Orbit controls: pointer/wheel handlers, lookAtPoint, animate loop.
//  Camera state is local to this module — not in the global state store.
// ═══════════════════════════════════════════════════════════════════

import { v3norm } from '../math/vec3.js';

let _camera, _renderer, _scene, _container, _H3;
let _raycastSphere, _showProbe;
let THREE;

let camDist = 3.2, panX = 0, panY = 0, panZ = 0;
let dragging = false, panning = false, px = 0, py = 0;
let rotY = 0.5, rotX = 0.3;
let clickStartX = 0, clickStartY = 0;
let rotSpeed = 0.002;
let _savedRotSpeed = 0.002;
const ROT_MAX = 0.002;

let _raycaster, _pointerNDC;

export function initOrbitControls(deps) {
  THREE = deps.THREE;
  _camera = deps.camera;
  _renderer = deps.renderer;
  _scene = deps.scene;
  _container = deps.container;
  _H3 = deps.H3;
  _raycastSphere = deps.raycastSphere;
  _showProbe = deps.showProbe;

  _raycaster = new THREE.Raycaster();
  _pointerNDC = new THREE.Vector2();

  _renderer.domElement.addEventListener("pointerdown", _onPointerDown);
  _renderer.domElement.addEventListener("contextmenu", e => e.preventDefault());
  window.addEventListener("pointermove", _onPointerMove);
  window.addEventListener("pointerup", _onPointerUp);
  _renderer.domElement.addEventListener("wheel", _onWheel, { passive: false });

  // Rotation speed slider
  const rotSlider = document.getElementById("rot-speed");
  const rotValInput = document.getElementById("rot-speed-val");
  if (rotSlider) {
    rotSlider.addEventListener("input", function() {
      applyRotSpeed(parseInt(this.value) / 100);
    });
  }
  if (rotValInput) {
    rotValInput.addEventListener("keydown", function(e) {
      if (e.key === "Enter") this.blur();
      if (e.key === "Escape") { this.value = (rotSpeed / ROT_MAX).toFixed(1); this.blur(); }
    });
    rotValInput.addEventListener("blur", function() {
      const v = parseFloat(this.value);
      if (isNaN(v)) { this.value = (rotSpeed / ROT_MAX).toFixed(1); return; }
      applyRotSpeed(v);
    });
  }
  const zoomReset = document.getElementById("zoom-reset");
  if (zoomReset) zoomReset.addEventListener("click", () => { camDist = 3.2; panX = 0; panY = 0; panZ = 0; });

  window.addEventListener("resize", () => {
    const nw = _container.clientWidth;
    _camera.aspect = nw / _H3;
    _camera.updateProjectionMatrix();
    _renderer.setSize(nw, _H3);
  });

  animate();
}

function _onPointerDown(e) {
  px = e.clientX; py = e.clientY;
  clickStartX = e.clientX; clickStartY = e.clientY;
  if (e.button === 1) { panning = true; e.preventDefault(); _container.style.cursor = "move"; }
  else { dragging = true; _container.style.cursor = "grabbing"; }
}

function _onPointerMove(e) {
  if (panning) {
    // Pan along camera-local right/up vectors so direction is consistent at any rotY
    const dx = (e.clientX - px) * 0.003 * camDist;
    const dy = (e.clientY - py) * 0.003 * camDist;
    // Camera right in world XZ = (cos(rotY), 0, -sin(rotY))
    panX -= dx * Math.cos(rotY);
    panZ += dx * Math.sin(rotY);
    panY += dy;
    px = e.clientX; py = e.clientY;
    return;
  }
  if (!dragging) return;
  rotY -= (e.clientX - px) * 0.007;
  rotX += (e.clientY - py) * 0.007;
  rotX = Math.max(-Math.PI / 2.1, Math.min(Math.PI / 2.1, rotX));
  px = e.clientX; py = e.clientY;
}

function _onPointerUp(e) {
  const wasDrag = Math.abs(e.clientX - clickStartX) > 4 || Math.abs(e.clientY - clickStartY) > 4;
  dragging = false; panning = false;
  _container.style.cursor = "grab";
  if (wasDrag) return;
  // Right-click: toggle rotation pause/resume
  if (e.button === 2) {
    if (rotSpeed === 0) {
      rotSpeed = _savedRotSpeed || ROT_MAX;
    } else {
      _savedRotSpeed = rotSpeed;
      rotSpeed = 0;
    }
    const rs = document.getElementById("rot-speed");
    const rv = document.getElementById("rot-speed-val");
    if (rs) rs.value = Math.round((rotSpeed / ROT_MAX) * 100);
    if (rv) rv.value = (rotSpeed / ROT_MAX).toFixed(1);
    return;
  }
  if (e.button !== 0) return;
  const rect = _renderer.domElement.getBoundingClientRect();
  _pointerNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  _pointerNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  _raycaster.setFromCamera(_pointerNDC, _camera);
  const hits = _raycaster.intersectObject(_raycastSphere);
  if (hits.length > 0) {
    const hp = hits[0].point;
    _showProbe(v3norm([hp.x, hp.y, hp.z]));
  }
}

function _onWheel(e) {
  e.preventDefault();
  camDist *= e.deltaY > 0 ? 1.08 : 0.93;
  camDist = Math.max(1.5, Math.min(8, camDist));
}

function applyRotSpeed(frac) {
  frac = Math.max(0, Math.min(1, Math.round(frac * 10) / 10));
  rotSpeed = ROT_MAX * frac;
  const rs = document.getElementById("rot-speed");
  const rv = document.getElementById("rot-speed-val");
  if (rs) rs.value = Math.round(frac * 100);
  if (rv) rv.value = frac.toFixed(1);
}

export function lookAtPoint(pt) {
  rotY = Math.atan2(pt[0], pt[2]);
  rotX = Math.asin(Math.max(-1, Math.min(1, pt[1])));
  panX = 0; panY = 0; panZ = 0;
  rotSpeed = 0;
  const rs = document.getElementById("rot-speed");
  const rv = document.getElementById("rot-speed-val");
  if (rs) rs.value = 0;
  if (rv) rv.value = "0.0";
}

function animate() {
  requestAnimationFrame(animate);
  if (!dragging) rotY += rotSpeed;
  _camera.position.set(
    camDist * Math.cos(rotX) * Math.sin(rotY) + panX,
    camDist * Math.sin(rotX) + panY,
    camDist * Math.cos(rotX) * Math.cos(rotY) + panZ
  );
  _camera.lookAt(panX, panY, panZ);
  _renderer.render(_scene, _camera);
}
