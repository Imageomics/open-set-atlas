"use strict";

// ═══════════════════════════════════════════════════════════════════
//  FC-balanced toggle: the most complex state transition in the app.
//
//  When fcBalanced changes, 11 things must happen in order:
//
//    1. Switch fc variant (fcBal ↔ fcUnbal)
//    2. Rebuild FC markers (dispose old geometry, create new)
//    3. Invalidate procedural shader cache (FC methods)
//    4. Invalidate conformal procedural cache (FC configs)
//    5. Clear conformal calibration score cache
//    6. Recompute FC ODIN per-class thresholds
//    7. Recompute FC ViM alpha
//    8. Rebuild sorted FC score arrays
//    9. Recompute all thresholds at current q
//   10. Update descriptions
//   11. Re-render active method
//
//  This module declares these as named effects with explicit
//  dependency ordering. The effect graph in app-state.js runs
//  them in the correct topological order.
//
//  Previously: 73 lines of imperative code in a single
//  addEventListener("change") callback (html:4081-4154).
// ═══════════════════════════════════════════════════════════════════

import { registerEffect } from './app-state.js';

// These will be bound during initialization by main.js.
// The effect system doesn't own the resources — it orchestrates them.
let _callbacks = {};

export function bindFCEffects(callbacks) {
  _callbacks = callbacks;
}

// 1. Switch FC variant
registerEffect("switchFCVariant", ["fcBalanced"], (state) => {
  if (_callbacks.switchVariant) _callbacks.switchVariant(state.fcBalanced);
});

// 2. Rebuild FC markers
registerEffect("rebuildFCMarkers", ["fcBalanced"], (state) => {
  if (_callbacks.rebuildMarkers) _callbacks.rebuildMarkers();
}, { after: ["switchFCVariant"] });

// 3. Invalidate procedural shader cache for FC methods
registerEffect("invalidateFCProceduralCache", ["fcBalanced"], (state) => {
  if (_callbacks.invalidateProceduralCache) _callbacks.invalidateProceduralCache();
}, { after: ["switchFCVariant"] });

// 4. Invalidate conformal procedural cache
registerEffect("invalidateConformalCache", ["fcBalanced"], (state) => {
  if (_callbacks.invalidateConformalCache) _callbacks.invalidateConformalCache();
}, { after: ["switchFCVariant"] });

// 5. Clear conformal calibration scores
registerEffect("clearConformalCalibScores", ["fcBalanced"], (state) => {
  if (_callbacks.clearCalibScores) _callbacks.clearCalibScores();
}, { after: ["switchFCVariant"] });

// 6-7. Recompute FC ODIN/ViM thresholds
registerEffect("recomputeFCOdinVim", ["fcBalanced"], (state) => {
  if (_callbacks.recomputeOdinVim) _callbacks.recomputeOdinVim(state.fcBalanced);
}, { after: ["switchFCVariant"] });

// 8. Rebuild sorted FC score arrays
registerEffect("rebuildSortedFCScores", ["fcBalanced"], (state) => {
  if (_callbacks.rebuildSortedScores) _callbacks.rebuildSortedScores();
}, { after: ["recomputeFCOdinVim"] });

// 9. Recompute thresholds — handled by threshold-engine.js
// (it declares { after: ["rebuildSortedFCScores"] })

// 10. Update descriptions
registerEffect("updateFCDescriptions", ["fcBalanced"], (state) => {
  if (_callbacks.updateDescriptions) _callbacks.updateDescriptions();
}, { after: ["rebuildSortedFCScores"] });

// 11. Re-render active method
registerEffect("rerenderActiveMethod", ["fcBalanced"], (state) => {
  if (_callbacks.rerenderActive) _callbacks.rerenderActive(state.activeMode);
}, { after: ["updateFCDescriptions"] });
