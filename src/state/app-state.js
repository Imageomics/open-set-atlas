"use strict";

// ═══════════════════════════════════════════════════════════════════
//  Reactive state with explicit effect graph.
//
//  This is not a generic pub/sub store. It is a topologically-ordered
//  effect system where each effect declares its dependencies and what
//  it invalidates. Changing `fcBalanced` triggers 11 effects in the
//  correct order — markers, caches, thresholds, descriptions, render.
//
//  The FC-balanced toggle is the litmus test. If this system can't
//  handle that transition correctly, it handles nothing correctly.
// ═══════════════════════════════════════════════════════════════════

const state = {
  // Method selection
  activeMode: "none",

  // Threshold controls
  currentQ: 0.05,

  // Conformal controls
  cpAlpha: 0.10,
  cpScoreType: "aps",  // "aps" | "msp" | "emb"

  // FC controls
  fcBalanced: true,

  // Camera
  camDist: 3.2,
  panX: 0,
  panY: 0,
  rotX: 0.3,
  rotY: 0.5,
  rotSpeed: 0.002,
  dragging: false,
  panning: false,

  // Probe
  probeDir: null,  // [x,y,z] or null

  // OOD point set
  oodSetType: "far",  // "far" | "near" | "random"
};

// ─── Effect graph ────────────────────────────────────────────────
const effects = [];
const effectsByName = new Map();

export function registerEffect(name, deps, run, opts = {}) {
  const entry = { name, deps, run, after: opts.after || [] };
  effects.push(entry);
  effectsByName.set(name, entry);
}

// ─── Simple key listeners (fires after all effects) ──────────────
const simpleListeners = new Map();

export function on(key, fn) {
  if (!simpleListeners.has(key)) simpleListeners.set(key, []);
  simpleListeners.get(key).push(fn);
}

export function off(key, fn) {
  const list = simpleListeners.get(key);
  if (list) {
    const idx = list.indexOf(fn);
    if (idx >= 0) list.splice(idx, 1);
  }
}

// ─── State access ────────────────────────────────────────────────

export function get(key) {
  return state[key];
}

export function getAll() {
  return { ...state };
}

// ─── State mutation with effect propagation ──────────────────────

let _batching = false;
let _pendingUpdates = null;

export function set(updates) {
  if (_batching) {
    Object.assign(_pendingUpdates, updates);
    return;
  }

  const changed = new Set();
  for (const [k, v] of Object.entries(updates)) {
    if (state[k] !== v) {
      state[k] = v;
      changed.add(k);
    }
  }
  if (changed.size === 0) return;

  // 1. Run named effects in topological order.
  _runEffects(changed);

  // 2. Fire simple key listeners for changed keys only.
  for (const key of changed) {
    const fns = simpleListeners.get(key);
    if (!fns) continue;
    for (const fn of fns) {
      try { fn(state[key], state); } catch (e) { console.error(e); }
    }
  }
}

export function batch(fn) {
  _batching = true;
  _pendingUpdates = {};
  try {
    fn();
  } finally {
    _batching = false;
    const updates = _pendingUpdates;
    _pendingUpdates = null;
    if (Object.keys(updates).length > 0) {
      set(updates);
    }
  }
}

function _runEffects(changedKeys) {
  const ran = new Set();
  const queue = effects.filter(e =>
    e.deps.some(d => changedKeys.has(d))
  );

  // Build set of effect names actually in this batch, so `after`
  // constraints only block on effects that will actually run.
  const inBatch = new Set(queue.map(e => e.name));

  // Topological pass: run effects whose `after` deps have either
  // already run OR are not in this batch (i.e., irrelevant to this
  // state change). This prevents a currentQ-only change from
  // blocking on rebuildSortedFCScores, which only fires on
  // fcBalanced changes.
  const pending = [...queue];
  let safety = pending.length * 2;
  while (pending.length > 0 && safety-- > 0) {
    const next = pending.findIndex(e =>
      e.after.every(dep => ran.has(dep) || !inBatch.has(dep))
    );
    if (next === -1) break;  // circular — shouldn't happen
    const e = pending.splice(next, 1)[0];
    try {
      e.run(state);
    } catch (err) {
      console.error(`Effect "${e.name}" failed:`, err);
    }
    ran.add(e.name);
  }
}
