import { describe, it, beforeEach } from 'node:test';
import { strict as assert } from 'node:assert';

// Import state modules — effect registrations happen at import time
import * as AppState from '../../src/state/app-state.js';
import { bindFCEffects } from '../../src/state/fc-effects.js';
import { initSortedScores, getCachedThresholds } from '../../src/state/threshold-engine.js';

describe('AppState: basic operations', () => {
  it('get returns initial state values', () => {
    assert.equal(AppState.get('currentQ'), 0.05);
    assert.equal(AppState.get('fcBalanced'), true);
    assert.equal(AppState.get('activeMode'), 'none');
    assert.equal(AppState.get('cpAlpha'), 0.10);
    assert.equal(AppState.get('cpScoreType'), 'aps');
    assert.equal(AppState.get('probeDir'), null);
    assert.equal(AppState.get('oodSetType'), 'far');
  });

  it('set updates state and returns changed keys via listener', () => {
    const changes = [];
    const listener = (val) => changes.push(val);
    AppState.on('currentQ', listener);
    AppState.set({ currentQ: 0.10 });
    assert.equal(AppState.get('currentQ'), 0.10);
    assert.equal(changes.length, 1);
    assert.equal(changes[0], 0.10);
    // Reset
    AppState.set({ currentQ: 0.05 });
    AppState.off('currentQ', listener);
  });

  it('set does not fire listener if value unchanged', () => {
    let fired = false;
    const listener = () => { fired = true; };
    AppState.on('activeMode', listener);
    AppState.set({ activeMode: 'none' }); // same as initial
    assert.equal(fired, false);
    AppState.off('activeMode', listener);
  });

  it('batch combines multiple sets into one effect pass', () => {
    const changes = [];
    AppState.on('currentQ', (val) => changes.push(['q', val]));
    AppState.on('cpAlpha', (val) => changes.push(['a', val]));

    AppState.batch(() => {
      AppState.set({ currentQ: 0.15 });
      AppState.set({ cpAlpha: 0.20 });
    });

    assert.equal(AppState.get('currentQ'), 0.15);
    assert.equal(AppState.get('cpAlpha'), 0.20);
    assert.equal(changes.length, 2);

    // Reset
    AppState.set({ currentQ: 0.05 });
    AppState.set({ cpAlpha: 0.10 });
    AppState.off('currentQ', changes[0]);
    AppState.off('cpAlpha', changes[1]);
  });
});

describe('FC-toggle effect ordering', () => {
  it('effects fire in correct topological order on fcBalanced change', () => {
    const order = [];

    bindFCEffects({
      switchVariant: () => order.push('switchVariant'),
      rebuildMarkers: () => order.push('rebuildMarkers'),
      invalidateProceduralCache: () => order.push('invalidateProceduralCache'),
      invalidateConformalCache: () => order.push('invalidateConformalCache'),
      clearCalibScores: () => order.push('clearCalibScores'),
      recomputeOdinVim: () => order.push('recomputeOdinVim'),
      rebuildSortedScores: () => order.push('rebuildSortedScores'),
      updateDescriptions: () => order.push('updateDescriptions'),
      rerenderActive: () => order.push('rerenderActive'),
    });

    AppState.set({ fcBalanced: false });

    // switchVariant must be first
    assert.equal(order[0], 'switchVariant');

    // rebuildMarkers, invalidateProceduralCache, invalidateConformalCache,
    // clearCalibScores must be after switchVariant
    const afterSwitch = ['rebuildMarkers', 'invalidateProceduralCache',
      'invalidateConformalCache', 'clearCalibScores'];
    for (const name of afterSwitch) {
      assert.ok(order.indexOf(name) > order.indexOf('switchVariant'),
        `${name} should fire after switchVariant`);
    }

    // recomputeOdinVim must be after switchVariant
    assert.ok(order.indexOf('recomputeOdinVim') > order.indexOf('switchVariant'));

    // rebuildSortedScores must be after recomputeOdinVim
    assert.ok(order.indexOf('rebuildSortedScores') > order.indexOf('recomputeOdinVim'));

    // updateDescriptions must be after rebuildSortedScores
    assert.ok(order.indexOf('updateDescriptions') > order.indexOf('rebuildSortedScores'));

    // rerenderActive must be after updateDescriptions
    assert.ok(order.indexOf('rerenderActive') > order.indexOf('updateDescriptions'));

    // Reset state
    AppState.set({ fcBalanced: true });
    // Restore empty callbacks
    bindFCEffects({});
  });
});

describe('Threshold recomputation via effects', () => {
  it('recomputeThresholds effect fires on currentQ change', () => {
    // Create minimal sorted arrays (5 classes, small arrays)
    const makeArr = (n, base) => Array.from({ length: n }, (_, i) => base + i * 0.01).sort((a, b) => a - b);
    const proto = {
      trainMsp: [makeArr(10, 0.5), makeArr(10, 0.4), makeArr(10, 0.6), makeArr(10, 0.3), makeArr(10, 0.55)],
      mlsScores: makeArr(50, 1.0),
      eboScores: makeArr(50, 0.5),
      tsMsp: [makeArr(10, 0.5), makeArr(10, 0.4), makeArr(10, 0.6), makeArr(10, 0.3), makeArr(10, 0.55)],
      cosScores: [makeArr(10, 0.8), makeArr(10, 0.7), makeArr(10, 0.9), makeArr(10, 0.6), makeArr(10, 0.75)],
      vmScores: [makeArr(10, 5), makeArr(10, 3), makeArr(10, 6), makeArr(10, 2), makeArr(10, 4)],
      kentScores: [makeArr(10, 5), makeArr(10, 3), makeArr(10, 6), makeArr(10, 2), makeArr(10, 4)],
      mahalPC: makeArr(50, 0.1),
      mahalS: makeArr(50, 0.1),
      knnDists: makeArr(50, 0.01),
      rmds: makeArr(50, -1),
      kdeScores: makeArr(50, 5),
      odinPerClass: [makeArr(10, 0.5), makeArr(10, 0.4), makeArr(10, 0.6), makeArr(10, 0.3), makeArr(10, 0.55)],
      vimScores: makeArr(50, 0.5),
    };
    const fc = {
      fcMspPerClass: [makeArr(10, 0.5), makeArr(10, 0.4), makeArr(10, 0.6), makeArr(10, 0.3), makeArr(10, 0.55)],
      fcMlsScores: makeArr(50, 1.0),
      fcEboScores: makeArr(50, 0.5),
      fcTsPerClass: [makeArr(10, 0.5), makeArr(10, 0.4), makeArr(10, 0.6), makeArr(10, 0.3), makeArr(10, 0.55)],
      fcCosScores: [makeArr(10, 0.8), makeArr(10, 0.7), makeArr(10, 0.9), makeArr(10, 0.6), makeArr(10, 0.75)],
      fcVmScores: [makeArr(10, 5), makeArr(10, 3), makeArr(10, 6), makeArr(10, 2), makeArr(10, 4)],
      fcKentScores: [makeArr(10, 5), makeArr(10, 3), makeArr(10, 6), makeArr(10, 2), makeArr(10, 4)],
      fcOdinSorted: [makeArr(10, 0.5), makeArr(10, 0.4), makeArr(10, 0.6), makeArr(10, 0.3), makeArr(10, 0.55)],
      fcVimSorted: makeArr(50, 0.5),
    };

    initSortedScores(proto, fc);

    // Trigger threshold recomputation
    AppState.set({ currentQ: 0.10 });
    const thr = getCachedThresholds();

    assert.ok(thr, 'cachedThr should be computed after currentQ change');
    assert.ok(typeof thr.mspGamma === 'number', 'should have mspGamma');
    assert.ok(typeof thr.mlsThr === 'number', 'should have mlsThr');
    assert.ok(typeof thr.fcMspGamma === 'number', 'should have fcMspGamma');
    assert.ok(!isNaN(thr.mspGamma), 'mspGamma should not be NaN');

    // Reset
    AppState.set({ currentQ: 0.05 });
  });
});
