// ═══════════════════════════════════════════════════════════════════
//  UI controls: setActive, q/alpha sliders, play buttons, FC toggle,
//  CP score radio, track bar, loading spinner.
// ═══════════════════════════════════════════════════════════════════

import * as AppState from '../state/app-state.js';
import { getCachedThresholds } from '../state/threshold-engine.js';

let _d; // deps

export function initControls(deps) {
  _d = deps;
  _wireButtons();
  _wireFCToggle();
  _wireCPScoreRadio();
  _wireAlphaSlider();
  _wireQSlider();
  _wirePlayButtons();
}

// ─── setActive ──────────────────────────────────────────────────

let _qPlaying = false, _qPlayTimer = 0;
let _alphaPlaying = false, _alphaPlayTimer = 0;

export function setActive(mode) {
  _d.setActiveMode(mode);
  AppState.set({ activeMode: mode });

  _d.layerKeys.forEach(k => { _d.layers[k].visible = (k === mode); });
  _d.descEl.textContent = _d.getDescs()[mode] || _d.getDescs().none;

  const isFC = _d.fcModes.has(mode) || mode === "cp-fc";
  _d.protoMarkers.visible = !isFC;
  _d.fcMarkers.visible = isFC;
  _d.fcBalancedLabel.style.opacity = isFC ? "1" : "0.3";
  _d.fcBalancedLabel.style.pointerEvents = isFC ? "auto" : "none";

  const isCP = _d.cpModes.has(mode);
  _d.cpSettingsEl.style.opacity = isCP ? "1" : "0.3";
  _d.cpSettingsEl.style.pointerEvents = isCP ? "auto" : "none";
  const qGroup = document.getElementById("q-slider-group");
  qGroup.style.opacity = isCP ? "0.3" : "1";
  qGroup.style.pointerEvents = isCP ? "none" : "auto";

  _d.qHideSpinner();
  _d.cpHideSpinner();

  if (isCP) _d.drawConformal(mode);
  if (!isCP && mode !== "mds" && mode !== "mds-s" && mode !== "none") {
    _d.showMethodAtQ(mode, _d.getCurrentQ());
  }

  // Reset play buttons
  var qpb = document.getElementById("q-play");
  if (qpb) { qpb.style.display = "none"; _qPlaying = false; clearTimeout(_qPlayTimer); qpb.innerHTML = "\u25B6"; }
  var apb = document.getElementById("alpha-play");
  if (apb) { apb.style.display = "none"; _alphaPlaying = false; clearTimeout(_alphaPlayTimer); apb.innerHTML = "\u25B6"; }

  qUpdateTrackBar();
  _d.cpUpdateCacheBar();
  _d.bgRestart();

  _d.allIds.forEach(id => {
    const btn = document.getElementById("b-" + id);
    if (btn) btn.classList.toggle("active", id === mode);
  });
  _d.updateDecisionTable(mode);
}

// ─── Button wiring ──────────────────────────────────────────────

function _wireButtons() {
  _d.allIds.forEach(id => {
    const btn = document.getElementById("b-" + id);
    if (btn) btn.onclick = () => setActive(id);
  });
}

// ─── FC toggle ──────────────────────────────────────────────────

function _wireFCToggle() {
  document.getElementById("fc-balanced-toggle").addEventListener("change", function() {
    _d.setFCBalanced(this.checked);
    AppState.set({ fcBalanced: this.checked });
    if (_d.fcModes.has(_d.getActiveMode())) setActive(_d.getActiveMode());
    if (_d.cpModes.has(_d.getActiveMode())) setActive(_d.getActiveMode());
  });
}

// ─── CP score radio ─────────────────────────────────────────────

function _wireCPScoreRadio() {
  document.querySelectorAll('input[name="cp-score"]').forEach(radio => {
    radio.addEventListener("change", function() {
      _d.setCpScoreType(this.value);
      AppState.set({ cpScoreType: this.value });
      _d.updateCPDescs();
      if (_d.cpModes.has(_d.getActiveMode())) setActive(_d.getActiveMode());
    });
  });
}

// ─── Alpha slider ───────────────────────────────────────────────

function _wireAlphaSlider() {
  const slider = document.getElementById("cp-alpha");
  const valInput = document.getElementById("cp-alpha-val");
  if (!slider || !valInput) return;

  function apply(val) {
    val = Math.round(val * 1000) / 1000;
    val = Math.max(0.001, Math.min(0.30, val));
    _d.setCpAlpha(val);
    AppState.set({ cpAlpha: val });
    slider.value = val;
    valInput.value = val.toFixed(3);
    _d.updateCPDescs();
    if (_d.cpModes.has(_d.getActiveMode())) {
      _d.updateConformalUniforms(_d.getActiveMode());
      _d.updateDecisionTable(_d.getActiveMode());
      _d.descEl.textContent = _d.getDescs()[_d.getActiveMode()] || _d.getDescs().none;
    }
  }
  _d.cpApplyAlpha = apply; // expose for play button

  slider.addEventListener("input", function() { valInput.value = parseFloat(this.value).toFixed(3); apply(parseFloat(this.value)); });
  slider.addEventListener("change", function() { apply(parseFloat(this.value)); });
  valInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") this.blur();
    if (e.key === "Escape") { this.value = _d.getCpAlpha().toFixed(3); this.blur(); }
  });
  valInput.addEventListener("blur", function() {
    const v = parseFloat(this.value);
    if (isNaN(v)) { this.value = _d.getCpAlpha().toFixed(3); return; }
    apply(v);
  });
}

// ─── Q slider ───────────────────────────────────────────────────

function _wireQSlider() {
  const slider = document.getElementById("q-slider");
  const valInput = document.getElementById("q-val");
  if (!slider || !valInput) return;

  function apply(val) {
    val = Math.round(val * 1000) / 1000;
    val = Math.max(0.001, Math.min(0.30, val));
    _d.setCurrentQ(val);
    slider.value = val;
    valInput.value = val.toFixed(3);
    AppState.set({ currentQ: val });
    _d.setCachedThr(getCachedThresholds());
    _d.updateQDescs();
    _d.updateFCDescs();
    if (_d.getActiveMode() !== "none" && !_d.cpModes.has(_d.getActiveMode()) &&
        _d.getActiveMode() !== "mds" && _d.getActiveMode() !== "mds-s") {
      _d.showMethodAtQ(_d.getActiveMode(), val);
    }
    _d.updateDecisionTable(_d.getActiveMode());
    _d.descEl.textContent = _d.getDescs()[_d.getActiveMode()] || _d.getDescs().none;
  }
  _d.applyQ = apply; // expose for play button

  slider.addEventListener("input", function() { valInput.value = parseFloat(this.value).toFixed(3); apply(parseFloat(this.value)); });
  slider.addEventListener("change", function() { apply(parseFloat(this.value)); });
  valInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") this.blur();
    if (e.key === "Escape") { this.value = _d.getCurrentQ().toFixed(3); this.blur(); }
  });
  valInput.addEventListener("blur", function() {
    const v = parseFloat(this.value);
    if (isNaN(v)) { this.value = _d.getCurrentQ().toFixed(3); return; }
    apply(v);
  });
}

// ─── Play buttons ───────────────────────────────────────────────

function _wirePlayButtons() {
  const qPlayBtn = document.getElementById("q-play");
  const alphaPlayBtn = document.getElementById("alpha-play");

  if (qPlayBtn) {
    qPlayBtn.addEventListener("click", function() {
      if (_qPlaying) {
        _qPlaying = false; clearTimeout(_qPlayTimer);
        this.innerHTML = "\u25B6"; return;
      }
      _qPlaying = true; this.innerHTML = "\u25A0";
      const slider = document.getElementById("q-slider");
      function step() {
        if (!_qPlaying) return;
        let v = parseFloat(slider.value) + 0.001;
        if (v > 0.30) v = 0.01;
        _d.applyQ(v);
        _qPlayTimer = setTimeout(step, 33);
      }
      step();
    });
  }

  if (alphaPlayBtn) {
    alphaPlayBtn.addEventListener("click", function() {
      if (_alphaPlaying) {
        _alphaPlaying = false; clearTimeout(_alphaPlayTimer);
        this.innerHTML = "\u25B6"; return;
      }
      _alphaPlaying = true; this.innerHTML = "\u25A0";
      const slider = document.getElementById("cp-alpha");
      function step() {
        if (!_alphaPlaying) return;
        let v = parseFloat(slider.value) + 0.001;
        if (v > 0.30) v = 0.01;
        _d.cpApplyAlpha(v);
        _alphaPlayTimer = setTimeout(step, 33);
      }
      step();
    });
  }
}

// ─── Track bar / loading state ──────────────────────────────────

function qSetTrackStyle(bg) {
  const slider = document.getElementById("q-slider");
  if (slider) slider.style.setProperty("--track-bg", bg);
}

export function qUpdateTrackBar() {
  const mode = _d.getActiveMode();
  if (!mode || mode === "none" || mode === "mds" || mode === "mds-s" ||
      _d.cpModes.has(mode)) {
    qSetTrackStyle("#dde0e4");
    return;
  }
  if (_d.isScoreFieldReady(mode)) {
    qSetTrackStyle("#b0b5bc");
    document.getElementById("q-play").style.display = "";
  } else {
    qSetTrackStyle("#dde0e4");
  }
}

export function qShowSpinner() {
  const el = document.getElementById("q-header-label");
  if (el) el.innerHTML = '<span class="cp-spinner"></span>';
}

export function qHideSpinner() {
  const el = document.getElementById("q-header-label");
  if (el) el.textContent = "q";
}
