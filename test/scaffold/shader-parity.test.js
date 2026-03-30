import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Module shader exports
import { vertShader } from '../../src/shaders/vert.js';
import { glslDirFromUV, glslPickColor } from '../../src/shaders/preamble.js';
import { logitFrag } from '../../src/shaders/logit-frag.js';
import { cosVmfFrag } from '../../src/shaders/cos-vmf-frag.js';
import { kentFrag } from '../../src/shaders/kent-frag.js';
import { odinFrag } from '../../src/shaders/odin-frag.js';
import { vimFrag } from '../../src/shaders/vim-frag.js';
import { rmdsFrag } from '../../src/shaders/rmds-frag.js';
import { conformalMspEmbFrag } from '../../src/shaders/conformal-msp-emb-frag.js';
import { conformalApsFrag } from '../../src/shaders/conformal-aps-frag.js';
import { scoreFieldAFrag } from '../../src/shaders/score-field-a-frag.js';
import { scoreFieldBFrag } from '../../src/shaders/score-field-b-frag.js';

// ── Extract original shader strings from the monolith ──────────
// The monolith defines shaders as JS arrays that reference _glslDirFromUV
// and _glslPickColor variables, then .join("\n"). We evaluate those
// definitions in a sandbox that provides the preamble variables.

const __dirname = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(resolve(__dirname, '..', '..', 'ood_methods_3d.html'), 'utf-8');

// Extract the module script content
const scriptMatch = html.match(/<script type="module">([\s\S]*?)<\/script>/);
const js = scriptMatch[1];

// Build the preamble variables exactly as the original monolith defined them
// (before our refactor, these were inline; now they're imported via preamble.js,
// but the monolith's shader arrays still reference them by the same variable names)
function extractMonolithShader(varName) {
  // Find the var declaration: var _shaderFragXxx = [ ... ].join("\n");
  // We need to handle multiline array literals.
  const patterns = [
    new RegExp(`var ${varName} = \\[([\\s\\S]*?)\\]\\.join\\("\\\\n"\\);`, 'm'),
    new RegExp(`var ${varName} = \\[([\\s\\S]*?)\\]\\.join\\('\\\\n'\\);`, 'm'),
  ];

  let arrayContent = null;
  for (const pat of patterns) {
    const m = js.match(pat);
    if (m) { arrayContent = m[1]; break; }
  }
  if (!arrayContent) return null;

  // Evaluate the array with _glslDirFromUV and _glslPickColor in scope
  // We also provide glslDirFromUV/glslPickColor since the refactored HTML
  // now uses the imported names in the array splices.
  const _glslDirFromUV = glslDirFromUV;
  const _glslPickColor = glslPickColor;
  try {
    const assembled = new Function(
      '_glslDirFromUV', '_glslPickColor', 'glslDirFromUV', 'glslPickColor',
      `return [${arrayContent}].join("\\n");`
    )(_glslDirFromUV, _glslPickColor, _glslDirFromUV, _glslPickColor);
    return assembled;
  } catch (e) {
    return null;
  }
}

describe('Shader string parity against monolith', () => {

  it('vertex shader matches', () => {
    const pattern = /var _shaderVert = \[([\s\S]*?)\]\.join\("\\n"\);/m;
    const m = js.match(pattern);
    assert.ok(m, '_shaderVert should exist in monolith');
    const original = new Function(`return [${m[1]}].join("\\n");`)();
    assert.equal(vertShader, original,
      'vertex shader should be identical');
  });

  it('preamble: glslDirFromUV matches', () => {
    const pattern = /var _glslDirFromUV = \[([\s\S]*?)\]\.join\("\\n"\);/m;
    const m = js.match(pattern);
    assert.ok(m, '_glslDirFromUV should exist in monolith');
    const original = new Function(`return [${m[1]}].join("\\n");`)();
    assert.equal(glslDirFromUV, original,
      'dirFromUV preamble should be identical');
  });

  it('preamble: glslPickColor matches', () => {
    const pattern = /var _glslPickColor = \[([\s\S]*?)\]\.join\("\\n"\);/m;
    const m = js.match(pattern);
    assert.ok(m, '_glslPickColor should exist in monolith');
    const original = new Function(`return [${m[1]}].join("\\n");`)();
    assert.equal(glslPickColor, original,
      'pickColor preamble should be identical');
  });

  const proceduralShaders = [
    { name: 'logit', varName: '_shaderFragLogit', moduleExport: logitFrag },
    { name: 'cos-vmf', varName: '_shaderFragCosVMF', moduleExport: cosVmfFrag },
    { name: 'kent', varName: '_shaderFragKent', moduleExport: kentFrag },
    { name: 'odin', varName: '_shaderFragODIN', moduleExport: odinFrag },
    { name: 'vim', varName: '_shaderFragViM', moduleExport: vimFrag },
    { name: 'rmds', varName: '_shaderFragRMDS', moduleExport: rmdsFrag },
    { name: 'conformal-msp-emb', varName: '_shaderFragConformalMSPEmb', moduleExport: conformalMspEmbFrag },
    { name: 'conformal-aps', varName: '_shaderFragConformalAPS', moduleExport: conformalApsFrag },
  ];

  for (const { name, varName, moduleExport } of proceduralShaders) {
    it(`procedural shader "${name}" matches monolith`, () => {
      const original = extractMonolithShader(varName);
      assert.ok(original, `${varName} should be extractable from monolith`);
      assert.ok(moduleExport, `module export for ${name} should exist`);

      if (original !== moduleExport) {
        // Find the first difference for a useful error message
        const maxLen = Math.max(original.length, moduleExport.length);
        let diffIdx = -1;
        for (let i = 0; i < maxLen; i++) {
          if (original[i] !== moduleExport[i]) { diffIdx = i; break; }
        }
        const ctx = 40;
        const origSnip = original.substring(Math.max(0, diffIdx - ctx), diffIdx + ctx);
        const modSnip = moduleExport.substring(Math.max(0, diffIdx - ctx), diffIdx + ctx);
        assert.fail(
          `${name} shader differs at char ${diffIdx}.\n` +
          `  Monolith: ...${origSnip}...\n` +
          `  Module:   ...${modSnip}...`
        );
      }
    });
  }

  const scoreFieldShaders = [
    { name: 'score-field-a', varName: '_shaderFragA', moduleExport: scoreFieldAFrag },
    { name: 'score-field-b', varName: '_shaderFragB', moduleExport: scoreFieldBFrag },
  ];

  for (const { name, varName, moduleExport } of scoreFieldShaders) {
    it(`score field shader "${name}" matches monolith`, () => {
      const original = extractMonolithShader(varName);
      assert.ok(original, `${varName} should be extractable from monolith`);
      assert.ok(moduleExport, `module export for ${name} should exist`);

      if (original !== moduleExport) {
        const maxLen = Math.max(original.length, moduleExport.length);
        let diffIdx = -1;
        for (let i = 0; i < maxLen; i++) {
          if (original[i] !== moduleExport[i]) { diffIdx = i; break; }
        }
        const ctx = 40;
        const origSnip = original.substring(Math.max(0, diffIdx - ctx), diffIdx + ctx);
        const modSnip = moduleExport.substring(Math.max(0, diffIdx - ctx), diffIdx + ctx);
        assert.fail(
          `${name} shader differs at char ${diffIdx}.\n` +
          `  Monolith: ...${origSnip}...\n` +
          `  Module:   ...${modSnip}...`
        );
      }
    });
  }
});
