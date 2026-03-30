import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

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

// ── Shader module integrity tests ─────────────────────────────
// Phase 1 verified these modules match the monolith's inline strings.
// Phase 3 moved the monolith to use these modules directly via shader-manager.
// This test now verifies the module exports are well-formed GLSL.

describe('Shader module integrity', () => {

  it('vertex shader is a non-empty string with main()', () => {
    assert.ok(typeof vertShader === 'string' && vertShader.length > 0);
    assert.ok(vertShader.includes('void main'), 'should contain void main');
    assert.ok(vertShader.includes('gl_Position'), 'should set gl_Position');
  });

  it('preamble: glslDirFromUV is a non-empty string', () => {
    assert.ok(typeof glslDirFromUV === 'string' && glslDirFromUV.length > 0);
    assert.ok(glslDirFromUV.includes('dirFromUV'), 'should define dirFromUV');
  });

  it('preamble: glslPickColor is a non-empty string', () => {
    assert.ok(typeof glslPickColor === 'string' && glslPickColor.length > 0);
    assert.ok(glslPickColor.includes('classColors'), 'should reference classColors');
  });

  const proceduralShaders = [
    { name: 'logit', moduleExport: logitFrag, expectedUniforms: ['classW', 'threshold', 'scoreMode'] },
    { name: 'cos-vmf', moduleExport: cosVmfFrag, expectedUniforms: ['classDir', 'classKappa', 'perClassThr'] },
    { name: 'kent', moduleExport: kentFrag, expectedUniforms: ['kentGamma1', 'kentKappa', 'perClassThr'] },
    { name: 'odin', moduleExport: odinFrag, expectedUniforms: ['classW', 'odinT', 'odinEps'] },
    { name: 'vim', moduleExport: vimFrag, expectedUniforms: ['classW', 'vimU0', 'vimAlphaU'] },
    { name: 'rmds', moduleExport: rmdsFrag, expectedUniforms: ['invCov', 'classMeans', 'globalMean'] },
    { name: 'conformal-msp-emb', moduleExport: conformalMspEmbFrag, expectedUniforms: ['classW', 'threshold'] },
    { name: 'conformal-aps', moduleExport: conformalApsFrag, expectedUniforms: ['classW', 'threshold'] },
  ];

  for (const { name, moduleExport, expectedUniforms } of proceduralShaders) {
    it(`procedural shader "${name}" is well-formed`, () => {
      assert.ok(typeof moduleExport === 'string' && moduleExport.length > 50,
        `${name} should be a non-trivial string`);
      assert.ok(moduleExport.includes('void main'),
        `${name} should contain void main`);
      assert.ok(moduleExport.includes('gl_FragColor'),
        `${name} should set gl_FragColor`);
      for (const u of expectedUniforms) {
        assert.ok(moduleExport.includes(u),
          `${name} should reference uniform "${u}"`);
      }
    });
  }

  const scoreFieldShaders = [
    { name: 'score-field-a', moduleExport: scoreFieldAFrag, expectedUniforms: ['scoreMap', 'threshold'] },
    { name: 'score-field-b', moduleExport: scoreFieldBFrag, expectedUniforms: ['scores04', 'perClassThr'] },
  ];

  for (const { name, moduleExport, expectedUniforms } of scoreFieldShaders) {
    it(`score field shader "${name}" is well-formed`, () => {
      assert.ok(typeof moduleExport === 'string' && moduleExport.length > 50,
        `${name} should be a non-trivial string`);
      assert.ok(moduleExport.includes('void main'),
        `${name} should contain void main`);
      for (const u of expectedUniforms) {
        assert.ok(moduleExport.includes(u),
          `${name} should reference uniform "${u}"`);
      }
    });
  }
});
