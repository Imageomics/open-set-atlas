// Loads precomputed.js (which assigns to window.OOD_PRECOMPUTED)
// into Node by providing a fake window object.

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(resolve(__dirname, '..', 'precomputed.js'), 'utf-8');

const fakeWindow = {};
const fn = new Function('window', src);
fn(fakeWindow);

export const PRE = fakeWindow.OOD_PRECOMPUTED;
