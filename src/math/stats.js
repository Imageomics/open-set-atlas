"use strict";

export function quantile(arr, q) {
  const xs = [...arr].sort((a,b) => a-b);
  if (xs.length <= 1) return xs[0] || 0;
  const pos = (xs.length-1)*q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return xs[lo];
  return xs[lo]*(1-(pos-lo)) + xs[hi]*(pos-lo);
}

// Fast quantile on a pre-sorted array (no copy, no sort)
export function quantileSorted(sorted, q) {
  if (sorted.length <= 1) return sorted[0] || 0;
  var pos = (sorted.length-1)*q;
  var lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo]*(1-(pos-lo)) + sorted[hi]*(pos-lo);
}

export function softmax(z) {
  const m = Math.max(...z);
  const ex = z.map(v => Math.exp(v-m));
  const s = ex.reduce((a,b) => a+b, 0);
  return ex.map(v => v/s);
}

export function logsumexp(logits) {
  const m = Math.max(...logits);
  return m + Math.log(logits.reduce((s, l) => s + Math.exp(l - m), 0));
}
