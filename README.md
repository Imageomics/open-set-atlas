# Open Set Atlas

Interactive 3D visualization of open set and out-of-distribution detection methods for building intuition on the unit sphere.

## What is this?

Training embeddings are projected onto S² (the unit sphere in R³) — every point is L2-normalized. Five synthetic classes are sampled from von Mises-Fisher distributions with varying concentration, sample size, and anisotropy. Multiple OOD detection methods are visualized as colored acceptance regions on the sphere, and their decisions on test points are compared in a live table.

## Methods implemented

**Softmax-based (Prototype and Learned FC):**
MSP, MLS, EBO (Energy), TempScale, ODIN, ViM

**Geometric / density-based (Prototype and Learned FC):**
Per-class Cosine, von Mises-Fisher, Kent (FB₅)

**Distance-based:**
Mahalanobis (per-class and shared covariance), Relative Mahalanobis (RMDS), KNN, KDE

**Conformal prediction:**
APS, 1-MSP, and Embedding cosine score functions with adjustable coverage level (α)

## Usage

Serve over HTTP (ES modules require it) and open `ood_methods_3d.html`:

```bash
npx serve .
# → http://localhost:3000/ood_methods_3d.html
```

Three.js loads from a CDN (requires internet on first load).

**Controls:**
- Click a method button to see its acceptance region on the sphere
- Drag the **q** slider (0.001 step) to adjust the threshold quantile — updates in real-time
- Drag the **α** slider (0.001 step) to adjust conformal coverage level — updates in real-time
- Press ▶ to animate q or α at 30fps across their full range
- Left-click drag to orbit, scroll to zoom, middle-click drag to pan
- Click on the sphere to probe any point (appears as ◆ in the decision table)
- Toggle between **Low-confidence** and **Boundary** test point sets
- Adjust rotation speed or reset zoom with controls below the legend

**Precomputed data:** `precomputed.js` (128 KB) provides class data, FC weights, and training scores so the browser skips sampling, FC training, and KNN distance computation. Textures are computed client-side on demand.

To regenerate precomputed data: `node precompute.js`

## Development

All JavaScript lives in `src/` as ES modules. The HTML is pure markup.

```
src/
├── main.js                 — composition root (data init, wiring)
├── config/                 — method registry, constants, class definitions
├── state/                  — reactive state, FC effect cascade, threshold engine
├── methods/                — scoring, uniforms, FC trainer, conformal logic
├── rendering/              — shader manager, orbit controls, ellipsoids
├── ui/                     — controls, decision table
├── math/                   — vec3, linalg, stats, sampling, fitting
└── shaders/                — GLSL fragment shaders as JS strings
```

```bash
node --test test/unit/*.test.js   # 190 tests
```

## License

MIT
