# Open Set Atlas

Interactive 3D visualization of open set and out-of-distribution detection methods for building intuition on the unit sphere.

> **Note:** This project is in a state of rapid development. Expect breaking changes, incomplete features, and rough edges.

> **Performance:** This visualization is computationally demanding. Method textures are rendered at 2048x1024 on the client, and background precomputation runs continuously until all methods are cached. Expect high CPU usage during initial loading. Avoid opening multiple tabs simultaneously.

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

Open `ood_methods_3d.html` in a browser. Three.js loads from a CDN (requires internet on first load).

**Controls:**
- Click a method button to see its acceptance region on the sphere
- Drag the **q** slider to adjust the threshold quantile (applies to all non-conformal methods)
- Drag the **α** slider to adjust conformal coverage level
- Left-click drag to orbit, scroll to zoom, middle-click drag to pan
- Click on the sphere to probe any point (appears as ◆ in the decision table)
- Toggle between **Low-confidence** and **Boundary** test point sets
- Adjust rotation speed or reset zoom with controls below the legend

**Precomputed data:** `precomputed.js` (128 KB) provides class data, FC weights, and training scores so the browser skips sampling, FC training, and KNN distance computation. Textures are computed client-side on demand.

To regenerate precomputed data: `node precompute.js`

## Repository contents

| File | Description |
|------|-------------|
| `ood_methods_3d.html` | The visualization (single-file app) |
| `precomputed.js` | Precomputed class data, thresholds, FC weights (128 KB) |
| `precompute.js` | Node.js script to regenerate `precomputed.js` |
| `TODO.md` | Roadmap and notes on approaches tried |

## License

MIT
