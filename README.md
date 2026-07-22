<p align="center"><img src="icon.png" alt="ComfyUI-TRELLIS-HiCache" width="640"></p>

# ComfyUI-TRELLIS-HiCache

<p>
  <a href="https://github.com/Archerkattri/ComfyUI-TRELLIS-HiCache/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Archerkattri/ComfyUI-TRELLIS-HiCache?color=1f6feb"></a>
  <a href="https://registry.comfy.org/publishers/archerkattri/nodes/comfyui-trellis-hicache"><img alt="Comfy Registry" src="https://img.shields.io/badge/Comfy%20Registry-comfyui--trellis--hicache-4b8bbe"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/Archerkattri/ComfyUI-TRELLIS-HiCache?color=0d9488"></a>
</p>


Training-free acceleration for **TRELLIS** image-to-3D in ComfyUI. It forecasts
the flow-matching velocity on skipped DiT steps instead of running the
transformer, on both TRELLIS stages (sparse-structure and SLaT), via the
[`hicache-pp`](https://pypi.org/project/hicache-pp/) library.

Pairs with [smthemex/ComfyUI_TRELLIS](https://github.com/smthemex/ComfyUI_TRELLIS)
(the `MODEL_TRELLIS` pipeline type). Same idea as
[ComfyUI-HiCache](https://github.com/Archerkattri/ComfyUI-HiCache) for Hunyuan3D.

## What it does

TRELLIS samples each stage with a flow-Euler loop that calls a DiT once (or twice,
under classifier-free guidance) per step. **TRELLIS HiCache Accelerate** replaces
the two flow DiTs with a wrapper that runs the transformer on a schedule and
*forecasts* the velocity on the steps in between:

* **`hermite`** — HiCache (dual-scaled physicist's Hermite polynomial, arXiv:2508.16984).
* **`dmd`** — HiCache++ (Dynamic Mode Decomposition / Prony exponential basis).
* **`auto`** — holdout-selected per step.

Two TRELLIS-specific details are handled correctly: the timestep schedule runs
`1 -> 0` (so run boundaries are detected by direction reversal, not a fixed
threshold), and classifier-free guidance issues the conditional and unconditional
forwards *separately* inside a guidance interval, so the patch keeps two parallel
forecast states and routes each forward to the right one. The SLaT stage returns a
sparse tensor whose active-voxel layout is fixed during a run, so the forecast
runs on its `.feats` and the sparse tensor is rebuilt from the last computed step.

## Measured (RTX 5090, TRELLIS-image-large, demo image, 25+25 steps)

| config | speedup | Chamfer vs stock (unit-cube units) |
|---|---|---|
| `interval=2`, both stages | **2.1x** | 0.0059 (near-lossless) |
| `interval=3`, both stages | **2.5x** | 0.0145 (more aggressive) |

`interval=2` is the default. Chamfer is the symmetric mean nearest-neighbour
distance between the stock and accelerated Gaussian point clouds; the object spans
~1.0, so 0.0059 is ~0.6% of its extent. The gaussian *count* shifts more than the
surface does, because TRELLIS' active-voxel threshold is sensitive near the
boundary — surface fidelity is the metric that matters and is what the Chamfer
column reports.

## Install

In ComfyUI: install via the ComfyUI Manager (search "TRELLIS HiCache"), or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Archerkattri/ComfyUI-TRELLIS-HiCache
pip install hicache-pp
```

## Use

`Trellis_LoadModel  ->  TRELLIS HiCache Accelerate  ->  Trellis_Sampler`

Set `enabled = Off` to bypass and restore the stock DiTs. The node never mutates
the pipeline it is given (copy-on-patch), so a cached node output always owns its
own configuration.

## Validation

`tests/test_patch.py` unit-tests the patch logic with a dummy DiT (no ComfyUI, no
GPU). `tests/validate_gpu.py` is the end-to-end GPU check that produced the table
above (loads a real TRELLIS pipeline, applies the patch, compares geometry and
wall-clock against stock).

Apache-2.0.
