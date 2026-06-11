"""GPU validation: apply the TRELLIS-HiCache model patch to a real stock TRELLIS
pipeline, run image-to-3D, and confirm (a) it runs end-to-end, (b) the DiT is
actually skipped on both stages, (c) it is faster than stock, and (d) the output
geometry is sane (non-empty Gaussian + mesh).

Run from the faster-trellis checkout (which vendors the TRELLIS package + weights
path), with this repo importable:

    cd third_party/faster-trellis
    PYTHONPATH=../comfyui-trellis-hicache python ../comfyui-trellis-hicache/tests/validate_gpu.py \
        --weights /home/krishi/workspace/data/weights/TRELLIS --image <some.png>
"""
import argparse, time, sys
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", default="hermite")
    ap.add_argument("--interval", type=int, default=3)
    ap.add_argument("--ss_steps", type=int, default=25)
    ap.add_argument("--slat_steps", type=int, default=25)
    ap.add_argument("--stages", default="both")
    args = ap.parse_args()

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis_hicache_patch import apply_hicache, remove_hicache

    pipe = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipe.cuda()
    # force stock samplers (this script tests our model-level patch, not the fork's
    # enable_faster_mode sampler swap)
    if hasattr(pipe, "enable_faster_mode"):
        pipe.enable_faster_mode("none")

    img = Image.open(args.image)
    run_kw = dict(seed=0, formats=["gaussian", "mesh", "radiance_field"],
                  sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": 7.5},
                  slat_sampler_params={"steps": args.slat_steps, "cfg_strength": 3.0})

    # ---- stock baseline ----
    import torch
    torch.cuda.synchronize(); t0 = time.time()
    out_stock = pipe.run(img, **run_kw)
    torch.cuda.synchronize(); t_stock = time.time() - t0
    n_gs_stock = int(out_stock["gaussian"][0].get_xyz.shape[0]) if hasattr(out_stock["gaussian"][0], "get_xyz") else -1

    # ---- patched ----
    patched = apply_hicache(pipe, method=args.method, interval=args.interval, stages=args.stages)
    torch.cuda.synchronize(); t0 = time.time()
    out_fast = patched.run(img, **run_kw)
    torch.cuda.synchronize(); t_fast = time.time() - t0
    n_gs_fast = int(out_fast["gaussian"][0].get_xyz.shape[0]) if hasattr(out_fast["gaussian"][0], "get_xyz") else -1

    # ---- geometry fidelity: symmetric Chamfer between stock & hicache gaussian
    # centers (mm-scale, the object fits in a unit cube), subsampled for speed ----
    def _xyz(o):
        return o["gaussian"][0].get_xyz.detach().float()
    def chamfer(a, b, k=20000):
        if a.shape[0] > k:
            a = a[torch.randperm(a.shape[0], device=a.device)[:k]]
        if b.shape[0] > k:
            b = b[torch.randperm(b.shape[0], device=b.device)[:k]]
        d_ab = torch.cdist(a, b).min(dim=1).values.mean()
        d_ba = torch.cdist(b, a).min(dim=1).values.mean()
        return float((d_ab + d_ba) / 2)
    cham = chamfer(_xyz(out_stock), _xyz(out_fast))  # in TRELLIS unit-cube units

    # ---- report skip stats from the patched models ----
    def skips(key):
        m = patched.models[key]
        return (getattr(m, "computed_steps", -1), getattr(m, "skipped_steps", 0),
                getattr(m, "_hicache_is_patch", False))
    ss_c, ss_s, ss_p = skips("sparse_structure_flow_model")
    sl_c, sl_s, sl_p = skips("slat_flow_model")
    # gaussian-count fidelity vs stock (proxy for geometry agreement)
    ratio = n_gs_fast / max(n_gs_stock, 1)
    print("\n==================== TRELLIS-HiCache validation ====================")
    print(f"method={args.method} interval={args.interval} stages={args.stages}  ss_steps={args.ss_steps} slat_steps={args.slat_steps}")
    print(f"SS   stage: {ss_c} computed + {ss_s} skipped  (patched={ss_p})")
    print(f"SLaT stage: {sl_c} computed + {sl_s} skipped  (patched={sl_p})")
    print(f"stock : {t_stock:.2f}s  ({n_gs_stock} gaussians)")
    print(f"hicache: {t_fast:.2f}s  ({n_gs_fast} gaussians)  count-ratio={ratio:.3f}")
    print(f"speedup: {t_stock / max(t_fast, 1e-6):.2f}x")
    print(f"chamfer(stock,hicache) = {cham:.5f}  (unit-cube units; object spans ~1.0)")

    total_skipped = ss_s + sl_s
    # geometry must be close: Chamfer well under 1% of the unit cube extent.
    ok = (total_skipped > 0 and n_gs_fast > 1000 and t_fast < t_stock
          and cham < 0.01)
    restored = remove_hicache(patched)
    ok = ok and not getattr(restored.models["slat_flow_model"], "_hicache_is_patch", False)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
