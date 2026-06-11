"""ComfyUI node: training-free HiCache / HiCache++ acceleration for TRELLIS.

One node, no ComfyUI-internal imports: it patches the two TRELLIS flow DiTs
(sparse-structure + SLaT) so skipped sampling steps are forecast with hicache-pp
instead of running the transformer. Drop it between the TRELLIS loader
(``MODEL_TRELLIS``) and the sampler.

All acceleration logic lives in :mod:`trellis_hicache_patch` (unit-testable with
no ComfyUI / GPU); this file is only the ComfyUI plumbing.
"""
try:
    from .trellis_hicache_patch import METHODS, STAGES, apply_hicache, remove_hicache
except ImportError:  # running as a flat module (tests / standalone)
    from trellis_hicache_patch import METHODS, STAGES, apply_hicache, remove_hicache


class TrellisHiCacheAccelerate:
    """Patch a TRELLIS pipeline's flow DiTs with a HiCache velocity forecast.

    Connect a ``MODEL_TRELLIS`` from the TRELLIS loader to ``model`` and feed the
    returned pipeline to the sampler. ``enabled = Off`` removes the patch and
    restores the stock DiTs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL_TRELLIS",),
                "method": (list(METHODS), {"default": "hermite"}),
                "interval": ("INT", {"default": 2, "min": 1, "max": 10,
                                     "tooltip": "Run the DiT once every `interval` "
                                     "steps; the rest are forecast. 2 is "
                                     "near-lossless (~0.6% Chamfer), 3 is more "
                                     "aggressive (~2.5x, ~1.5% Chamfer)."}),
                "stages": (list(STAGES), {"default": "both",
                           "tooltip": "Which flow stage to accelerate: the "
                           "sparse-structure DiT, the SLaT DiT, or both."}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 10,
                                 "tooltip": "Always compute the first N steps of "
                                 "each run before forecasting begins."}),
                "enabled": ("BOOLEAN", {"default": True,
                            "tooltip": "Off removes the patch and restores the "
                            "stock TRELLIS DiTs."}),
            },
            "optional": {
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),
                "dmd_history": ("INT", {"default": 5, "min": 3, "max": 16}),
                "max_order": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
        }

    RETURN_TYPES = ("MODEL_TRELLIS",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "TRELLIS/HiCache"
    DESCRIPTION = ("Training-free acceleration for TRELLIS image-to-3D: forecast "
                   "the flow-matching velocity on skipped DiT steps (HiCache "
                   "Hermite / HiCache++ DMD, via the hicache-pp library).")

    def patch(self, model, method="hermite", interval=2, stages="both",
              warmup_steps=2, enabled=True, sigma=0.5, dmd_history=5, max_order=1):
        if not enabled:
            return (remove_hicache(model),)
        return (apply_hicache(
            model, method=method, interval=interval, warmup_steps=warmup_steps,
            max_order=max_order, sigma=sigma, dmd_history=dmd_history, stages=stages,
        ),)


NODE_CLASS_MAPPINGS = {"TrellisHiCacheAccelerate": TrellisHiCacheAccelerate}
NODE_DISPLAY_NAME_MAPPINGS = {"TrellisHiCacheAccelerate": "TRELLIS HiCache Accelerate"}
