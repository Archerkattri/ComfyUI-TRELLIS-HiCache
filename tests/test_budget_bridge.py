"""Integration checks for TRELLIS branch-local budget manifests."""

import pathlib
import sys

import torch

PACK_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))

from trellis_hicache_patch import HiCacheModelPatch


class _DiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("base", torch.ones(1, 2))

    def forward(self, x, timestep, cond=None, **kwargs):
        return self.base * (1.0 + float(timestep))


def test_horizon_cap_is_counted_per_cfg_branch():
    patch = HiCacheModelPatch(
        _DiT(), method="hermite", interval=4, warmup_steps=1,
        max_horizon=1, stage_id="slat_flow_model",
    )
    for timestep in [1000.0, 900.0, 800.0, 700.0, 600.0]:
        patch(torch.zeros(1, 2), timestep)
        patch(torch.zeros(1, 2), timestep)
    report = patch.budget_manifest
    assert report["schema"] == "hicache-pp.trellis-budget-manifest.v1"
    assert set(report["branches"]) == {"cond", "uncond"}
    assert report["counts"]["fallback"] > 0
    assert report["counts"]["full"] + report["counts"]["forecast"] + report["counts"]["fallback"] == 10

