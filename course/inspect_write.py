"""Compare one clean KV write from identical inputs on the official T2V path."""
import csv
import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.causal_inference import load_sage_v1_attention
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.attention import attention


def main():
    output_csv = Path(sys.argv.pop(1)).resolve()
    sage = load_sage_v1_attention()
    original = WanDiffusionWrapper.forward
    writes = 0

    def forward(model, *args, **kwargs):
        nonlocal writes
        caches = kwargs.get("kv_cache")
        if caches is None or kwargs["timestep"].flatten()[0].item() != 0:
            return original(model, *args, **kwargs)
        writes += 1
        if writes != 2:
            return original(model, *args, **kwargs)

        start = int(kwargs["current_start"])
        end = int(caches[0]["local_end_index"].item())
        assert start > 0 and end > start
        assert all(c["local_end_index"].item() == c["global_end_index"].item() == end for c in caches)
        assert all(c["is_init"] for c in kwargs["crossattn_cache"])
        device = kwargs["noisy_image_or_video"].device
        before = [{s: c[s][:, :end].detach().cpu() for s in ("k", "v")} for c in caches]
        cpu_before, gpu_before = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        original_backends = [c.get("self_attention") for c in caches]
        baseline_output = original(model, *args, **kwargs)
        reference = [{s: c[s][:, start:end].detach().cpu() for s in ("k", "v")} for c in caches]
        cpu_after, gpu_after = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        rows = []
        try:
            for label, kernel in (("native_repeat", attention), ("sage", sage)):
                for c, saved in zip(caches, before):
                    for s in ("k", "v"):
                        c[s][:, :end].copy_(saved[s])
                    c["global_end_index"].fill_(end)
                    c["local_end_index"].fill_(end)
                    c["self_attention"] = kernel
                torch.set_rng_state(cpu_before)
                torch.cuda.set_rng_state(gpu_before, device)
                original(model, *args, **kwargs)
                for layer, (c, saved, ref) in enumerate(zip(caches, before, reference), start=1):
                    for s in ("k", "v"):
                        used = c[s][:, :end].detach().cpu()
                        error = used[:, start:end].float() - ref[s].float()
                        rows.append(dict(backend=label, layer=layer, tensor=s,
                                         history_unchanged=torch.equal(used[:, :start], saved[s][:, :start]),
                                         current_equal=torch.equal(used[:, start:end], ref[s]),
                                         relative_l2=(error.norm() / ref[s].float().norm().clamp_min(1e-12)).item(),
                                         max_abs=error.abs().max().item()))
        finally:
            for c, saved, ref, backend in zip(caches, before, reference, original_backends):
                for s in ("k", "v"):
                    c[s][:, :start].copy_(saved[s][:, :start])
                    c[s][:, start:end].copy_(ref[s])
                c["global_end_index"].fill_(end)
                c["local_end_index"].fill_(end)
                if backend is None:
                    c.pop("self_attention", None)
                else:
                    c["self_attention"] = backend
            torch.set_rng_state(cpu_after)
            torch.cuda.set_rng_state(gpu_after, device)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return baseline_output

    os.chdir(ROOT)
    torch.set_num_threads(4)
    WanDiffusionWrapper.forward = forward
    try:
        runpy.run_path(str(ROOT / "inference.py"), run_name="__main__")
    finally:
        WanDiffusionWrapper.forward = original


if __name__ == "__main__":
    main()
