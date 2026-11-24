"""Evaluate two official VBench dimensions on the pilot prompts, on the remote GPU."""
import argparse
import csv
import json
from pathlib import Path

import torch
from easydict import EasyDict
from vbench.dynamic_degree import DynamicDegree, dynamic_degree
from vbench.subject_consistency import subject_consistency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    assets = Path(__file__).resolve().parents[1] / ".course" / "eval"
    policies = ("native", "sage", "native_write", "native_last_denoise")
    prompt_count = len((study / "prompts.txt").read_text().splitlines())
    inputs = [(policy, index, str(study / policy / "videos" / f"{index}-0_ema.mp4"))
              for policy in policies for index in range(prompt_count)]
    clips = [filename for _, _, filename in inputs]
    device = torch.device("cuda:0")
    model = torch.hub.load(str(assets / "dino"), "dino_vitb16", source="local", pretrained=False)
    state = torch.load(assets / "weights" / "dino_vitbase16_pretrain.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(device)
    _, subjects = subject_consistency(model, clips, device, read_frame=False)
    del model
    torch.cuda.empty_cache()
    options = EasyDict(model=str(assets / "weights" / "raft-things.pth"), small=False,
                       mixed_precision=False, alternate_corr=False)
    _, motions = dynamic_degree(DynamicDegree(options, device), clips)
    per_subject = {r["video_path"]: float(r["video_results"]) for r in subjects}
    per_motion = {r["video_path"]: bool(r["video_results"]) for r in motions}
    rows = [dict(policy=p, prompt=i, subject_consistency=per_subject[f], dynamic_degree=per_motion[f])
            for p, i, f in inputs]
    out = study / "summary"
    out.mkdir(exist_ok=True)
    with (out / "vbench.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for p in policies:
        selected = [r for r in rows if r["policy"] == p]
        summary[p] = {k: sum(r[k] for r in selected) / len(selected)
                      for k in ("subject_consistency", "dynamic_degree")}
    (out / "vbench.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
