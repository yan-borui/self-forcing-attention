"""Summarize the paired Self Forcing inference runs on the experiment host."""
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

POLICIES = ("native", "probe", "sage", "native_write", "native_last_denoise")
TIMING = re.compile(r"prompt=(\d+) pipeline_seconds=([\d.]+) peak_allocated_mib=([\d.]+) peak_reserved_mib=([\d.]+)")
OPERATOR = re.compile(r"prompt=(\d+) role=(\w+) attention_calls=(\d+) self_attention_ms=([\d.]+)")


def video_path(study, role, prompt):
    return study / role / "videos" / f"{prompt}-0_ema.mp4"


def frames(filename):
    with av.open(str(filename)) as handle:
        for frame in handle.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def video_metrics(reference, candidate):
    from itertools import zip_longest

    differences, motion = [], []
    same = True
    previous = None
    for ref, image in zip_longest(frames(reference), frames(candidate)):
        if ref is None or image is None or ref.shape != image.shape:
            raise ValueError(f"Frame count or geometry differs: {candidate}")
        delta = np.abs(np.subtract(image, ref, dtype=np.int16))
        differences.append(float(delta.mean()) / 255)
        same = same and not delta.any()
        if previous is not None:
            motion.append(float(np.abs(np.subtract(image, previous, dtype=np.int16)).mean()) / 255)
        previous = image
    if len(differences) != 81:
        raise ValueError(f"Expected 81 frames in {candidate}, got {len(differences)}")
    return dict(frames=len(differences), same_decoded_frames=bool(same),
                rgb_mae=float(np.mean(differences)), last24_rgb_mae=float(np.mean(differences[-24:])),
                adjacent_rgb_change=float(np.mean(motion)))


def latents(study, role, prompt):
    output = torch.load(video_path(study, role, prompt).with_suffix(".pt"), map_location="cpu", weights_only=True)
    if output.shape != (21, 16, 60, 104):
        raise ValueError(f"Unexpected output latent shape: {output.shape}")
    return output.float()


def chunk_metrics(reference, candidate, prompt, policy):
    rows = []
    for chunk, (ref, new) in enumerate(zip(reference.split(3), candidate.split(3))):
        diff = new - ref
        rows.append(dict(prompt=prompt, policy=policy, chunk=chunk + 1, equal=torch.equal(ref, new),
                         relative_l2=(diff.norm() / ref.norm().clamp_min(1e-12)).item(),
                         max_abs=diff.abs().max().item()))
    return rows


def write_csv(filename, rows):
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(study, out):
    display = (("native", "Native BF16"), ("sage", "Sage"), ("native_write", "BF16 on write"),
               ("native_last_denoise", "BF16 on last denoise"))
    sample_frames = (0, 40, 80)
    width, height, margin = 312, 180, 138
    sheet = Image.new("RGB", (margin + len(sample_frames) * width, 28 + len(display) * height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, frame in enumerate(sample_frames):
        draw.text((margin + col * width + 5, 8), f"Frame {frame+1}", fill="black")
    for row, (role, label) in enumerate(display):
        y = 28 + row * height
        draw.text((5, y + height // 2), label, fill="black")
        for index, pixels in enumerate(frames(video_path(study, role, 0))):
            if index in sample_frames:
                preview = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
                sheet.paste(preview, (margin + sample_frames.index(index) * width, y))
    sheet.save(out / "tokyo-frames.jpg", quality=91)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    out = study / "summary"
    out.mkdir(exist_ok=True)
    prompt_count = len((study / "prompts.txt").read_text(encoding="utf-8").splitlines())
    torch.set_num_threads(2)
    results, chunks = [], []
    for policy in POLICIES:
        timings = {int(i): (float(t), float(a), float(r)) for i, t, a, r in
                   TIMING.findall((study / policy / "inference.log").read_text(errors="replace"))}
        if len(timings) != prompt_count:
            raise ValueError(f"Missing timing for {policy}")
        for idx in range(prompt_count):
            reference = latents(study, "native", idx)
            chunks.extend(chunk_metrics(reference, latents(study, policy, idx), idx, policy))
            t, allocated, reserved = timings[idx]
            measures = video_metrics(video_path(study, "native", idx), video_path(study, policy, idx))
            results.append(dict(prompt=idx, policy=policy, seconds=t, peak_allocated_mib=allocated,
                                peak_reserved_mib=reserved, **measures))

    write_csv(out / "videos.csv", results)
    write_csv(out / "chunks.csv", chunks)
    official = video_metrics(video_path(study, "official", 0), video_path(study, "native", 0))
    repeat = chunk_metrics(latents(study, "native", 0), latents(study, "native_repeat", 0), 0, "native_repeat")
    regular = [p for p in POLICIES if p != "probe"]
    warm = {p: [r for r in results if r["policy"] == p and r["prompt"] > 0] for p in regular}
    reference_seconds = median(r["seconds"] for r in warm["native"])
    timing = {}
    for p, rows in warm.items():
        seconds = median(r["seconds"] for r in rows)
        timing[p] = dict(warm_median_s=seconds, speedup=reference_seconds / seconds,
                         max_allocated_mib=max(r["peak_allocated_mib"] for r in rows),
                         median_last24_rgb_mae=median(r["last24_rgb_mae"] for r in results if r["policy"] == p),
                         median_last_chunk_relative_l2=median(r["relative_l2"] for r in chunks if r["policy"] == p and r["chunk"] == 7))
    probe = [row for row in chunks if row["policy"] == "probe"]
    operator_rows = []
    for role in ("profile_native", "profile_sage"):
        log = (study / role / "inference.log").read_text(errors="replace")
        for prompt, phase, calls, ms in OPERATOR.findall(log):
            operator_rows.append(dict(policy=role, prompt=int(prompt), phase=phase, calls=int(calls), gpu_ms=float(ms)))
    if len(operator_rows) != 12:
        raise ValueError(f"Incomplete operator timing: {len(operator_rows)} rows")
    write_csv(out / "attention.csv", operator_rows)
    summary = dict(prompts=prompt_count, warm_ids=list(range(1, prompt_count)), official=official,
                   native_repeat_all_equal=all(r["equal"] for r in repeat),
                   native_repeat_max_relative_l2=max(r["relative_l2"] for r in repeat),
                   probe_first_two_equal=all(r["equal"] for r in probe if r["chunk"] <= 2),
                   probe_by_chunk=[dict(chunk=i, median_relative_l2=median(r["relative_l2"] for r in probe if r["chunk"] == i))
                                   for i in range(1, 8)], policies=timing)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    contact_sheet(study, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
