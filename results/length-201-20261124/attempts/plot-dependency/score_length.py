"""Paired analysis of the 201-frame rolling-window experiment."""
import argparse
import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path

import av
import cv2
import lpips
import numpy as np
import torch
from easydict import EasyDict
from PIL import Image, ImageDraw
from torch.nn import functional as F
from vbench.dynamic_degree import DynamicDegree, dynamic_degree
from vbench.subject_consistency import subject_consistency

from summarize_pilot import TIMING, chunk_metrics, frames, video_path


ROLES = ("native", "sage", "native_write", "native_last_denoise")
LABELS = ("Native BF16", "Full Sage", "BF16 on write", "BF16 on last denoise")
SCENES = ("tokyo", "mammoths", "astronaut", "coast", "rally", "table-tennis")
N_FRAMES = 201
ENDPOINTS = (81, 153, 201)


def write_rows(dest, rows):
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def models(assets, device):
    dino = torch.hub.load(str(assets / "dino"), "dino_vitb16", source="local", pretrained=False)
    state = torch.load(assets / "weights/dino_vitbase16_pretrain.pth", map_location="cpu", weights_only=True)
    dino.load_state_dict(state, strict=True)
    opts = EasyDict(model=str(assets / "weights/raft-things.pth"), small=False,
                    mixed_precision=False, alternate_corr=False)
    motion = DynamicDegree(opts, device)
    # Loading under vbench preserves the release file's relative imports.
    spec = importlib.util.spec_from_file_location("vbench.imaging_quality", assets / "length/imaging_quality.py")
    imaging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imaging)
    musiq = imaging.MUSIQ(pretrained_model_path=str(assets / "length/weights/musiq_spaq_ckpt-358bb6af.pth"))
    alex = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    return dino.eval().to(device), motion, imaging, musiq.eval().to(device), alex.eval().to(device)


def vbench_scores(filename, device, nets):
    dino, motion, imaging, musiq, _ = nets
    features, image_scores = [], []
    def save_features(_model, _inputs, result):
        features.append(result.detach().cpu())

    def save_quality(_model, _inputs, result):
        image_scores.append(result.item() / 100.0)

    clip = [str(filename)]
    with dino.register_forward_hook(save_features):
        _, subject_rows = subject_consistency(dino, clip, device, read_frame=False)
    _, motion_rows = dynamic_degree(motion, clip)
    with musiq.register_forward_hook(save_quality):
        image_score, _ = imaging.technical_quality(musiq, clip, device, imaging_quality_preprocessing_mode="longer")
    if len(features) != N_FRAMES or len(image_scores) != N_FRAMES:
        raise ValueError(f"Unexpected VBench frame count for {filename}: {len(features)}, {len(image_scores)}")
    features = F.normalize(torch.cat(features), dim=-1)
    previous = (features[1:] * features[:-1]).sum(-1).clamp_min(0)
    first = (features[1:] * features[0]).sum(-1).clamp_min(0)
    # Same first-frame anchor and temporal pairs as the official full-video score.
    subject_frames = [None] + ((previous + first) / 2).tolist()
    return dict(subject=subject_rows[0]["video_results"], dynamic=bool(motion_rows[0]["video_results"]),
                imaging=image_score), subject_frames, image_scores


def fidelity(reference, candidate, metric, device):
    if reference == candidate:
        return [0.0] * N_FRAMES, [0.0] * N_FRAMES
    pairs = itertools.zip_longest(frames(reference), frames(candidate))
    mse, perceptual = [], []
    for batch in iter(lambda: list(itertools.islice(pairs, 6)), []):
        if any(a is None or b is None or a.shape != b.shape for a, b in batch):
            raise ValueError(f"Unpaired frames for {candidate}")
        tensors = [torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device).float().div_(255)
                   for images in zip(*batch)]
        mse.extend((tensors[0] - tensors[1]).square().mean(dim=(1, 2, 3)).tolist())
        perceptual.extend(metric(*tensors, normalize=True).flatten().tolist())
    if len(mse) != N_FRAMES:
        raise ValueError(f"{candidate}: decoded {len(mse)} frames")
    return mse, perceptual


def latent_frames(filename):
    z = torch.load(filename.with_suffix(".pt"), weights_only=True, map_location="cpu")
    if tuple(z.shape) != (51, 16, 60, 104):
        raise ValueError(f"{filename}: {z.shape}")
    return z.float()


def measure(base, role, idx, scene, seed, device, nets):
    filename = video_path(base, role, idx)
    reference = video_path(base, "native", idx)
    with av.open(str(filename)) as handle:
        stream = handle.streams.video[0]
        if (stream.frames, stream.width, stream.height, stream.average_rate) != (201, 832, 480, 16):
            raise ValueError(f"Unexpected video stream: {filename}")
    timing = {int(i): (float(t), float(peak)) for i, t, peak, _ in
              TIMING.findall((base / role / "inference.log").read_text(errors="replace"))}
    seconds, peak = timing[idx]
    common = dict(seed=seed, scene=scene, policy=role, prompt=idx)
    ref = latent_frames(reference)
    chunks = [dict(common, **{k: v for k, v in row.items() if k not in common})
              for row in chunk_metrics(ref, latent_frames(filename), idx, role)]
    mse, perceptual = fidelity(reference, filename, nets[-1], device)
    scores, subjects, image_scores = {}, [None] * N_FRAMES, [None] * N_FRAMES
    if role != "probe":
        scores, subjects, image_scores = vbench_scores(filename, device, nets)
    video = dict(common, seconds=seconds, peak_mib=peak, subject=scores.get("subject"),
                 dynamic=scores.get("dynamic"), imaging=scores.get("imaging"))
    per_frame = [dict(common, frame=i + 1, mse=mse[i], lpips=perceptual[i],
                      subject=subjects[i], imaging=image_scores[i]) for i in range(N_FRAMES)]
    windows = []
    for end in ENDPOINTS:
        group = per_frame[end - 24:end]
        mean_mse = np.mean([r["mse"] for r in group])
        row = dict(common, end=end, psnr_db=(-10 * math.log10(mean_mse) if mean_mse else None),
                   lpips=np.mean([r["lpips"] for r in group]),
                   latent_l2=chunks[(end + 3) // 12 - 1]["relative_l2"], subject=None, imaging=None)
        if scores:
            row.update(subject=np.mean([r["subject"] for r in group]), imaging=np.mean([r["imaging"] for r in group]))
        windows.append(row)
    return dict(video=video, chunks=chunks, frames=per_frame, windows=windows)


def filmstrip(base, idx, dest):
    samples = (1, 41, 81, 117, 153, 177, 201)
    w, h, margin = 240, 138, 125
    sheet = Image.new("RGB", (margin + w * len(samples), 26 + h * len(ROLES)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, number in enumerate(samples):
        draw.text((margin + col * w + 4, 6), f"Frame {number}", fill="black")
    for row, (role, label) in enumerate(zip(ROLES, LABELS)):
        top = 26 + row * h
        draw.text((3, top + h // 2), label, fill="black")
        for frame_idx, pixels in enumerate(frames(video_path(base, role, idx)), 1):
            if frame_idx in samples:
                thumb = Image.fromarray(pixels).resize((w, h), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (margin + samples.index(frame_idx) * w, top))
    sheet.save(dest, quality=90)


def plot_curves(chunks, windows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for role, name, style in zip((*ROLES[1:], "probe"),
                                  (*LABELS[1:], "One Sage write (n=4)"), ("o-", "s-", "^-", "d--")):
        group = [r for r in chunks if r["policy"] == role]
        values = [np.median([r["relative_l2"] for r in group if r["chunk"] == i]) for i in range(1, 18)]
        axes[0].plot(range(1, 18), 100 * np.asarray(values), style, ms=3, label=name)
        recent = [r for r in windows if r["policy"] == role]
        ys = [np.median([r["lpips"] for r in recent if r["end"] == end]) for end in ENDPOINTS]
        axes[1].plot(ENDPOINTS, ys, style, ms=5, label=name)
    axes[0].axvline(7.5, linestyle=":", linewidth=1, color="gray")
    axes[0].set(xlabel="Generated block", ylabel="Median relative latent L2 (%)", xticks=[1, 4, 7, 10, 13, 17])
    axes[1].set(xlabel="Last frame of 24-frame window", ylabel="Median LPIPS vs. BF16", xticks=ENDPOINTS)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "length-curves.png", dpi=165)
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    study = parser.parse_args().study.resolve()
    out = study / "summary"
    cache = out / "clips"
    cache.mkdir(parents=True, exist_ok=True)
    previews = out / "previews"
    previews.mkdir(exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(3)
    device = torch.device("cuda:0")
    nets = models(Path(__file__).resolve().parents[1] / ".course/eval", device)
    videos, chunks, per_frame, windows = [], [], [], []
    for seed, scenes in ((2026, SCENES), (2027, SCENES[-2:])):
        base = study / f"seed-{seed}"
        for idx, scene in enumerate(scenes):
            roles = (*ROLES, "probe") if seed == 2026 and idx < 4 else ROLES
            for role in roles:
                dest = cache / f"{seed}-{scene}-{role}.json"
                if not dest.exists():
                    print(f"MEASURE {seed} {scene} {role}", flush=True)
                    case = measure(base, role, idx, scene, seed, device, nets)
                    tmp = dest.with_suffix(".tmp")
                    tmp.write_text(json.dumps(case, allow_nan=False), encoding="utf-8")
                    tmp.replace(dest)
                else:
                    case = json.loads(dest.read_text())
                videos.append(case["video"])
                chunks.extend(case["chunks"])
                per_frame.extend(case["frames"])
                windows.extend(case["windows"])
            preview = previews / f"{seed}-{scene}.jpg"
            if not preview.exists():
                filmstrip(base, idx, preview)
    write_rows(out / "videos.csv", videos)
    write_rows(out / "chunks.csv", chunks)
    write_rows(out / "frames.csv", per_frame)
    write_rows(out / "windows.csv", windows)
    plot_curves(chunks, windows, out)
    print(f"DONE {len(videos)} clips, {len(chunks)} blocks, {len(per_frame)} frames, {len(windows)} windows", flush=True)


if __name__ == "__main__":
    main()
