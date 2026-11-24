"""Aggregate paired length-study measurements from score_length.py."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read(filename):
    with filename.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(filename, rows):
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def column(rows, key, operation=mean):
    values = [float(row[key]) for row in rows if row[key] != ""]
    return operation(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    out = parser.parse_args().study.resolve() / "summary"
    videos, windows, chunks = [read(out / f"{name}.csv") for name in ("videos", "windows", "chunks")]
    policies, trends, pairs = [], [], []
    for policy in ("native", "sage", "native_write", "native_last_denoise", "probe"):
        group = [r for r in videos if r["policy"] == policy]
        warm = [r for r in group if int(r["prompt"]) > 0]
        motion = [r["dynamic"] == "True" for r in group if r["dynamic"]]
        policies.append(dict(policy=policy, n=len(group), n_warm=len(warm), warm_seconds=column(warm, "seconds", median),
                             peak_mib=column(warm, "peak_mib", max), subject=column(group, "subject"),
                             dynamic=sum(motion) if motion else None, imaging=column(group, "imaging")))
        for end in (81, 153, 201):
            current = [r for r in windows if r["policy"] == policy and int(r["end"]) == end]
            trends.append(dict(policy=policy, end=end, n=len(current),
                               latent_l2=column(current, "latent_l2", median),
                               lpips=column(current, "lpips", median), psnr_db=column(current, "psnr_db", median),
                               subject=column(current, "subject"), imaging=column(current, "imaging")))
    for row in policies:
        row["speedup"] = policies[0]["warm_seconds"] / row["warm_seconds"] if row["policy"] != "probe" else None

    last = [r for r in windows if r["end"] == "201" and r["policy"] != "probe"]
    by_case = {(r["scene"], r["seed"], r["policy"]): r for r in last}
    metrics = ("lpips", "psnr_db", "latent_l2", "subject", "imaging")
    for scene, seed in dict.fromkeys((r["scene"], r["seed"]) for r in last):
        a = by_case[(scene, seed, "native_write")]
        b = by_case[(scene, seed, "native_last_denoise")]
        pairs.append(dict(scene=scene, seed=seed, **{f"write_minus_last_{k}": float(a[k]) - float(b[k])
                                                     for k in metrics}))
    probe = [r for r in chunks if r["policy"] == "probe"]
    trace = [dict(chunk=i, median=column([r for r in probe if int(r["chunk"]) == i], "relative_l2", median))
             for i in range(1, 18)]
    equal_start = all(r["equal"] == "True" for r in probe if int(r["chunk"]) <= 2)
    write(out / "policies.csv", policies)
    write(out / "trends.csv", trends)
    write(out / "paired-tail.csv", pairs)
    summary = dict(policies=policies, trends=trends, paired_tail=pairs, probe_first_two_equal=equal_start, probe=trace)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{len(videos)} videos, {len(windows)} windows, {len(pairs)} paired A/B comparisons")


if __name__ == "__main__":
    main()
