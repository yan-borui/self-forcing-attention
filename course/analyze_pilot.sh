#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd -P)
study=${1:?supply the absolute pilot result directory}
[[ "$study" == "$root"/runs/* ]]
cd "$root"
py="$root/.venv/bin/python"
export PATH="$root/.venv/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
export XDG_CACHE_HOME="$root/.course/cache" HF_HOME="$root/.course/cache/hf" TORCH_HOME="$root/.course/cache/torch"
export TRITON_CACHE_DIR="$root/.course/cache/triton" TORCHINDUCTOR_CACHE_DIR="$root/.course/cache/inductor"
export CUDA_CACHE_PATH="$root/.course/cache/cuda" MPLCONFIGDIR="$root/.course/cache/matplotlib" TMPDIR="$root/.course/tmp"
mkdir -p "$study/diagnostics" "$study/summary"
while [[ ! -f "$study/driver.exit" ]]; do sleep 5; done
[[ $(cat "$study/driver.exit") == 0 ]]
failed=0

measure() {
    local label=$1
    shift
    local destination="$study/diagnostics/$label"
    mkdir -p "$destination"
    if [[ -f "$destination/exit.txt" && $(cat "$destination/exit.txt") == 0 ]]; then return; fi
    local log rc=0
    log=$(mktemp "$destination/run-XXXXXX.log")
    printf '%s START %s\n' "$(date -Is)" "$label"
    "$@" > "$log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$destination/exit.txt"
    printf '%s END %s rc=%s\n' "$(date -Is)" "$label" "$rc"
    if (( rc != 0 )); then failed=1; fi
}

cli=(--config_path configs/self_forcing_dmd.yaml --checkpoint_path "$root/checkpoints/self_forcing_dmd.pt"
     --data_path "$study/first.txt" --use_ema --seed 2026 --num_output_frames 21 --num_samples 1 --save_with_index)

reference_latents() (
    cd "$root/.course/upstream" || return 1
    COURSE_REFERENCE="$study/diagnostics/reference/latents.pt" "$py" -c \
        'import os, runpy, torch; state = runpy.run_path("inference.py", run_name="__main__"); torch.save(state["latents"][0].detach().cpu(), os.environ["COURSE_REFERENCE"])' \
        "${cli[@]}" --output_folder "$study/diagnostics/reference/videos"
)

measure summary "$py" course/summarize_pilot.py "$study"
measure quality env PYTHONPATH="$root/.course/eval/packages:$root/.course/eval/VBench" \
    "$py" course/score_videos.py "$study"
measure reference reference_latents
measure kv "$py" course/inspect_write.py "$study/summary/kv.csv" "${cli[@]}" \
    --attention_policy native --output_folder "$study/diagnostics/kv/videos"
printf '%s\n' "$failed" > "$study/analysis.exit"
exit "$failed"
