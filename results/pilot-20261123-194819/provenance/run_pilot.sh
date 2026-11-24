#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd -P)
study=${1:?supply the absolute result directory in this worktree}
[[ "$study" == "$root"/runs/* ]]
mkdir -p "$study/attempts"
trap 'printf "%s\n" "$?" > "$study/driver.exit"' EXIT

export PATH="$root/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
export XDG_CACHE_HOME="$root/.course/cache" HF_HOME="$root/.course/cache/hf"
export TRITON_CACHE_DIR="$root/.course/cache/triton" TORCHINDUCTOR_CACHE_DIR="$root/.course/cache/inductor"
export TORCH_HOME="$root/.course/cache/torch" CUDA_CACHE_PATH="$root/.course/cache/cuda"
export MPLCONFIGDIR="$root/.course/cache/matplotlib" TMPDIR="$root/.course/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

if [[ ! -f "$study/prompts.txt" ]]; then
    head -n 4 "$root/prompts/MovieGenVideoBench_extended.txt" > "$study/prompts.txt"
    head -n 1 "$study/prompts.txt" > "$study/first.txt"
    head -n 2 "$study/prompts.txt" > "$study/profile.txt"
fi
if [[ ! -f "$study/environment.txt" ]]; then
    {
        git -C "$root" rev-parse HEAD
        uname -srm
        /usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
        "$root/.venv/bin/python" -VV
        "$root/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
        "$HOME/.local/bin/uv" pip freeze --python "$root/.venv/bin/python"
    } > "$study/environment.txt"
    cp "$root/configs/self_forcing_dmd.yaml" "$root/configs/default_config.yaml" "$study/"
    git -C "$root" diff -- inference.py pipeline/causal_inference.py wan/modules/causal_model.py > "$study/implementation.patch"
fi

run_case() {
    local label=$1 src=$2 prompts=$3
    shift 3
    if [[ -f "$study/$label/exit.txt" ]] && [[ $(cat "$study/$label/exit.txt") == 0 ]]; then
        return
    fi
    local attempt rc
    attempt=$(mktemp -d "$study/attempts/$label.XXXXXX")
    local -a command=("$root/.venv/bin/python" inference.py
        --config_path configs/self_forcing_dmd.yaml --checkpoint_path "$root/checkpoints/self_forcing_dmd.pt" --use_ema
        --data_path "$prompts" --num_output_frames 21 --num_samples 1 --seed 2026 --save_with_index
        --output_folder "$attempt/videos" "$@")
    printf '%q ' "${command[@]}" > "$attempt/command.txt"
    printf '\n' >> "$attempt/command.txt"
    printf '%s START %s %s\n' "$(date -Is)" "$label" "$attempt"
    if (cd "$src" && "${command[@]}") > "$attempt/inference.log" 2>&1; then rc=0; else rc=$?; fi
    printf '%s\n' "$rc" > "$attempt/exit.txt"
    printf '%s END %s rc=%s\n' "$(date -Is)" "$label" "$rc"
    if (( rc != 0 )); then return "$rc"; fi
    ln -s "attempts/$(basename "$attempt")" "$study/$label"
}

common=(--save_latents --report_latency)
run_case official "$root/.course/upstream" "$study/first.txt"
run_case native "$root" "$study/prompts.txt" "${common[@]}" --attention_policy native
run_case probe "$root" "$study/prompts.txt" "${common[@]}" --sage_write_probe_chunk 1
run_case sage "$root" "$study/prompts.txt" "${common[@]}" --attention_policy sage
run_case native_write "$root" "$study/prompts.txt" "${common[@]}" --attention_policy native_write
run_case native_last_denoise "$root" "$study/prompts.txt" "${common[@]}" --attention_policy native_last_denoise
run_case native_repeat "$root" "$study/first.txt" "${common[@]}" --attention_policy native
run_case profile_native "$root" "$study/profile.txt" --profile_self_attention --attention_policy native
run_case profile_sage "$root" "$study/profile.txt" --profile_self_attention --attention_policy sage
