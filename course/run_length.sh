#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd -P)
study=${1:?supply a result directory under this worktree}
[[ "$study" == "$root"/runs/* && -d "$study" ]]
cd "$root"
exec 9>"$study/driver.lock"
flock -n 9 || { echo 'This study already has a driver'; exit 1; }
trap 'printf "%s\n" "$?" > "$study/driver.exit"' EXIT
export PATH="$root/.venv/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME="$root/.course/cache"
export HF_HOME="$XDG_CACHE_HOME/hf" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TMPDIR="$root/.course/tmp"
unset PYTHONPATH
py="$root/.venv/bin/python"
smi=/usr/lib/wsl/lib/nvidia-smi
mkdir -p "$study"/{seed-2026,seed-2027,attempts,provenance}
cp "$root/course/run_length.sh" "$study/provenance/"
src="$root/prompts/MovieGenVideoBench_extended.txt"
sed -n '1,4p;68p;347p' "$src" > "$study/seed-2026/prompts.txt"
sed -n '68p;347p' "$src" > "$study/seed-2027/prompts.txt"
head -n 4 "$study/seed-2026/prompts.txt" > "$study/probe-prompts.txt"
test "$(wc -l < "$study/seed-2026/prompts.txt")" -eq 6
test "$(wc -l < "$study/seed-2027/prompts.txt")" -eq 2
"$py" - configs/self_forcing_dmd.yaml "$study/long.yaml" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
config.model_kwargs.local_attn_size = 21
config.model_kwargs.sink_size = 0
OmegaConf.save(config, sys.argv[2])
PY
cp configs/default_config.yaml "$study/provenance/"
git diff -- inference.py pipeline/causal_inference.py wan/modules/causal_model.py > "$study/provenance/implementation.patch"
{
    date -Is
    uname -srm
    git rev-parse HEAD
    "$smi" --query-gpu=name,driver_version,memory.total --format=csv
    "$py" -VV
    "$py" -c "import torch, importlib.metadata as m; print(torch.__version__, torch.version.cuda, m.version('sageattention'), m.version('triton'), m.version('flash-attn'))"
} > "$study/environment.txt"

# Serialize the course's own work and respect other users of this shared GPU.
exec 8>"$root/.course/gpu.lock"
flock 8
idle_count=0
while (( idle_count < 2 )); do
    read -r mem util < <("$smi" --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    apps=$("$smi" --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -z "$apps" ]] && (( mem < 2048 && util < 10 )); then
        idle_count=$((idle_count + 1))
    else
        idle_count=0
        printf '%s QUEUED memory=%sMiB utilization=%s%%\n' "$(date -Is)" "$mem" "$util"
    fi
    if (( idle_count < 2 )); then sleep 15; fi
done
printf '%s GPU acquired\n' "$(date -Is)"

run_case() {
    local seed=$1 label=$2 prompts=$3
    shift 3
    local dest="$study/seed-$seed/$label" attempt rc=0
    if [[ -f "$dest/exit.txt" && $(cat "$dest/exit.txt") == 0 ]]; then return; fi
    attempt=$(mktemp -d "$study/attempts/$seed-$label.XXXXXX")
    local -a cmd=("$py" inference.py --config_path "$study/long.yaml"
        --checkpoint_path "$root/checkpoints/self_forcing_dmd.pt" --use_ema
        --data_path "$prompts" --num_output_frames 51 --num_samples 1 --seed "$seed"
        --save_with_index --save_latents --report_latency --output_folder "$attempt/videos" "$@")
    printf '%q ' "${cmd[@]}" > "$attempt/command.txt"
    printf '\n' >> "$attempt/command.txt"
    printf '%s START seed=%s %s %s\n' "$(date -Is)" "$seed" "$label" "$attempt"
    "${cmd[@]}" > "$attempt/inference.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$attempt/exit.txt"
    printf '%s END seed=%s %s rc=%s\n' "$(date -Is)" "$seed" "$label" "$rc"
    if (( rc != 0 )); then return "$rc"; fi
    ln -s "../attempts/$(basename "$attempt")" "$dest"
}

for policy in native sage native_write native_last_denoise; do
    run_case 2026 "$policy" "$study/seed-2026/prompts.txt" --attention_policy "$policy"
done
run_case 2026 probe "$study/probe-prompts.txt" --sage_write_probe_chunk 1
for policy in native_last_denoise native_write sage native; do
    run_case 2027 "$policy" "$study/seed-2027/prompts.txt" --attention_policy "$policy"
done
