#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd -P)
study=${1:?supply the absolute study directory}
[[ "$study" == "$root"/runs/* && -d "$study" ]]
cd "$root"
exec 9>"$study/analysis.lock"
flock -n 9 || { echo 'An evaluator is already active'; exit 1; }
trap 'printf "%s\n" "$?" > "$study/analysis.exit"' EXIT
while [[ ! -f "$study/driver.exit" || ! -f "$study/assets.exit" ]]; do sleep 15; done
[[ $(cat "$study/driver.exit") == 0 && $(cat "$study/assets.exit") == 0 ]]

exec 8>"$root/.course/gpu.lock"
flock 8
smi=/usr/lib/wsl/lib/nvidia-smi
while true; do
    read -r mem util < <("$smi" --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    apps=$("$smi" --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -z "$apps" ]] && (( mem < 2048 && util < 10 )); then break; fi
    printf '%s Queued for quality evaluation\n' "$(date -Is)"
    sleep 20
done
export PATH="$root/.venv/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$root/.course/cache" HF_HOME="$root/.course/cache/hf" TORCH_HOME="$root/.course/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TMPDIR="$root/.course/tmp"
export PYTHONPATH="$root/.course/eval/length/packages:$root/.course/eval/packages:$root/.course/eval/VBench"
cp course/{score_length.py,analyze_length.sh,summarize_pilot.py} "$study/provenance/"
printf '%s Evaluation started\n' "$(date -Is)"
"$root/.venv/bin/python" course/score_length.py "$study"
