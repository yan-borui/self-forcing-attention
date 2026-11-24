#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$root"
assets="$root/.course/eval/length"
mkdir -p "$assets/packages" "$assets/weights" "$root/.course/cache/torch/hub/checkpoints"
export UV_CACHE_DIR="$root/.course/cache/uv"
"$HOME/.local/bin/uv" pip install --python "$root/.venv/bin/python" --target "$assets/packages" \
    --index-url https://pypi.org/simple --no-deps lpips==0.1.4 pyiqa==0.1.13 addict==2.4.0 yapf==0.43.0

fetch() {
    local dest=$1 url=$2
    if [[ -s "$dest" ]]; then return; fi
    curl --fail --location --silent --show-error --retry 2 --connect-timeout 15 --max-time 600 \
        "$url" -o "$dest.part" || return
    mv "$dest.part" "$dest"
}

fetch "$assets/imaging_quality.py" 'https://raw.githubusercontent.com/Vchitect/VBench/1039fe66d7aa2f4b214003ca2d35f5011c48bfee/vbench/imaging_quality.py'
fetch "$root/.course/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth" 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
fetch "$assets/weights/musiq_spaq_ckpt-358bb6af.pth" 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth' || \
    fetch "$assets/weights/musiq_spaq_ckpt-358bb6af.pth" 'https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_spaq_ckpt-358bb6af.pth'
ls -l "$assets/imaging_quality.py" "$assets/weights/musiq_spaq_ckpt-358bb6af.pth" "$root/.course/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
