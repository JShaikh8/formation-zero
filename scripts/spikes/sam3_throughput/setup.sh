#!/usr/bin/env bash
# Environment for spike GRD-32 (SAM 3 image-model throughput on this laptop). Safe to re-run.
#
#   scripts/spikes/sam3_throughput/setup.sh
#
# 1. Upstream code at a pinned commit in .vendor/sam3. Linked into the venv, not pip-installed:
#    upstream's metadata pins numpy<2, which the spike venv (numpy 2) does not need to honour.
# 2. Reuses the GRD-95 spike venv, .venv-sam3d (torch with MPS, av, opencv), and adds the four
#    small pure-Python packages SAM 3's imports need. Creates the venv if it is missing.
# 3. The checkpoint (facebook/sam3, sam3.pt, 3.45 GB) on the film drive (<film root>/models/sam3),
#    with the Hugging Face cache there too. The repo is gated: request access on its page first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SHA=f1647e8d0c1d280991f4d1b7d974c2704537fe66      # facebookresearch/sam3, 2026-10-23
HF_REPO=facebook/sam3
HF_REV=3c879f39826c281e95690f02c7821c4de09afae7   # checkpoint revision measured in results.json
VENDOR="$ROOT/.vendor/sam3"
VENV="${SAM3_VENV:-$ROOT/.venv-sam3d}"

echo "== upstream code @ ${SHA:0:7}"
if [ ! -d "$VENDOR/.git" ]; then
  mkdir -p "$VENDOR"
  git -C "$VENDOR" init -q
  git -C "$VENDOR" remote add origin https://github.com/facebookresearch/sam3.git
fi
if [ "$(git -C "$VENDOR" rev-parse -q --verify HEAD || true)" != "$SHA" ]; then
  git -C "$VENDOR" fetch -q --depth 1 origin "$SHA"
  git -C "$VENDOR" checkout -q FETCH_HEAD
fi

echo "== film drive"
FILM_ROOT="${FZ_FILM_ROOT:-}"
if [ -z "$FILM_ROOT" ] && [ -f "$ROOT/.env" ]; then
  FILM_ROOT="$(sed -n 's/^[[:space:]]*FZ_FILM_ROOT=//p' "$ROOT/.env" | tr -d "\"'" | tail -1)"
fi
FILM_ROOT="${FILM_ROOT:-$ROOT/data/raw}"
case "$FILM_ROOT" in
  /Volumes/*)
    VOL="/Volumes/$(echo "${FILM_ROOT#/Volumes/}" | cut -d/ -f1)"
    if ! mount | grep -q " on $VOL "; then
      echo "   film drive $VOL is not mounted. Plug it in and re-run this script."
      exit 1
    fi ;;
esac
MODELS="$FILM_ROOT/models"
# Package and model caches stay off the boot disk. The login token stays where `hf auth login` put it.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FILM_ROOT/cache/uv}" UV_LINK_MODE=copy
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
export HF_HOME="$MODELS/hf"
echo "   $FILM_ROOT"

echo "== venv $VENV"
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV"
uv pip install -q --python "$VENV/bin/python" \
  torch torchvision timm huggingface_hub av pillow opencv-python numpy psutil \
  "ftfy==6.1.1" regex iopath pycocotools
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ln -sfn "$VENDOR/sam3" "$SITE/sam3"
"$VENV/bin/python" - "$ROOT/scripts/spikes/sam3_throughput" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import torch
from bench import import_sam3
import_sam3()
print(f"   torch {torch.__version__}  mps={torch.backends.mps.is_available()}  sam3 imports")
EOF

echo "== weights"
CKPT="$MODELS/sam3"
if [ -f "$CKPT/sam3.pt" ]; then
  echo "   $CKPT/sam3.pt already present"
else
  if ! "$VENV/bin/hf" auth whoami >/dev/null 2>&1; then
    echo "   Not logged in to Hugging Face. The checkpoint is gated:"
    echo "     1. request access at https://huggingface.co/$HF_REPO"
    echo "     2. $VENV/bin/hf auth login"
    echo "     3. re-run this script"
    exit 1
  fi
  mkdir -p "$CKPT"
  "$VENV/bin/hf" download "$HF_REPO" sam3.pt config.json --revision "$HF_REV" --local-dir "$CKPT"
fi
echo "== ready"
