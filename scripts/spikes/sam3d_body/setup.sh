#!/usr/bin/env bash
# Environment for spike GRD-95 (SAM 3D Body on All-22 player crops). Safe to re-run.
#
#   scripts/spikes/sam3d_body/setup.sh
#
# 1. Upstream code at a pinned commit in .vendor/ (it ships no package metadata).
# 2. Its own venv, .venv-sam3d, with only what inference needs: no detectron2, pyrender or
#    MoGe, which upstream lists for its demo and training. Kept apart from .venv so torch and
#    ultralytics never touch the project's dependencies.
# 3. Weights on the film drive (<film root>/models), not the boot disk. This step needs the
#    drive mounted and Hugging Face access to the gated checkpoint; without either it says
#    what is missing and stops, and the first two steps still count.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SHA=b5c765a0d89d789985e186d396315e7590887b94   # facebookresearch/sam-3d-body, 2026-02-19
VENDOR="$ROOT/.vendor/sam-3d-body"
VENV="$ROOT/.venv-sam3d"
HF_REPO="${SAM3D_REPO:-facebook/sam-3d-body-vith}"

echo "== upstream code @ ${SHA:0:7}"
if [ ! -d "$VENDOR/.git" ]; then
  mkdir -p "$VENDOR"
  git -C "$VENDOR" init -q
  git -C "$VENDOR" remote add origin https://github.com/facebookresearch/sam-3d-body.git
fi
if [ "$(git -C "$VENDOR" rev-parse -q --verify HEAD || true)" != "$SHA" ]; then
  git -C "$VENDOR" fetch -q --depth 1 origin "$SHA"
  git -C "$VENDOR" checkout -q FETCH_HEAD
fi

echo "== venv $VENV"
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV"
uv pip install -q --python "$VENV/bin/python" \
  torch torchvision \
  roma pytorch-lightning yacs omegaconf einops braceexpand timm huggingface_hub \
  ultralytics av pillow
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ln -sfn "$VENDOR/sam_3d_body" "$SITE/sam_3d_body"
"$VENV/bin/python" - <<'EOF'
import torch, ultralytics, sam_3d_body.sam_3d_body_estimator  # noqa: F401
print(f"   torch {torch.__version__}  mps={torch.backends.mps.is_available()}  ultralytics {ultralytics.__version__}")
EOF

echo "== weights"
FILM_ROOT="${FZ_FILM_ROOT:-}"
if [ -z "$FILM_ROOT" ] && [ -f "$ROOT/.env" ]; then
  FILM_ROOT="$(sed -n 's/^[[:space:]]*FZ_FILM_ROOT=//p' "$ROOT/.env" | tr -d "\"'" | tail -1)"
fi
FILM_ROOT="${FILM_ROOT:-$ROOT/data/raw}"
case "$FILM_ROOT" in
  /Volumes/*)
    VOL="/Volumes/$(echo "${FILM_ROOT#/Volumes/}" | cut -d/ -f1)"
    if ! mount | grep -q " on $VOL "; then
      echo "   film drive $VOL is not mounted: weights pending. Plug it in and re-run this script."
      exit 0
    fi ;;
esac
MODELS="$FILM_ROOT/models"
CKPT="$MODELS/$(basename "$HF_REPO")"
if [ -f "$CKPT/model.ckpt" ] && [ -f "$CKPT/assets/mhr_model.pt" ]; then
  echo "   $CKPT already present"
else
  if ! "$VENV/bin/hf" auth whoami >/dev/null 2>&1; then
    echo "   Not logged in to Hugging Face. The checkpoint is gated:"
    echo "     1. request access at https://huggingface.co/$HF_REPO"
    echo "     2. $VENV/bin/hf auth login"
    echo "     3. re-run this script"
    exit 1
  fi
  mkdir -p "$CKPT"
  "$VENV/bin/hf" download "$HF_REPO" --local-dir "$CKPT"
fi
echo "   YOLO11 weights download to $MODELS on the first run."
echo "== ready"
