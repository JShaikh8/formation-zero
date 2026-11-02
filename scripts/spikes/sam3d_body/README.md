# Spike GRD-95: SAM 3D Body on All-22 players

**Question.** Does SAM 3D Body give a usable 3D body for an All-22 player? All-22 players are
small in frame, padded and helmeted, and the model was trained mostly on people seen up close.
The answer decides what milestone 8 renders (GRD-59, decision D-016): fitted bodies, or capsules
posed from 2D stance.

**Why now.** It costs an afternoon, and it is the one part of the 3D virtual camera that a
basketball demo cannot answer for us. Everything else in that view is tracking (milestones 3,
4 and 7).

## What it does

1. `pick` takes frames at 30% and 70% through 6 sideline and 6 end-zone takes, spread over the
   game, from the 1080p source. That is 24 frames: roughly near the snap, then live play. The
   takes come from the hand-labelled views in `data/groundtruth/<game>.shot_views.json`.
2. `run` finds people with YOLO11 (COCO person class; D-007 allows it for prototypes). It keeps
   those with their feet on green turf, then picks 8 isolated and 8 crowded people per angle,
   spread evenly over box height, at most 3 from any one frame. SAM 3D Body (ViT-H checkpoint,
   body decoder only) fits a body to each person and records seconds per person.
3. `sheet` draws each crop in four panels: the crop (nearest-neighbour, so you see how few
   pixels the model had), the mesh over it, the mesh from the side, and the mesh from above.
4. `grade` records one letter per crop. `summary` breaks the grades down by angle, crowding and
   height, and turns the timing into the cost of a game.

## One-time setup

1. Request access to the gated checkpoint at https://huggingface.co/facebook/sam-3d-body-vith
   (requires a Hugging Face account).
2. Plug in the film drive.
3. `scripts/spikes/sam3d_body/setup.sh`. It builds `.venv-sam3d` (about 1.2 GB on the boot
   disk) and pins the upstream code in `.vendor/`. It downloads the checkpoint to
   `<film root>/models/`, or tells you to run `.venv-sam3d/bin/hf auth login` first.
   Re-running it is safe.

## Run

```bash
PY=.venv-sam3d/bin/python; S=scripts/spikes/sam3d_body/spike.py
$PY $S pick
$PY $S run          # --hfov 30 to try a narrower lens than upstream's default (~47 degrees)
$PY $S sheet        # data/derived/spikes/sam3d_body/2026_wk01_SF-LA/sheets/
$PY $S grade gggrb...   # one letter per crop, in id order (c01, c02, ...); or c07=b to fix one
$PY $S summary
```

## Grading

| Letter | Meaning |
|---|---|
| `g` good | Posture matches the player: stance, lean, which leg is forward, roughly where the arms are. Would look right in the 3D view up close. |
| `r` rough | Overall posture right (standing, running, crouched, facing the right way), limbs off. Fine from a distance. |
| `b` broken | Wrong posture or facing, merged with a neighbour, twisted, or not a person. |
| `x` not a player | Coaches, staff or photographers on the sideline. Left out of the counts. |

## Decision rule

- **Fitted bodies for everyone** if at least 4 in 5 isolated crops on both angles grade `g` or
  `r`, and fewer than 1 in 5 of the smallest sideline crops grade `b`.
- **Fitted bodies for skill players and officials, stance templates for linemen** if isolated
  crops pass but crowded crops are mostly `b`.
- **Capsules posed from 2D keypoints** if isolated crops in the common size range are mostly `b`.
- Timing decides how many keyframes per play, not whether to do it.

Record the outcome on GRD-95 and, if it changes the plan, as a decision next to D-016.

## Shortcuts taken, on purpose

- No detectron2, pyrender or MoGe: boxes come from YOLO11 and meshes are drawn by a small
  numpy point renderer. Upstream's inference path hard-codes CUDA in two places. `spike.py`
  redirects both to MPS and converts float64 to float32 (MPS has no float64). The MHR body
  model is TorchScript that casts to float64 inside, so it runs on CPU while the backbone and
  decoder run on MPS: 1.06 s per person against 24 s all on CPU, with the same bodies to within
  2 mm (median). If MPS still fails, the run reloads on CPU and says so.
- The camera is upstream's default (focal length = image diagonal). All-22 cameras are long
  lenses, so `--hfov` is there to check whether the lens assumption changes the bodies.
- The turf filter drops players standing on painted end zones.
- The labelled takes cover the first 32 minutes of the film only.

## One whole play

`play.py` runs the same model down a tracked take on the 1080p film, with the real lens read from the
painted lines. Play 40 of Chargers-Chiefs, 2026-11-01, the game's best-tracked take, so a best case:
4,924 bodies on 226 frames in 1 h 58 min, 1.45 s a body.

| | Before the snap | In play | After |
|---|---|---|---|
| Joints about the pelvis, median cm a frame | 1.6 | 7.0 | 2.9 |
| Hip turn past 60 degrees in one frame | 0 of 1,298 | 79 of 2,368 | 2 of 1,232 |
| Giants (under 0.8 of the track's running distance) | 1 of 1,320 | 65 of 2,372 | 3 of 1,232 |

Flips follow size: 0.2% of steps on boxes over 110 px, 10.7% under 55 px. Bones hold within 4% in a
track, but thigh 0.38 m and shin 0.35 m are short, and bodies sit at 0.81 of the true distance from the
camera. Stand them on the field by their feet in yards, not by the model's distance. Fits hold vertices
and joints only; pose-space filtering needs the pose parameters saved too.
