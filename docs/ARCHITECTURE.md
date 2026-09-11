# Gridiron — System Architecture Blueprint

NFL All-22 coaches-film analysis: local computer vision for the bulk of every play, a hosted
vision model for the hard frames, football intelligence on top, and views of the play that the
camera never gave us (2D bird's-eye, 3D virtual camera).

This document is the build contract. It replaces the earlier plan file (lost) and reflects the
code that already exists in this repository as of 2026-09-11.

Decisions changed on 2026-09-11, after the first draft:

- **Stats Perform is out.** `gridiron/data/statsapi.py` and `stats_pbp.py` stay in the tree but
  are parked: nothing new depends on them and the `gridiron-stats` CLI is not part of any
  milestone. The NFL's own API (`api.nfl.com`) is the play-by-play and identity backbone
  instead (section 5). nflverse stays for the free labels the NFL API does not carry.
- **SAM 3 is the teacher, not the runtime detector** (section 6). It does the labelling; a small
  real-time model does the per-frame work.
- **Labelling is semi-automatic by design** (section 7). Human time per game is budgeted in
  minutes, spent approving contact sheets, never drawing boxes.

---

## 0. Where we are today

The repo is not empty. Four commits, 158 passing tests, and these working pieces:

| Layer | Module | Status |
|---|---|---|
| Play-by-play backbone | `gridiron/data/pbp.py`, `teams.py` | Working. nflverse normalised to one schema, per-play `play_uid`. The NFL API adapter (section 5) is the next source; `statsapi.py` / `stats_pbp.py` are parked. |
| Field model | `gridiron/configs/nfl_field.py` | Working. Yard-accurate landmark grid, LOS anchoring from `yardline_100`. |
| Shot segmentation | `gridiron/perception/shots.py` | Working, calibrated on `2025_wk20_LA-CHI`. Continuous film → camera cuts. |
| Play index | `gridiron/perception/play_index.py` | Working. Viterbi view labelling + pairing → 184 plays matched PBP exactly. |
| Intelligence | `gridiron/intelligence/personnel.py`, `formations.py` | Working. Personnel parsing; slot-based formation taxonomy in a snap-relative frame. |
| Store + chat | `gridiron/data/store.py`, `gridiron/chat/*` | Working. DuckDB semantic layer, two constrained tools, hosted Claude agent. |

What does **not** exist yet is everything between "this stretch of film is play 17" and
"here is where all 22 players were on every frame of play 17": decoding, detection, tracking,
field registration, identity, phase detection, the reasoning router, and the visual outputs.
That is what this blueprint designs.

Hardware and runtime facts that shape the design:

- Apple M4, 16 GB unified memory, Metal 3. GPU inference runs through PyTorch **MPS**.
- Only system Python 3.9.6 is installed today. Current torch and ultralytics wheels target
  3.10+, and 3.9 has been end-of-life since October 2025. Section 2.1 fixes this without
  Homebrew.
- One game of NFL+ All-22 is a ~96-minute 1080p screen recording, roughly 5–8 GB, with the
  two angles of each play shown **sequentially** (sideline, then end zone), not side by side.

---

## 1. System architecture

### 1.1 The one-paragraph version

Film enters once, is cut into per-play, per-angle clips using the play index we already have,
and every clip runs through a **local perception pipeline** on the M4 that produces a
frame-by-frame tracking table in real field yards. A **field registration** step, anchored to
the known line of scrimmage from play-by-play, is what turns pixels into yards and is the single
hardest piece. **Trigger rules** watch the local output for the situations it cannot resolve
(piles, contested catches, lost ball, lost identity, failed registration) and hand a handful of
keyframes plus context to a **hosted vision model** through a provider-agnostic router. Everything
lands in **Parquet on disk, queried through DuckDB**, in the same schema as the NFL Big Data
Bowl tracking data so the intelligence and chat layers work on either. The **2D bird's-eye view
is not a feature bolted on at the end: it is literally the tracking table drawn**, and the 3D
virtual camera is that same table with player pose lifted off the ground plane.

### 1.2 Component diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  A. VIDEO INGESTION                                                                      │
│                                                                                          │
│  source film ──► probe ──► proxy (540p, frame-aligned) ──► shots ──► play index          │
│  (immutable)      │                                         (exists)  (exists)           │
│                   └──► per-play clips: sideline/pNNN.mp4 + endzone/pNNN.mp4              │
│                        + snap-aligned frame iterator (PyAV, seek by frame number)        │
└───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        │ play_uid, two clips, PBP row (down/dist/LOS/desc)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  B. LOCAL INFERENCE PIPELINE  (per play, per angle; MPS)                                 │
│                                                                                          │
│   ┌────────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌────────────┐      │
│   │ Detector   │──►│ Tracker  │──►│ Team classify │──►│ Jersey   │──►│ Pose       │      │
│   │ YOLO11 →   │   │ ByteTrack│   │ SigLIP+KMeans │   │ OCR/cls  │   │ (optional) │      │
│   │ RF-DETR    │   └──────────┘   └───────────────┘   └──────────┘   └────────────┘      │
│   └────────────┘                                                                         │
│   ┌──────────────────────────────┐   ┌────────────────┐   ┌─────────────────────────┐    │
│   │ Field registration           │   │ Snap / phase   │   │ Ball                    │    │
│   │ landmark keypoints → H(t)    │   │ pre / live /   │   │ detect + trajectory     │    │
│   │ + LOS anchor + smoothing     │   │ post-whistle   │   │                         │    │
│   └──────────────────────────────┘   └────────────────┘   └─────────────────────────┘    │
│                                        │                                                 │
│                    pixels → yards; two angles fused by snap-aligned time                 │
│                                        ▼                                                 │
│                 tracking/<play_uid>.parquet   (Big Data Bowl schema + confidence)         │
└───────────────┬────────────────────────────────────────────────────┬─────────────────────┘
                │                                                    │ quality signals
                ▼                                                    ▼
┌──────────────────────────────────┐        ┌───────────────────────────────────────────────┐
│  D. INTELLIGENCE (exists, grows) │        │  C. REASONING ROUTER                          │
│  roles · formation · personnel   │        │  triggers → ReasoningRequest(keyframes, ctx)  │
│  technique (3-tech…) · motion    │◄───────│  asyncio queue · budget · content-hash cache   │
│  coverage · routes · events      │ merges │  VisionProvider: Anthropic | OpenAI | Gemini  │
└───────────────┬──────────────────┘        └───────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  E. STORAGE / EXPORT                                                                     │
│  Parquet per artifact per play · DuckDB views over all of it · manifests with hashes     │
│  Exports: overlay MP4 · 2D top-down MP4/GIF · JSON per play · CSV · chat (exists)        │
│  Later: web film room (Next.js + Three.js) reading the same Parquet                      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Component responsibilities

**A. Video ingestion (`gridiron/ingest/`)**

- `probe`: fps, frame count, keyframe positions, codec, resolution. Written once per source file
  to a JSON sidecar so no stage re-probes.
- `proxy`: one 540p H.264 rendition per source, encoded with the same fps and a forced keyframe
  every second. Every cheap pass (shots, snap detection, registration search) runs on the proxy.
  Frame numbers are identical between proxy and source, so an index found cheaply addresses the
  expensive file exactly.
- `clips`: reads `derived/shots/<game>.plays.parquet` and writes `raw/.../sideline/pNNN.mp4`
  and `endzone/pNNN.mp4` with a 1.5 s pad on both sides. Re-encoded (not stream-copied) so each
  clip begins on a keyframe and seeking inside it is exact.
- `frames`: the one frame-iterator abstraction the whole pipeline uses. Takes a clip and a frame
  range, yields `(frame_index, timestamp, ndarray)`. Handles the seek-then-drop pattern that
  `shots.py` already uses.

**B. Local inference pipeline (`gridiron/perception/`)**

Every model sits behind a `Protocol` in `perception/base.py` and is chosen by name from a
YAML config, so swapping YOLO11 for RF-DETR is a config change. The stages, in dependency order:

1. **Detect**: players, referees, ball. Start with a COCO-pretrained YOLO11 (`person` class
   works out of the box on All-22 at 1080p); fine-tune on ~300 labelled frames for ball and
   referee. Runs on the full-resolution source frame, sliced with `InferenceSlicer` for the
   sideline angle where far-side players are small.
2. **Track**: ByteTrack via `supervision`. Tracks carry a per-track confidence and a
   fragmentation count that the router reads.
3. **Team classify**: SigLIP crop embeddings → UMAP → KMeans(3) per play (home, away,
   officials), then reconciled across the whole game so the cluster ids are stable.
4. **Jersey identity**: number read on every crop in a track, voted across the track, then
   matched to the game roster (NFL API `/football/v2/rosters`, jersey + position per
   `gsisId`) and to the play's `stats[]` records, which name every player who touched the
   play with `gsisPlayerJerseyNumber`. That is exact supervision for who was on the field.
   Bootstrapped by a vision-model read of track contact sheets constrained to the roster's
   number set, then replaced by a small 0–99 crop classifier trained from those labels.
5. **Field registration**: a keypoint model finds yard-line × hash and yard-line × sideline
   intersections plus yard numbers in each frame. The play's LOS from PBP pins the repeating
   5-yard grid to an absolute yard line. Per-frame homographies are smoothed over time and
   rejected when reprojection error spikes. A classical line-detection path (Hough over the
   edge signal `shots.py` already computes) is the fallback when the keypoint model is unsure.
6. **Snap and phase**: the offensive line is the most reliable clock in football. Motion
   energy of the OL cluster is near zero pre-snap and jumps at the snap. Phases per frame:
   `pre_snap`, `live`, `post_whistle`. The last stationary pre-snap frame is the formation
   frame the intelligence layer reads.
7. **Ball**: detection is unreliable in isolation, so the ball track is a constrained
   trajectory (parabolic in flight, attached to a carrier otherwise) fitted over the detector
   hits, and the router is invoked when the fit fails.
8. **Pose** (optional, enabled per run): YOLO11-pose keypoints for stance (three-point,
   two-point), for lifting players off the ground plane in the 3D view, and for contact events.
9. **Project and fuse**: every foot point goes through `H(t)` into field yards. The sideline
   and end-zone clips are the same play, so they are aligned at the snap frame and fused:
   the end-zone angle resolves lateral position (y) well and depth (x) poorly, and the sideline
   angle is the opposite. The fused table carries both per-angle estimates and the merged one.

Output per play: `derived/tracking/<play_uid>.parquet` in Big Data Bowl column names
(`frame_id, nfl_id/track_id, team, jersey, x, y, s, a, dis, o, dir, event`) plus
`conf`, `angle_source`, `phase`.

**C. Reasoning router (`gridiron/reasoning/`)**

- `triggers.py` turns the pipeline's quality signals into typed `ReasoningRequest` objects.
  Initial trigger set:

  | Trigger | Condition on local output | What we ask the model |
  |---|---|---|
  | `pile` | ≥ 6 players within a 3-yard radius and track fragmentation in that region | Ball carrier, forward progress spot, whether ball crossed a line |
  | `contested_catch` | Ball trajectory terminates within 1.5 yd of players from both teams | Catch / incomplete / interception, who, and where feet landed |
  | `ball_lost` | No ball detection for > 12 frames during `live` phase | Who has the ball, or where it is |
  | `identity_unresolved` | Jersey vote confidence < threshold when the track ends | Jersey number from the best 6 crops |
  | `registration_failed` | Homography reprojection error > threshold for > 10 frames | Which yard line is at the LOS, which side the numbers face |
  | `phase_ambiguous` | Snap detector finds 0 or > 1 candidate snap frames | The snap frame index from a contact sheet |

- `router.py` is an `asyncio` queue with a per-game token budget, concurrency limit, retry
  with backoff, and a content-hash cache (hash of the keyframes + prompt + schema), so re-running
  a game never re-bills for an unchanged request.
- `providers/` implements one `VisionProvider` protocol per vendor. Each `analyze()` call takes
  keyframes (6–12 frames, both angles, downscaled), a context block (down, distance, PBP
  description, our current tracking hypothesis) and a Pydantic output schema, and returns the
  validated object. Anthropic is the first implementation. OpenAI and Gemini are thin siblings.
- Results are written to `derived/reasoning/<play_uid>.parquet` and merged into the tracking
  and label tables by the intelligence layer, never overwriting the local estimate, always
  recorded alongside it with its source.

**D. Intelligence (`gridiron/intelligence/`)**

Existing modules stay. New modules consume the tracking table:

- `roles.py`: position assignment from formation-frame geometry (OL, QB, RB, TE, WR, DL, LB,
  CB, S), corrected by roster position when identity is known.
- `technique.py`: defensive-line alignment relative to the offensive line (0, 1, 2i, 2, 3, 4i,
  4, 5, 7, 9), plus wide-9 and head-up flags, computed from lateral offsets in the snap frame.
- `motion.py`: pre-snap motion and shifts from tracking between the first stationary frame and
  the snap.
- `coverage.py`: man vs zone heuristics from defensive-back movement in the first second after
  the snap, and shell (single-high, two-high) from safety depth at the snap.
- `routes.py`: route classification from receiver paths relative to the LOS.
- `events.py`: snap, throw, catch, tackle, out-of-bounds from trajectory features, in Big
  Data Bowl `event` vocabulary.

The same modules run unchanged on Big Data Bowl tracking data, which is how they get validated
before our own CV output is trustworthy.

**E. Storage and export (`gridiron/data/`, `gridiron/viz/`)**

- One Parquet file per artifact per play, one DuckDB database file per data root with views
  over all of them. No row-store database; nothing here is transactional.
- A `manifest.json` per game lists every artifact with its input hashes and the config hash
  that produced it. The pipeline runner uses these to skip stages whose inputs are unchanged.
- Exports: overlay MP4 (boxes, tracks, jersey, phase), 2D top-down MP4 and PNG contact sheets,
  per-play JSON, CSV of the tracking table, and the existing chat interface.

### 1.4 Pipeline orchestration (`gridiron/pipeline/`)

A small stage graph, not Airflow or Prefect. Each `Stage` declares its inputs, outputs, and
config keys. The `Runner` walks the graph for a set of `play_uid`s, checks manifests, and runs
what is stale. Decoding and CPU-bound stages fan out with `concurrent.futures`
(the M4 has ten cores); model inference stays in one process because there is one MPS device;
the router is `asyncio`. A `--dry-run` prints what would run and what the router would bill.

### 1.5 The views the camera never gave us

**2D bird's-eye (deliverable in the first CV milestone).** The tracking table in field yards,
rendered on the `NFLFieldConfiguration` grid at the source fps. Because both angles are fused,
this view is better than either camera alone. It is also the input to every intelligence module,
so it gets validated for free.

**3D virtual camera (second milestone).** Each player is placed at their fused `(x, y)` on the
ground plane. Height comes from pose keypoints (head-to-ankle pixel span through the homography's
local scale) or from the roster when pose is off. Players are rendered as posed capsules in
Three.js with a free camera: behind the QB, behind the safeties, 30 yards up, anywhere. This is
the Madden-replay style, not a photoreal re-render, and it is honest about what two sequential
wide-angle camera views can support.

**Photoreal novel views (research, not planned).** Dynamic Gaussian splatting or NeRF needs
multiple simultaneous viewpoints. All-22 gives us two sequential ones. Flagged so nobody spends
a month on it; revisit if multi-camera feeds ever become available.

---

## 2. Tech stack

### 2.1 Runtime

| Concern | Choice | Why |
|---|---|---|
| Python | **3.12 via `uv`** | `uv` installs with one `curl`, needs no Homebrew, manages interpreters and the venv, and resolves the CV stack in seconds. System 3.9 stays untouched. |
| Package/venv | `uv` + `pyproject.toml` (existing) | Keep the existing extras layout: `core`, `cv`, `chat`, `dev`; add `reasoning`, `viz`. |
| GPU | PyTorch with MPS | Only accelerator on the machine. Every model call goes through a `device` setting so CUDA works on a rented box unchanged. |

Install path (no Homebrew required):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
cd ~/projects/gridiron && uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[cv,chat,reasoning,viz,dev]'
```

### 2.2 Video

| Concern | Choice | Why |
|---|---|---|
| Decode / probe / encode | **PyAV** (bundles ffmpeg) | Already in use in `shots.py`; no system ffmpeg needed; frame-accurate seeking. |
| Image ops | **OpenCV** | Homography solve/warp, Hough lines, crop/resize, drawing. |
| Proxy renditions | PyAV encode, H.264, 540p, keyframe every 1 s | Cheap passes run 4–6× faster and stay frame-aligned with the source. |

### 2.3 Perception models

| Task | Start with | Swap to / alternative | License note |
|---|---|---|---|
| Player / ref / ball detection | **Ultralytics YOLO11** (n/s/m) | **RF-DETR** (`rfdetr`, Apache-2.0) before any distribution; D-FINE as another Apache option | Ultralytics is AGPL; fine for a prototype we run ourselves. |
| Tracking | **ByteTrack** via `supervision` | BoT-SORT (in ultralytics) if re-identification after occlusion matters more | MIT |
| Team split | SigLIP embeddings + UMAP + KMeans (roboflow `sports.TeamClassifier`) | Colour histograms as a zero-model fallback | Apache / MIT |
| Field landmarks | **YOLO11-pose** trained on ~200–400 labelled frames of intersections and numbers | Classical Hough lines + PBP anchor as fallback | Label in CVAT or Roboflow |
| Jersey numbers | Vision-model read of per-track contact sheets, constrained to the roster's numbers (weak labels) | Small 0–99 crop classifier trained from those labels + NFL API `stats[]` jersey records; EasyOCR as an offline fallback | PaddleOCR dropped: PaddlePaddle wheels on Apple Silicon are unreliable. |
| Pose | YOLO11-pose (2D, every frame) | **SAM 3D Body** on keyframes for full 3D mesh + skeleton (feeds the virtual camera); WHAM if temporally-consistent 3D pose across a whole clip is needed | |
| Auto-labelling teacher | **SAM 3** (text-promptable detect + segment + track) | Grounding DINO + SAM 2 if SAM 3 weights are gated for us | Offline only; section 6 explains why it is not the runtime detector. |
| Mask refinement in piles | **SAM 3** on the router's keyframes before they are sent | SAM 2 (lighter) | Not on the per-frame path. |

### 2.4 Reasoning and data

| Concern | Choice | Why |
|---|---|---|
| Play-by-play + identity | **NFL API** (`api.nfl.com`, bearer token) via a Python port of turbo-meme's token client | Per-play `playId`, down/distance/yard line, play type, description, `stats[]` with `gsisPlayerId` + jersey, rosters with jersey + position. Section 5. |
| Personnel / formation labels | **nflverse** (`nfl_data_py`, free) | The NFL API game detail carries no `offense_personnel` / `offense_formation`; nflverse does, for seasons FTN charted. Joined to NFL API plays by GSIS game id + play id. |
| Hosted vision | **Anthropic Claude** first (`anthropic` SDK, `AsyncAnthropic`); OpenAI and Gemini behind the same protocol | Already in the repo for chat; one vendor relationship; structured outputs. |
| Structured I/O | **Pydantic v2** | Validates every provider response and every Parquet row schema. |
| Async | `asyncio` for the router; `concurrent.futures` for decode | Right tool for each: I/O-bound vs CPU-bound. |
| Storage | **Parquet** (pyarrow) per artifact per play + **DuckDB** views | Columnar, zero-copy scans, joins across a season in memory on 16 GB; already chosen and working. |
| Labels / ground truth | JSON per game under `data/groundtruth/`, versioned | Same convention `play_index.py` uses. |
| Config | **pydantic-settings** + YAML files under `gridiron/configs/` | Model choice, provider choice, thresholds, device, budgets all live outside code. |
| Cache | SQLite file under `data/cache/` keyed by content hash | Router responses, SigLIP embeddings. |

### 2.5 Visualisation and tooling

| Concern | Choice |
|---|---|
| Overlays on broadcast frames | `supervision` annotators + OpenCV, encoded with PyAV |
| 2D top-down | OpenCV drawing on the field grid (fast), matplotlib for contact sheets |
| 3D virtual camera | Three.js in the film-room web app, fed by the Parquet through DuckDB-WASM or a small FastAPI endpoint |
| Annotation | CVAT (self-hosted, Docker) or Roboflow Annotate for keypoints and boxes |
| Tests | pytest with synthetic tracking fixtures; one small real clip fixture checked into `tests/fixtures/` |
| Lint / format | ruff (existing) |
| CLI | argparse, one umbrella `gridiron` entrypoint dispatching to the existing stage commands |

---

## 3. Codebase file structure

Additions are marked `+`. Existing files are listed so the whole tree is visible.

```
gridiron/                                   repo root
├── pyproject.toml                          extras: cv, chat, +reasoning, +viz, dev
├── README.md
├── .env.example                            +NFL_API_CLIENT_ID/KEY/SECRET/DEVICE_ID, +ANTHROPIC_API_KEY, +GRIDIRON_DEVICE
├── docs/
│   ├── ARCHITECTURE.md                     this file
│   ├── +SCHEMAS.md                         every Parquet artifact, column by column
│   └── +LABELLING.md                       how to label field keypoints / balls / jerseys
├── gridiron/
│   ├── __init__.py
│   ├── ids.py                              game_key / play_uid (exists)
│   ├── +cli.py                             `gridiron <stage> ...` umbrella entrypoint
│   ├── configs/
│   │   ├── nfl_field.py                    field landmark model (exists)
│   │   ├── +settings.py                    pydantic-settings: paths, device, budgets, model/provider names
│   │   ├── +models/                        YAML per model choice
│   │   │   ├── +detector.yolo11.yaml
│   │   │   ├── +detector.rfdetr.yaml
│   │   │   ├── +tracker.bytetrack.yaml
│   │   │   ├── +field.keypoints.yaml
│   │   │   └── +jersey.easyocr.yaml
│   │   └── +providers/                     YAML per hosted vendor
│   │       ├── +anthropic.yaml
│   │       ├── +openai.yaml
│   │       └── +gemini.yaml
│   ├── data/
│   │   ├── layout.py                       GamePaths (exists; grows new artifact paths)
│   │   ├── pbp.py  teams.py  store.py      (exist)
│   │   ├── stats_pbp.py  statsapi.py       (exist, PARKED — Stats Perform, unused)
│   │   ├── +nflapi/                        api.nfl.com — port of turbo-meme's lib/nfl-api
│   │   │   ├── +token_client.py            identity/v3 exchange, cached token, one 401 re-mint
│   │   │   ├── +game_detail.py             gamedetailsbyslug -> our PBP schema (join plays by playId)
│   │   │   ├── +rosters.py                 /football/v2/rosters -> jersey, position, gsisId
│   │   │   ├── +teams.py                   /experience/v1/teams -> team GUID by abbreviation
│   │   │   ├── +stat_types.py              GSIS statType table (copied from nfl-stat-types.ts)
│   │   │   └── +slugs.py                   game_key <-> NFL slug; season parsed from the slug
│   │   ├── +schemas.py                     Pydantic + pyarrow schemas for every artifact
│   │   ├── +rosters.py                     game roster view: jersey -> player, position (any source)
│   │   ├── +bdb.py                         Big Data Bowl loader -> our tracking schema
│   │   ├── +manifest.py                    per-game artifact manifest with input/config hashes
│   │   └── +cache.py                       SQLite content-hash cache
│   ├── +ingest/
│   │   ├── +probe.py                       fps, frames, keyframes -> sidecar JSON
│   │   ├── +proxy.py                       540p frame-aligned rendition
│   │   ├── +clips.py                       play index -> per-play per-angle clips
│   │   └── +frames.py                      the frame iterator every stage uses
│   ├── perception/
│   │   ├── shots.py  play_index.py         (exist)
│   │   ├── +base.py                        Protocols: Detector, Tracker, TeamClassifier,
│   │   │                                   JerseyReader, FieldRegistrar, PoseEstimator
│   │   ├── +registry.py                    name -> implementation, from YAML
│   │   ├── +detect/
│   │   │   ├── +yolo.py
│   │   │   └── +rfdetr.py
│   │   ├── +track/
│   │   │   └── +bytetrack.py
│   │   ├── +field/
│   │   │   ├── +keypoints.py               landmark detector wrapper
│   │   │   ├── +lines.py                   classical Hough fallback
│   │   │   ├── +homography.py              per-frame H, LOS anchoring, temporal smoothing
│   │   │   └── +fuse.py                    sideline + endzone fusion at snap-aligned time
│   │   ├── +team.py
│   │   ├── +jersey.py
│   │   ├── +ball.py                        detections -> constrained trajectory
│   │   ├── +pose.py
│   │   ├── +snap.py                        snap frame + pre/live/post phases
│   │   └── +pipeline.py                    per-play orchestration -> tracking parquet
│   ├── intelligence/
│   │   ├── personnel.py  formations.py     (exist)
│   │   ├── +roles.py  +technique.py  +motion.py  +coverage.py  +routes.py  +events.py
│   │   └── +merge.py                       fold router results into labels without overwriting
│   ├── +reasoning/
│   │   ├── +base.py                        VisionProvider protocol, ReasoningRequest/Result
│   │   ├── +triggers.py                    quality signals -> requests
│   │   ├── +router.py                      asyncio queue, budget, cache, retries
│   │   ├── +keyframes.py                   pick + downscale + contact-sheet the frames to send
│   │   ├── +prompts/                       one .md per trigger type
│   │   └── +providers/
│   │       ├── +anthropic.py  +openai.py  +gemini.py
│   ├── +pipeline/
│   │   ├── +stages.py                      Stage dataclass: inputs, outputs, config keys
│   │   └── +runner.py                      DAG walk, manifest checks, fan-out, --dry-run
│   ├── +labelling/                        semi-automatic labels (section 7)
│   │   ├── +sample.py                      pick frames across angles, phases, halves, weather
│   │   ├── +teacher_sam3.py                SAM 3 text-prompt boxes/masks -> candidate labels
│   │   ├── +field_selftrain.py             classical H + LOS anchor -> reprojected keypoint labels
│   │   ├── +jersey_weak.py                 vision-model contact-sheet reads -> jersey labels
│   │   ├── +review.py                      builds the approve/reject HTML contact sheets
│   │   ├── +train.py                       fine-tune student models from approved labels
│   │   └── +export.py                      approved labels -> YOLO / COCO dataset folders
│   ├── +viz/
│   │   ├── +overlay.py                     boxes/tracks/jersey/phase on broadcast frames
│   │   ├── +topdown.py                     2D field render, MP4 and contact sheets
│   │   └── +export.py                      JSON / CSV / MP4 writers
│   └── chat/                               (exists)
├── models/                                 weights, gitignored; README with download commands
├── data/                                   gitignored except groundtruth/
│   ├── raw/<season>/wk<NN>/<AWAY>_at_<HOME>/{source,proxy,sideline,endzone}/
│   ├── pbp/
│   ├── groundtruth/                        versioned hand labels
│   ├── derived/{shots,tracking,labels,reasoning,registration}/
│   ├── cache/
│   └── gridiron.duckdb
├── scripts/                                one-off tooling (label export, weight download)
├── notebooks/                              exploration only; nothing imports from here
├── tests/
│   ├── (existing 11 test modules)
│   ├── +fixtures/                          one 3-second real clip, synthetic tracking tables
│   ├── +test_ingest_*.py  +test_field_*.py  +test_snap.py  +test_triggers.py  +test_router.py
└── web/                                    film room, built last (Next.js + Three.js)
```

Rules that keep this modular:

1. Every model and every hosted provider is reached only through a `Protocol` in a `base.py`
   and constructed only by a registry from a YAML name. No stage imports `ultralytics` or
   `anthropic` directly.
2. Every stage reads and writes Parquet whose schema is declared in `data/schemas.py`. Stages
   never pass Python objects to each other across the runner.
3. Heavy imports (`torch`, `ultralytics`, `av`) stay inside functions, as `shots.py` already
   does, so the core package imports without the CV stack installed.
4. `notebooks/` and `scripts/` are leaves. Nothing under `gridiron/` imports from them.

---

## 4. Build order

Each milestone ends with something you can watch or query.

| # | Milestone | Proof it works |
|---|---|---|
| 1 | Runtime upgrade + ingestion: uv/3.12, proxy, clips, frame iterator; NFL API adapter | Every play of the demo game as two clips joined to NFL API `playId`; `gridiron clips` idempotent via manifest |
| 2 | Label bootstrap: SAM 3 teacher on sampled frames, first review sheet, first student detector | Student detector trained from zero hand-drawn boxes; review sheet took under 30 minutes |
| 3 | Detect + track + team split, overlay export | Overlay MP4 of ten plays with stable ids and correct team colours |
| 4 | Field registration + snap/phase + projection | **2D top-down MP4** of ten plays; formation frame matches nflverse `offense_formation` |
| 5 | Intelligence on real tracking: roles, technique, motion | Chat answers "show me every 3-tech alignment" from our own CV output |
| 6 | Reasoning router with `pile` and `contested_catch` triggers | Goal-line plays resolved with source recorded; cache hit on re-run |
| 7 | Jersey identity + roster join | Player names in the top-down view; validated against NFL API `stats[]` jersey records |
| 8 | 3D virtual camera in the web film room, SAM 3D Body on keyframes | Free-camera replay of one drive |

Milestone 4 is the one that decides whether the whole thing works. It is scheduled as early as
its dependencies allow.

---

## 5. The NFL API as the data backbone

Source of truth: turbo-meme's `docs/nfl-api-access.md` and `apps/backend/src/lib/nfl-api/`.
We port the client to Python rather than call the TypeScript app, because the pipeline runs
offline on a laptop and needs the raw payloads on disk.

### 5.1 What we take from each endpoint

| Endpoint | What gridiron uses it for |
|---|---|
| `/experience/v1/gamedetailsbyslug/{slug}?includeReplays=false` | The play list. From `driveChart.plays[]`: `playId`, `quarter`, `clockTime`, `down`, `yardsRemaining`, `yardLine` (team-relative, e.g. `PHI 1`), `playType`, `playDescription`, `playScored`, `playDeleted`, `driveSequence`, and `stats[]`. `homeTeam.id` / `awayTeam.id` are the team GUIDs; `externalIds[source=gsis]` is the GSIS game id used to join nflverse. |
| `stats[]` on each play | Per-player records: `statType`, `teamId`, `yards`, `gsisPlayerId`, `gsisPlayerName`, `gsisPlayerJerseyNumber`. This is the identity supervision: for every play we know which jersey numbers passed, carried, caught, tackled, were targeted, hit the QB, or were flagged. Codes come from the GSIS table copied verbatim from `packages/shared/src/nfl-stat-types.ts`; we never restate a code inline. |
| `/football/v2/rosters?season=&teamId=` | Jersey, position, `gsisId`, legal and common names for every player. Players nest inside `rosters[0]`, not beside it. The set of jersey numbers per team is what constrains the jersey classifier's output space. |
| `/experience/v1/teams?season=` | Team GUID by abbreviation, needed to call rosters. |
| `/football/v2/experience/weekly-game-details?...` | Slate for a week: slug and GSIS id per game, so `game_key` maps to a slug without typing. |

Rules carried over from turbo-meme, unchanged:

- Credentials must be issued to us. Nothing lifted from nfl.com's web bundle. They are read
  from `.env` and `is_configured()` gates every call, so the pipeline runs without them (on
  cached payloads or nflverse alone) and says so.
- Join `driveChart.plays` to anything else by `playId`, never by index. The chart includes
  `GAME_START` and other bookkeeping rows.
- Season comes from the slug (`...-2025-reg-1`), never the kickoff date.
- Raw payloads are cached to `data/pbp/nfl_raw/<game_key>/*.json` before any adaptation, so
  a field the adapter drops is one grep away rather than one round trip away.

### 5.2 How it feeds the pipeline

- **Play index join.** `play_index.py` currently joins film plays to nflverse rows by order.
  With the NFL API, filmable rows are `playDeleted != true` and not an admin entry (timeout,
  two-minute warning, end of period, using the `isNflAdminPlay` test ported from
  `nfl-stat-types.ts`). Same ordered join, same count check, richer rows.
- **Field registration anchor.** `yardLine` like `PHI 1` plus the possessing team (from the
  drive's `teamId`) converts to `yardline_100` and then to absolute field x through
  `NFLFieldConfiguration.line_of_scrimmage_x`. `yardsRemaining` places the first-down line.
- **Identity.** `stats[]` jersey numbers are positive labels for tracks, and the roster's full
  number set is the closed vocabulary. Together they replace everything Stats Perform was for.
- **Personnel and formation.** Not in the NFL API game detail. nflverse still supplies
  `offense_personnel`, `offense_formation`, `defense_personnel` for the seasons it covers, joined
  by GSIS game id and play id. Where nflverse is silent, the CV layer's own formation read is
  the label, and the doc says so.

What the NFL API still does not give us: pre-snap motion, play-action, and per-frame timing.
Those come from the CV layer, as before.

---

## 6. SAM 3 and the current model landscape

Short answer: SAM 3 is the best tool available for *making labels* and for *reasoning-grade
segmentation on a few frames*. It is the wrong tool for the per-frame detector, and the reason
is arithmetic, not quality.

### 6.1 What SAM 3 is

Meta's SAM 3 (November 2025) adds **promptable concept segmentation**: give it a text prompt
("football player", "referee", "football") or an exemplar box and it detects, segments and
tracks every instance of that concept in an image or video. It replaces the two-model
Grounding-DINO-plus-SAM-2 pattern with one model, and it is markedly better at open-vocabulary
detection than either. It shipped alongside **SAM 3D Body**, which recovers a full 3D human mesh
and skeleton from a single image, and **SAM 3D Objects**.

### 6.2 Why it is not the runtime detector

The model is roughly 850M parameters. Meta quotes about 30 ms per frame for 100+ objects on an
H200. On an M4 with 16 GB unified memory through MPS, expect low single-digit frames per second
at full resolution; this has not been measured here and the first task of milestone 2 is to
measure it. The workload per game is large:

| Quantity | Value |
|---|---|
| Plays per game | ~180 |
| Angles per play | 2 |
| Seconds per clip | ~15 |
| Frames at 25 fps | ~135,000 per game |

At 2 fps that is about 19 hours per game for detection alone. A YOLO11-s or RF-DETR-small
student runs at 30–60 fps on the same hardware, so the same game takes under an hour, and the
student is trained on SAM 3's own output, so it inherits most of the accuracy where it matters.

### 6.3 Where SAM 3 is used

1. **Teacher for auto-labelling** (section 7). Text prompts on a few thousand sampled frames
   produce boxes and masks that become the training set for the student detector. This is the
   single biggest reduction in human labelling time in the whole plan.
2. **Router pre-processing.** On the 6–12 keyframes sent to the hosted model for a pile or a
   contested catch, SAM 3 masks the players and the ball first, so the hosted model receives an
   annotated frame rather than a raw one. Prompt-level cost is a few frames per play, which is
   fine.
3. **Ball trajectory recovery.** Where the student loses the ball in flight, SAM 3 with the
   exemplar of the ball from the release frame re-finds it on the sparse frames the trajectory
   fitter asks for.
4. **SAM 3D Body for the virtual camera.** Full 3D pose per player on the snap frame and a
   handful of live frames per play. That is what makes the 3D view a posed human rather than a
   capsule.

### 6.4 The rest of the model landscape, ranked for this project

| Task | Recommendation | Alternatives considered |
|---|---|---|
| Per-frame detection | **RF-DETR** (Apache-2.0, DINOv2 backbone, nano to large; real-time, and there is a segmentation variant) as the primary student. YOLO11 or YOLO26 (Ultralytics, AGPL) as the fastest path to a first result. | D-FINE, RT-DETRv2 (both Apache) are fine substitutes behind the same protocol. |
| Tracking | **ByteTrack** via `supervision`. Cheap, robust, association is on boxes only. | BoT-SORT (adds re-id); SAM 3's built-in video tracking works but is heavy for 22+ objects over long clips. |
| Field registration | Trained keypoint model, self-labelled (section 7.2). | No off-the-shelf NFL field model exists; soccer calibration work (TVCalib, PnLCalib) informs the approach but does not transfer. |
| 2D pose | YOLO11-pose or RTMPose every frame. | ViTPose for accuracy on far players. |
| 3D pose | **SAM 3D Body** on keyframes. | WHAM for video-consistent 3D over a whole clip; 4D-Humans as a lighter option. |
| Jersey read bootstrap | Hosted vision model on contact sheets (weak labels). | EasyOCR offline; PaddleOCR rejected for Apple Silicon reasons. |

Everything in this table sits behind a protocol in `perception/base.py`, so any row can be
swapped by editing a YAML file.

---

## 7. Semi-automatic labelling

Constraint: more compute, less human time. The human's job is to approve or reject contact
sheets with one keystroke per image. Nobody draws boxes or clicks keypoints. Target: under one
hour of review per game for the first game, dropping toward minutes as the students improve.

### 7.1 Players, referees, ball

1. `labelling/sample.py` picks ~2,000 frames from the demo game, stratified across both
   angles, all three phases (pre-snap, live, post-whistle), and both halves of the film. The
   second half matters: snow starts falling mid-game and turns the field white, which is
   exactly the shift that broke the earlier view classifier.
2. `labelling/teacher_sam3.py` runs SAM 3 with three text prompts on every sampled frame and
   writes boxes, masks and scores. Geometry filters remove obvious errors: a "player" whose foot
   point is outside the field mask, a "football" larger than a helmet, duplicate boxes with
   high overlap.
3. `labelling/review.py` renders a contact sheet: each frame with the teacher's boxes drawn,
   sorted by teacher confidence ascending. The reviewer presses one key per frame: keep, drop,
   or flag. Flagged frames go into a short list for a second look with masks shown. Approved
   labels are written to `data/groundtruth/<game_key>.detections.json`, time-keyed like the
   existing shot labels.
4. `labelling/train.py` fine-tunes the student on the approved set, then runs it on a held-out
   set of teacher-labelled frames. Frames where student and teacher disagree above a threshold
   form the next review sheet. Two or three rounds is the expectation.
5. The ball gets one extra filter: the NFL API says whether the play was a pass, so a ball
   track must contain a parabolic flight segment on pass plays and must not on runs. Detections
   that violate that are dropped before review.

### 7.2 Field landmarks, with zero clicks

The field is a known drawing, so the labels can be generated rather than annotated.

1. On each sampled frame, classical line detection (Hough over the edge map `shots.py` already
   computes) finds the family of parallel yard lines and the hash-mark rows. The NFL API's
   `yardLine` for that play says which absolute yard line is at the line of scrimmage, and the
   first-down line follows from `yardsRemaining`. If the film carries painted LOS and
   first-down graphics, as the `shots.py` notes suggest, those two lines are the strongest
   anchors on the screen and are found by colour.
2. From those lines plus the sidelines, an initial homography is solved with RANSAC on the
   easy frames (wide, stable, well-lit). Reprojection error decides "easy".
3. `labelling/field_selftrain.py` projects the full landmark grid from
   `NFLFieldConfiguration` back into each easy frame. Those projected points are the keypoint
   labels. No human looked at them.
4. Train the keypoint student on the easy frames, run it on all frames, re-solve the
   homography with RANSAC, keep frames whose reprojection error is low, retrain. Frames that
   never converge go on a contact sheet with the projected grid overlaid, and the reviewer
   keeps or drops them.

### 7.3 Jersey numbers

1. Tracks from milestone 3 give crops per player per frame. `labelling/jersey_weak.py` builds
   one contact sheet per track (the best 8 crops by size and sharpness) and asks the hosted
   vision model for the number, constrained to the roster's number set for that team. This is
   a router call with its own trigger type and budget line.
2. The NFL API `stats[]` records supply hard labels for the tracks that touched the play:
   passer, carrier, receiver, tackler, target, defender. Agreement between the weak read and a
   hard label promotes the track to the training set without review.
3. Train the 0–99 crop classifier on the promoted tracks. Disagreements between the classifier
   and the weak read go on the review sheet.

### 7.4 Teams, snap frame, phases

Team assignment is fully automatic (SigLIP clustering per play, reconciled across the game).
The snap frame comes from offensive-line motion energy; a contact sheet of the chosen frame per
play, 184 thumbnails on a few pages, lets the reviewer catch misses in a couple of minutes.

### 7.5 Tooling

No CVAT, no Roboflow account. `labelling/review.py` writes a self-contained HTML page with
keyboard shortcuts that saves decisions to a JSON file next to it. That JSON is versioned under
`data/groundtruth/` with the existing shot-view labels. CVAT stays an option for the rare case
where a box genuinely has to be drawn by hand.

---

## 8. What "hosted vision budget" means

Every time the router sends frames to a hosted model, that call costs money, priced per token
of input and output. Images are billed as tokens by their pixel size; a frame downscaled to
about 1000 by 1000 pixels is roughly 1,300 input tokens. The budget is the number the router
refuses to exceed per game, and it decides how eager the triggers can be.

Estimates at Anthropic list prices, using 8 frames and a short structured answer per request:

| Model | Input $/M tokens | Output $/M tokens | Cost per request (est.) |
|---|---|---|---|
| Claude Opus 5 | 5.00 | 25.00 | about $0.07 |
| Claude Sonnet 5 | 2.00 | 10.00 | about $0.03 |

What that means per game of ~180 plays:

| Routing policy | Requests per game | Opus 5 (est.) | Sonnet 5 (est.) |
|---|---|---|---|
| Triggers only, ~15 % of plays fire | ~30 | about $2 | about $1 |
| Every play, one angle | ~180 | about $13 | about $5 |
| Every play, both angles, plus jersey contact sheets | ~500 | about $35 | about $15 |

These are estimates, not quotes. The router logs actual `usage` from every response, and the
content-hash cache means re-running a game never re-bills for unchanged requests. A sensible
starting budget is $5 per game with triggers only; the first real number replaces the estimate.
