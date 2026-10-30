# Spike GRD-32: SAM 3 throughput on this laptop

**Question.** How fast does Meta's SAM 3 image model run here, with the three text prompts the
labelling plan uses ("player", "referee", "football")? `docs/ARCHITECTURE.md` section 6.2
guessed "low single-digit frames per second" at full resolution on the M4 through PyTorch MPS,
about 19 hours per game at 2 fps, and said nobody had measured it. This measures it.

**Answer: about 0.16 frames per second sustained, not 2.** SAM 3 runs entirely on MPS once
four small shims are in place (below). One 1080p frame with all three prompts takes 3.5 s on a
rested machine. The fanless MacBook Air throttles within about 90 seconds, and after that a frame
takes 6.3 s. 576p film is no faster, because the model squashes every frame to 1008x1008. A whole
game (135,000 frames) would take about 236 hours (10 days) at the settled rate, 12 times the
plan's estimate. The 2,000-frame labelling set takes about 3.5 hours, which is fine. This backs
section 6.2's conclusion (SAM 3 is the teacher, not the runtime detector) by a wider margin than
the section claimed.

Machine: MacBook Air, Apple M4 with an 8-core GPU, 16 GB, fanless, macOS 15.7.4, on AC power.
torch 2.14.1, sam3 at `f1647e8`, checkpoint `facebook/sam3@3c879f3`.

## Results

### Rested, then throttled (1080p, batched prompts, bf16)

`sustained`: warm up, rest 8 minutes, then 1080p frames back to back for 6 minutes.

| Stretch | fps | s per frame | of which image encoder |
|---|---|---|---|
| First 20 frames (the burst number) | 0.287 | 3.48 | |
| First minute | 0.290 | 3.45 | 2.63 s |
| Whole 6 minutes | 0.186 | 5.38 | |
| Last minute (settled) | **0.159** | **6.29** | 4.91 s |

fps per 30 s window: 0.30, 0.29, 0.25, 0.18, 0.15, 0.14, 0.15, 0.15, 0.16, 0.16, 0.16, 0.16.
The work per frame does not change; the GPU slows down as it heats up and settles at a little
over half its rested speed.

### Configs compared (interleaved, warm machine, 15 frames each)

`burst` runs every config on the same frames, round-robin, so heat slows them all alike. The
machine was already warm, so compare these with each other and use the table above for
absolute speed.

| Config | fps | s per frame | image encoder | 3 prompts | players / referees / footballs per frame |
|---|---|---|---|---|---|
| 1080p, upstream API, fp32 | 0.122 | 8.22 | 6.38 | 1.84 | 22.2 / 1.3 / 0.1 |
| 1080p, batched, fp32 | 0.124 | 8.06 | 6.38 | 1.68 | 22.2 / 1.3 / 0.1 |
| 1080p, batched, bf16 | 0.148 | 6.75 | 5.26 | 1.49 | 22.0 / 1.3 / 0.1 |
| 1080p, batched, fp16 | 0.149 | 6.73 | 5.22 | 1.51 | 22.1 / 1.3 / 0.1 |
| 576p, upstream API, fp32 | 0.124 | 8.06 | 6.27 | 1.80 | 21.5 / 2.5 / 0.2 |
| 576p, batched, fp32 | 0.126 | 7.95 | 6.29 | 1.66 | 21.5 / 2.5 / 0.2 |
| 576p, batched, bf16 | 0.150 | 6.65 | 5.21 | 1.44 | 21.2 / 2.5 / 0.2 |

- **Prompts.** Upstream's API makes one call per prompt (`set_text_prompt`), re-encoding the
  text each time. The model also accepts the three prompts as one batch of three queries on the
  same image, with the text encoded once per run. That gives identical detections and saves 2%:
  the text encoder is cheap, and the grounding work still scales with the number of prompts.
- **Precision.** bf16 or fp16 autocast takes 16% less time than fp32 in the same mode (1.19x
  faster), with player counts within 0.3 per frame. Upstream runs its own video predictor under bf16 autocast.
- **Resolution.** 576p and 1080p cost the same to within 2%. The ViT image encoder (about 78% of
  the time) always sees 1008x1008.

### Internal resolution

SAM 3 resizes every frame to **1008x1008** (patch 14, so 72x72 tokens) without keeping the
aspect ratio. A 1920x1080 frame is scaled by 0.525 across and 0.933 down. A 1024x576 frame is
scaled by 0.98 across and 1.75 down. So 1080p film gives the model no more horizontal detail than
576p does. Splitting a 1080p frame into two 960-wide tiles would double the horizontal detail and
double the cost (not tried).

### Memory

840.5M parameters, all float32: 3.4 GB of weights. On MPS the peak current allocation is 3.6 to
3.8 GB. The driver allocation peaks at 5.2 GB for fp32 alone, 6.1 GB for bf16 alone, and 7.2 GB
with all seven configs in one process. Process RSS stays under 1.2 GB because MPS memory is not
counted there. The model fits in 16 GB, but it pushes other apps into swap: swap grew by about
2.6 GB during the runs.

### Cost of a game

| Rate | fps | One game (135,000 frames, section 6.2) | Labelling set (2,000 frames, section 7.1) |
|---|---|---|---|
| Plan's assumption | 2 | 19 h | 0.3 h |
| Rested | 0.287 | 131 h | 1.9 h |
| 6-minute average | 0.186 | 202 h | 3.0 h |
| **Settled** (what a long job sees) | **0.159** | **236 h (10 days)** | **3.5 h** |
| Upstream API in fp32, warm (from the comparison) | 0.122 | 307 h | 4.6 h |

Other costs, none of them the bottleneck: loading the model takes about 100 s per process
(building 840M weights on the CPU, then reading 3.45 GB from the USB drive), so a labelling job
should load it once. Sequential decode runs at 400 fps for 1080p and 1,150 fps for 576p.

### Quality glance

Six frames drawn with masks and boxes (batched, bf16, upstream's 0.5 threshold), in
`<film root>/spikes/sam3/overlays/`. Player masks are yellow, referees magenta, footballs green.

| Frame | Angle, shot | Players | Referees | Footballs | What stands out |
|---|---|---|---|---|---|
| 1080p `f068627`, snow game, before the snow | sideline, tight, live play | 22 | 0 | 0 | Every on-field player boxed; the pile-up is boxed body by body |
| 1080p `f124799`, snow on the field | sideline, wide | 23 | 3 | 0 | All players found, far side included. What looks like one duplicate box. A Bears player in a dark jersey scored 0.54 as **referee** and was not boxed as a player |
| 1080p `f159269`, snow | end zone, pre-snap | 17 | 2 | 0 | Every visible player; both officials at 0.96; the skycam rig ignored |
| 576p `f048644`, clear | sideline, wide kickoff | 29 | 2 | 1 | Every on-field player, the smallest about 40 px tall. 9 of the 29 are uniformed people on the sidelines. The one "football" (0.59) is the ball painted in the NFL shield at midfield; the real ball on the tee was missed |
| 576p `f075502`, clear | end zone, pre-snap | 20 | 2 | 0 | Every visible player; both officials |
| 576p `f004322`, clear | end zone, live play | 22 | 4 | 0 | Every player; all four referee hits are real officials (one cut off at the frame edge, one on the far sideline) |

- **Players:** no on-field player was visibly missed in any of the six. The smallest on-field
  players in them are about 40 px tall at 576p and 65 px at 1080p. The widest shots, where
  far-side players get smaller still, are not in this sample, so recall there is untested. Off-field people in uniform also count as
  players, so the field-mask filter in section 7.1 is needed.
- **Referees:** usable. Real officials score 0.73 to 0.96. There was one confusion, a
  dark-jersey player in the snow.
- **Football:** not usable as is. Over the bursts it averaged 0.1 to 0.2 detections per frame,
  and the only hit in the overlays is a painted logo. The ball needs its own approach: a lower
  threshold, tiles, or tracking plus the pass/run filter in section 7.1.

## What it does

1. `frames` takes 40 frames per film from labelled takes (data/groundtruth, read only): 10
   takes per camera angle, at 30% and 70% through each take. The 1080p frames come from the
   snow game (2025 wk20 LA at CHI) and the 1024x576 frames from the clear game (2026 wk01 SF at
   LA). They are saved as PNG on the film drive, and sequential decode speed is recorded.
2. `sustained` loads SAM 3, warms up, rests, then runs 1080p frames back to back and reports fps
   per 30 s window.
3. `burst` runs the seven configs interleaved: each round runs one frame through each config.
   - `seq` is upstream's documented API: `set_image` once, then `set_text_prompt` for each prompt.
   - `batched` runs one grounding pass with the three prompts as a batch of three queries.
   - `fp32`, or `bf16`/`fp16` autocast on MPS.
4. `overlay` draws masks and boxes on the frames you name and counts detections.
5. `summary` prints the numbers and the cost of a game.

Every timed frame runs all three prompts and copies boxes, scores and full-resolution masks back
to the CPU, so its time is what a labelling pass would actually wait for.

## One-time setup

1. Request access to the gated checkpoint at https://huggingface.co/facebook/sam3, then
   `.venv-sam3d/bin/hf auth login`.
2. Plug in the film drive.
3. `scripts/spikes/sam3_throughput/setup.sh`. It pins the upstream code in `.vendor/sam3`
   (132 MB on the boot disk). It reuses the GRD-95 spike venv `.venv-sam3d`, adding about 7 MB
   of small packages (ftfy, regex, iopath, portalocker, pycocotools), and creates the venv if
   it is missing. It downloads `sam3.pt` (3.45 GB) to `<film root>/models/sam3/`, and keeps the
   package and Hugging Face caches on the drive too. Re-running it is safe.

## Run

```bash
PY=.venv-sam3d/bin/python; B=scripts/spikes/sam3_throughput/bench.py
$PY $B frames
$PY $B sustained --rest 480        # about 16 min: load, warm up, rest 8 min, run 6 min
$PY $B burst                       # about 15 min: 7 configs x (3 warm-up + 15 timed) frames
$PY $B overlay --picks-1080p f068627 f124799 f159269 --picks-576p f048644 f075502 f004322
$PY $B summary
```

The raw numbers, per-frame times, versions and timestamps go to `results.json` next to this
file. Frames and overlays go to `<film root>/spikes/sam3/`.

## Getting it to run on MPS

Upstream targets CUDA. Four things get in the way on Apple silicon, and none of them is real
work on the image path. `bench.py` works around each one in-process and leaves the vendored code
untouched:

| Problem | Fix |
|---|---|
| `import pkg_resources`, used only to find the tokenizer vocabulary, is gone from setuptools 81+ | A two-line shim module |
| `import triton` (CUDA only) in the video tracker's distance transform, imported at module load | Stub that one module |
| Two caches built with `device="cuda"` literals (sine position encodings, decoder box coordinates), and one `pin_memory()` | `redirect_cuda()`: factory calls asking for `cuda` get the run device; pinning is a no-op |
| The ViT MLP calls a fused CUDA kernel, `aten::_addmm_activation`, that also forces bfloat16. MPS lacks the kernel, so it falls back to the CPU and hands bfloat16 to the next float32 layer, and MPS **aborts the process** (`Destination NDArray and Accumulator NDArray cannot have different datatype`) | `unfuse_vit_mlp()`: `gelu(fc1(x))` in the caller's precision |

With those four in place, every op runs on MPS and no CPU fallbacks were reported. The model has
no float64. Upstream's `numpy<2` pin is not needed and is not honoured.

## Caveats

- The machine was shared. Another agent's pipeline jobs ran now and then, one stale process held
  a CPU core at 100% throughout, and the load average was 2 to 4. The work is GPU-bound (98% GPU
  utilisation during the runs), so the effect is small but not zero. Swap was already at 20 GB
  before the runs started.
- On battery it may be slower; these numbers are on AC power.
- macOS reports no thermal levels on Apple silicon without sudo (`pmset -g therm` is empty), so
  throttling is inferred from the timing curve: per-frame time roughly doubles while the work
  stays the same. "Rested" means 8 minutes idle after the model loaded, not a machine that had
  been off.
- The first burst attempt ran configs one after another with 45 s idle between them. Per-frame
  time climbed from 6.9 to 8.9 s within each config, so it was stopped after two configs. Those
  numbers are kept in `results.json` under `burst_sequential_first_try`.
- The sustained run cycles 40 frames. The cost barely depends on the frame, because the image
  encoder dominates.
- Quality is a glance at six images, not a measurement.
- Not tried: Core ML or the Neural Engine, `torch.compile` on MPS, MLX ports, tiling. A rented
  CUDA GPU would label the 2,000 frames in minutes: Meta quotes about 30 ms per frame on an H200.
