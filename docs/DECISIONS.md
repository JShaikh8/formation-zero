# Decision log

One entry per decision that shaped the project. Newest at the bottom. Each entry says what was
decided, why, what else was considered, and what it cost us. A superseded entry stays in the log
with a pointer to the one that replaced it. The tracker site renders this file.

Format for a new entry: a `## D-NNN — Title` heading, then `- key: value` lines for date,
status (accepted, superseded, revisit), tags, and optionally supersedes, then the prose.

## D-001 — Build football logic ourselves on top of roboflow/sports scaffolding
- date: 2026-06-24
- status: accepted
- tags: cv, architecture

**Decision.** Import roboflow/sports for the generic pieces (view transformer, team classifier,
tracker wiring, slicing) and write every football-specific piece in our own package.

**Why.** The scaffolding is soccer-only. Its geometry, pitch model and calibration keypoints do
not transfer. Its infrastructure does.

**Alternatives.** Fork the repo and edit it into a football version. Rejected: we would carry
their soccer assumptions and their release cadence.

**Gained / lost.** Gained a working tracker pipeline on day one. Lost nothing we needed.

## D-002 — ByteTrack for tracking, not a segmentation model
- date: 2026-06-24
- status: accepted
- tags: cv, tracking

**Decision.** Track players with ByteTrack on detector boxes.

**Why.** Segmentation-based video trackers were too slow for 22+ objects over thousands of clips
and, at the time, gated. Box association is cheap and good enough when the detector is good.

**Revisit when.** A tracker with re-identification is needed to survive long occlusions in piles.

## D-003 — DuckDB and Parquet as the store
- date: 2026-06-24
- status: accepted
- tags: data, storage

**Decision.** One Parquet file per artifact per play; DuckDB views over all of them; no server.

**Why.** Columnar files scan fast, a whole season fits in memory on the laptop, and there is no
database to run or back up. The Big Data Bowl data is already in this shape.

**Alternatives.** Postgres (a server for no transactional need), SQLite (row store, slow scans).

## D-004 — Hosted Claude for the chat brain; everything else self-hosted
- date: 2026-06-24
- status: accepted
- tags: llm, hosting, cost

**Decision.** The chat interface uses a hosted Claude model through the official SDK, driving two
constrained tools. The model never writes SQL. All vision stays on our hardware.

**Why.** Text-to-tool over a whitelist is safe and cheap; running a large language model locally
is not worth the effort for a chat box. Keeping vision local keeps per-game cost near zero.

**Gained / lost.** Gained a working chat over real data in one session. Lost independence from
one vendor for the chat piece; the provider is swappable behind a protocol.

## D-005 — Play-by-play is the alignment backbone and anchors field registration
- date: 2026-06-24
- status: accepted
- tags: data, cv, registration

**Decision.** The official play list is the canonical ordering of the game. Film clips are
joined to it in order. The line of scrimmage from that list pins the camera-to-field
transform to an absolute yard line.

**Why.** A football field repeats every five yards. From one frame you cannot tell the 30 from
the 35 without something external. The play-by-play is that something.

**Gained / lost.** Solved the hardest registration ambiguity with data we already had. Cost:
the join is only valid when film play count equals play-by-play play count, and the code
refuses to guess when they disagree.

## D-006 — Validate intelligence on Big Data Bowl data before our own vision output
- date: 2026-06-24
- status: accepted
- tags: strategy, data

**Decision.** Every football-intelligence module is written against the Big Data Bowl tracking
schema and tested there first. Our vision layer produces the same schema.

**Why.** It gives a working demo before the hardest computer vision is done, and it separates
"the football logic is wrong" from "the tracking is wrong."

## D-007 — YOLO11 first, RF-DETR before anything ships
- date: 2026-06-24
- status: accepted
- tags: cv, license

**Decision.** Prototype detection with Ultralytics YOLO11. Swap to RF-DETR (Apache-2.0) behind
the same interface before any distribution.

**Why.** YOLO11 is the fastest path to a first result on a Mac. Its AGPL license is fine for a
tool we run ourselves and a problem for anything we hand to others.

## D-008 — Stay on system Python 3.9 for now
- date: 2026-06-24
- status: superseded
- tags: runtime

**Decision.** Lower the project's Python requirement to 3.9 because it was the only interpreter
on the machine and Homebrew was unavailable.

**Superseded by.** D-012.

## D-009 — A sequence model for camera-angle labelling, not a per-shot classifier
- date: 2026-08-04
- status: accepted
- tags: cv, film

**Decision.** Label each camera take as sideline or end zone with Viterbi decoding over the whole
film, using a strong prior that adjacent takes alternate.

**Why.** Two hand-rolled per-shot classifiers looked convincing on a sample and landed near
chance on the full film (snow changed the field's contrast mid-game). The film's structure is
not weak: hand labels showed adjacent takes alternate 37 out of 37 times. The sequence model
scored 92 to 96 percent on held-out labels; the per-shot threshold scored 84.

**Gained / lost.** Gained a play index whose count matched the play-by-play exactly. Lost the
ability to label a single take in isolation; the model needs its neighbours.

## D-010 — Keep the existing repository; do not restart
- date: 2026-09-10
- status: accepted
- tags: strategy

**Decision.** The blueprint builds on the summer's code rather than starting over.

**Why.** Four commits, 158 passing tests, and every locked decision above already embodied in
working modules. Restarting would rebuild the same foundation with the same decisions.

**Gained / lost.** Gained weeks. Lost nothing except the option to rename things, which is cheap
anyway.

## D-011 — EasyOCR over PaddleOCR for the jersey bootstrap
- date: 2026-09-10
- status: accepted
- tags: cv, identity, apple-silicon

**Decision.** Use EasyOCR (PyTorch-based) as the offline text reader; do not depend on
PaddleOCR.

**Why.** PaddlePaddle wheels are unreliable on Apple Silicon. Either way the OCR is a bootstrap:
the long-term jersey reader is a small classifier trained from labels.

## D-012 — Python 3.12 through uv
- date: 2026-09-10
- status: accepted
- tags: runtime
- supersedes: D-008

**Decision.** Install uv with one curl command, let it install Python 3.12, and move the project
to it.

**Why.** Python 3.9 has been end-of-life since October 2025 and current PyTorch and Ultralytics
wheels target 3.10+. uv needs no Homebrew.

## D-013 — The NFL API is the data backbone; the earlier second source is parked
- date: 2026-09-11
- status: accepted
- tags: data, vendor

**Decision.** Port the NFL API client from the owner's other project and make it the source of
plays, yard lines, and per-play player involvement. The second-source client built in July
stays in the tree, unused.

**Why.** The NFL's own feed carries a stable play id, down, distance, yard line, play type,
description, and a per-play list of players with jersey numbers, which is exactly the
supervision the identity layer needs. One vendor fewer.

**Alternatives.** Keep both and reconcile. Rejected: two sources for one fact is a trap unless
the disagreement is the product, and here it is not.

**Gained / lost.** Gained an official per-play identity record. Lost nothing the NFL API does
not carry, except personnel and formation labels, which nflverse still supplies for past seasons.

**Revisit when.** nflverse stops covering a season we need labels for.

## D-014 — SAM 3 is the teacher, not the runtime detector
- date: 2026-09-11
- status: accepted
- tags: cv, models

**Decision.** Use Meta's SAM 3 to generate training labels from text prompts and to mask the
few frames sent for hosted reasoning. Run a small real-time detector (RF-DETR or YOLO) on every
frame.

**Why.** Quality is not the issue; throughput is. A game is roughly 135,000 frames across both
angles. An 850-million-parameter model on a 16 GB M4 runs at low single-digit frames per second,
which is about 19 hours per game for detection alone. A student trained on SAM 3's own output
runs the same game in under an hour.

**Alternatives.** Grounding DINO plus SAM 2 as the teacher (kept as the fallback if SAM 3 weights
are gated). SAM 3's own video tracker (heavy for 22 objects over long clips).

**Gained / lost.** Gained near-zero hand labelling. Lost some accuracy on the rarest cases,
which the router is designed to catch.

## D-015 — Labelling is semi-automatic: compute over human time
- date: 2026-09-11
- status: accepted
- tags: labelling, process

**Decision.** No one draws boxes. A teacher model proposes labels, a human approves contact
sheets with one keystroke per image, and the student retrains on approved labels until it
agrees with the teacher. Field keypoints are generated by projecting the known field grid
through an initial homography, with no clicks at all.

**Why.** The owner's time is the scarce resource; the laptop's time is not.

**Gained / lost.** Gained a labelling loop that scales to many games. Cost: a first-time build of
the review tooling and a few rounds of retraining per game.

## D-016 — The 3D view is tracking plus pose; photoreal novel views are out of scope
- date: 2026-09-10
- status: accepted
- tags: 3d, scope

**Decision.** Render the virtual camera from fused player positions and 3D pose (SAM 3D Body on
keyframes) in a web viewer. Do not attempt Gaussian splatting or NeRF.

**Why.** Photoreal novel-view synthesis needs multiple simultaneous viewpoints. All-22 gives two
sequential replays of the same play, which supports positions and pose, not a re-render of the
pixels.

**Revisit when.** Multi-camera feeds become available.

## D-017 — Hosted vision budget: $5 per game to start, triggers only
- date: 2026-09-11
- status: accepted
- tags: cost

**Decision.** The router sends frames to the hosted model only when local models signal trouble,
under a $5 per game cap, with a content-hash cache so re-runs are free.

**Why.** Estimated cost with triggers only is about $2 per game on Claude Opus 5 and about $1 on
Sonnet 5; sending every play from both angles would be about $35. The cap protects against
trigger rules that fire too often while we tune them.

**What would change it.** If a measured accuracy gain on piles and contested catches justifies
a higher cap, the log records the new number and what it bought.

## D-018 — Public name: Formation Zero; package name stays gridiron
- date: 2026-09-11
- status: superseded
- tags: naming

**Decision.** The project, the GitHub repository and the tracker site are called Formation
Zero. The Python package keeps its working name, gridiron, so nothing has to be renamed in code.

**Why.** Renaming a package touches every import for no functional gain. A public name and a
code name can differ.

**Superseded by.** D-019, the same day. The reasoning above was weak at this size.

## D-019 — One name everywhere: the package is formation_zero
- date: 2026-09-11
- status: accepted
- tags: naming
- supersedes: D-018

**Decision.** Rename the Python package from `gridiron` to `formation_zero`, the repository
folder to `formation-zero`, and the command-line tools to `fz-pull-pbp`, `fz-chat` and
`fz-shots`. The working name survives only in the June journal entry, as history.

**Why.** The owner asked why two names existed and whether the code was worth keeping. It was,
and the rename was cheap: one package directory, four entry points, thirty files that mention
the name, fifteen minutes. In six months it would be an afternoon. A single name removes a
question every new reader would otherwise ask.

**Gained / lost.** Gained one name. Lost nothing; all 158 tests pass after the rename.

## D-020 — The play record is the product contract; a read API serves it
- date: 2026-09-11
- status: accepted
- tags: product, data, api

**Decision.** Every play produces one versioned JSON record with sections for the situation,
the official result, personnel, pre-snap (initial formation, shifts, motions with type and
direction, formation at the snap, defensive front and shell), at-snap, live (routes with
route-tree labels and break points, run gap and scheme, quarterback drop and time to throw,
blocks, coverage, pass rush, ball flight, an events timeline), post-snap (tackle, spot, yards,
yards after catch), and a per-player block with alignment, role, technique, stance, direction
and body orientation at the snap. Every field carries its source and a confidence. The schema
is written in milestone 1 and served by a read-only JSON API in milestone 9.

**Why.** The owner read the plan and found those items named as modules but never specified as
an output that a client could receive. Without a contract, each stage would invent its own
shape and the export would be an afterthought. Writing the record first means every stage
fills in a field that already exists, and a client can integrate against it before the vision
work is finished, because the official sections are populated from day one.

**Alternatives.** Export the tracking Parquet and let clients derive football from it.
Rejected: the football reading is the product; clients should not have to rebuild it.

**Gained / lost.** Gained a stable deliverable and a place for every future feature to land.
Cost: a schema to maintain and version, and one more milestone.

## D-021 — Corrections are first-class: the human fix is the best label we will ever get
- date: 2026-09-13
- status: accepted
- tags: product, data, labelling, process

**Decision.** Design the correction loop now, before any vision output exists: every field of a
play record can be corrected in the film room, corrections are stored as versioned hand labels
with the machine's value kept beside them, they feed evaluation and retraining, and a re-run
never overwrites a reviewed play. Schema, storage and API endpoints are specified in the
blueprint (section 9.5); the editing UI is built once there is real output to correct.

**Why.** The owner asked whether there would be a way to teach the system where it is wrong,
and asked before it was needed. Bolting corrections on later means every stage would have to be
reopened to respect them; designing them in means every stage writes fields that can be
overridden from day one.

**Alternatives.** Corrections as ad-hoc edits to the output files. Rejected: unattributed, lost
on re-run, invisible to evaluation.

**Gained / lost.** Gained a path from "the model is wrong here" to "the model learned." Cost: two
blocks in the schema and an append-only log, both trivial now.
