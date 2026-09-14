# data/ — inputs and run archives

Only this file and `input/` are committed; the root `.gitignore` excludes `data/runs/`. As of 2026-09-10 the directory holds **126 GB across five runs**,
down from 352 GB across sixteen.

## What belongs here

```
data/
├── input/
│   ├── input_config/          # sweep configuration
│   └── input_ifcs/            # Data_base_1.ifc, the single base apartment model
└── runs/
    └── <YYYYMMMDDHHMM>/       # one run, e.g. 2026JUL122211
        ├── input/             # 1.1 MB, this run's own copy of the config
        ├── intermediate/      # SDF sample arrays, embedded IFCs, per-stage checkpoints
        ├── checkpoints_hash/  # ~140 MB, the SDF auto-decoder
        └── output/
            ├── graph_*.json           # the training graphs, one file per graph
            ├── holdout/               # the holdout graphs, same format
            ├── graphormer_split_metrics.json
            ├── evaluation_results/
            └── experiments/<arm>/     # one directory per trained head
                ├── holdout_evaluation_report.json
                ├── test_evaluation_report.json
                ├── graphormer_split_metrics.json
                ├── ablation_manifest.json   # ablation arms only
                └── checkpoint/              # ~162 MB, ablation arms only
```

`Data_base_1.ifc` is referenced by name in `scripts/generate_graph_dataset.py` and
`src/DataPreprocessor.py`. `data/input` is 544 KB and is the only genuinely *input* part of
this tree; everything under `runs/` is output.

## Which run backs which experiment

Confirmed against each `paper_section/exp*/extract_data.py`.

| Paper experiment | Run and arm |
|---|---|
| 1, prediction-head ablation | `2026JUL100130`, arms `offset` and `scalar_s`; plus `experiments/exp1_absolute` and `exp2_offset` at the repository root |
| 2, shortcut audit | `2026JUL110024` (leaked), arms `scalar_s` and `scalar_mag_side`, against `2026JUL112325_good_1` (de-identified) |
| 3, pose distillation | `2026JUL112325`, arms `scalar_s_rerun2_nopose`, `scalar_s_pose_w0.1`, `scalar_s_pose_w0.3`; `2026JUL112325_good_1` is the first w = 0 run |
| 4, rotation scaling | `2026JUL112325/…/scalar_s_pose_w0.3` (81 copies) against `2026JUL122211` (161 copies, the final model) |
| 5, GINE baseline | `2026JUL122211`, arm `gin_scalar_s_pose_w0.3` against `scalar_s_pose_w0.3` in the same run |
| 6, failure anatomy | `2026JUL122211`, arm `scalar_s_pose_w0.3` |
| 7, single-rule ablation | `2026JUL122211`, arms `ablation_six_window`, `ablation_single_wall`, `ablation_semantic` |

**`2026JUL122211` is the run to keep if only one can be kept.** It carries all five arms that
Experiments 4 to 7 read, and its holdout report is the source of the paper's 92.7% side
accuracy, 6.2 mm median error and n = 9,648.

## Why the data is this large

Three independent causes, measured on 2026-09-10. None of them is the model and none is the
evidence.

### 1. SDF sample point clouds: 80 GB of the 126 GB

Stage 2 draws surface and volume sample points for every component of every transform copy
and writes them as monolithic arrays:

| File | 81-copy run | Layout |
|---|---|---|
| `samples_points.npy` | 17.3 G | N × 3 × float32 |
| `samples_sdf.npy` | 5.8 G | N × float32 |
| `samples_obj_idx.npy` | 5.8 G | N × int32 |

That is 20 bytes per sample point, and 17.3 G / 12 B gives **N ≈ 1.55 × 10⁹ points** for 81
copies. Scaling to 161 copies gives 3.07 × 10⁹, which is exactly the figure Appendix A
reports, and 3.07 × 10⁹ × 20 B is the 58 GB that the deleted 161-copy rerun carried. The
arithmetic closes, so nothing here is anomalous: it is simply what one billion sample points
costs.

`2026JUL100130` predates the three-array split and holds the same thing as a single
`samples_dict.npy`, 7.6 G.

**These arrays are consumed only by stage 2.** Once the auto-decoder has converged, its
latents are written back into `intermediate/embedded_ifcs/` and the point clouds are never
read again. The decisive evidence is `2026JUL122211` itself: all three arrays were already
deleted from it, its `intermediate/` is **585 MB**, and every `extract_data.py` still
reproduces its numbers, including the paper's headline.

Current holdings:

| Run | `intermediate/` | of which sample arrays | remainder |
|---|---|---|---|
| `2026JUL112325` | 29.0 G | 28.8 G | 259 M |
| `2026JUL112325_good_1` | 29.0 G | 28.8 G | 259 M |
| `2026JUL110024` | 14.8 G | 14.6 G | 212 M |
| `2026JUL100130` | 7.8 G | 7.6 G (`samples_dict.npy`) | 212 M |
| `2026JUL122211` | 585 M | none, already pruned | 585 M |

**Deleting the sample arrays from the four unpruned runs recovers 79.8 GB and takes `data/`
from 126 GB to about 46 GB.** What must stay behind in `intermediate/` is the ~250 MB
remainder: `embedded_ifcs/` (the latents, which *are* the product of stage 2),
`expanded_ifcs/`, `graphormer_jax_split_checkpoints/`, `gin_jax_split_checkpoints/`,
`objects/`, the `idx_*` dictionaries, `samples_manifest.json`, `samples_stats.json`,
`hash_grid_bbox.json` and the stage logs.

The cost of dropping them is that re-running stage 2 from scratch means re-sampling, which is
2.8 minutes of sampling plus 4.0 hours of auto-decoder training per Appendix A. The seed is
fixed, so it is reproducible rather than merely repeatable.

### 2. The graph files are 52% whitespace

`output/graph_*.json` is 14.9 G in `2026JUL122211`: 119,796 files at a median of 130.5 KB.
Re-serialised compactly the same content is 59 KB against 123 KB on disk, so **52% of every
graph file is pretty-print indentation**. That is 7.3 GB of whitespace in one run.

### 3. Each node carries its 128-d latent as JSON text

`node["geometry"]` is the 128-float SDF latent. It accounts for **73% of the node block**,
39.5 KB of 54.1 KB across 30 nodes. As text at roughly 18 characters per float that is about
2.3 KB per node; the same numbers as float32 binary would be 512 B, so the encoding inflates
them about 4.5×, before the whitespace above is counted.

The graphs are also nearly identical to one another. Between two neighbouring files, 27 of 30
nodes are byte-identical and the entire `links` block is identical; only `type`,
`ObjectType`, `coordinates`, `pos_index` and `wall_s` differ, in one or two nodes. The
benchmark varies pose, room semantics and component dimensions over a fixed skeleton, so
119,796 files re-serialise the same latents and the same topology over and over. This is
inherent to the design rather than a defect, but it is why 14.9 GB encodes so little
independent information.

Nothing needs doing about causes 2 and 3 unless the graphs are to be deposited. If they are,
a compact re-dump halves them for free, and moving the latents out to a single `.npy` keyed
by node id would take the 14.9 G to roughly 1 GB.

## What to keep for the paper

Measured per run. "Evidence" is the set of files the seven `extract_data.py` scripts open.

| Run | Total | Evidence | Training graphs | Holdout graphs | Checkpoints | `intermediate/` |
|---|---|---|---|---|---|---|
| `2026JUL122211` | 17.4 G | 79 M | 14.9 G (119,796) | 1.2 G (9,648) | 871 M | 585 M |
| `2026JUL112325` | 37.6 G | 49 M | 7.5 G (60,300) | 617 M (4,824) | 622 M | 29.0 G |
| `2026JUL112325_good_1` | 37.3 G | 13 M | 7.5 G (60,300) | 617 M (4,824) | 297 M | 29.0 G |
| `2026JUL110024` | 19.0 G | 19 M | 3.8 G (30,552) | 308 M (2,412) | 296 M | 14.8 G |
| `2026JUL100130` | 13.0 G | 22 M | 4.3 G (34,596) | 496 M (3,844) | 327 M | 7.8 G |

**Tier 1, the paper's evidence. About 180 MB in total.** Keep unconditionally; this is what a
Zenodo deposit needs, and it is small enough to commit to git directly:

- `output/experiments/<arm>/holdout_evaluation_report.json` and `test_evaluation_report.json`
- `output/experiments/<arm>/graphormer_split_metrics.json`, and `ablation_manifest.json` on the ablation arms
- `output/evaluation_results/`
- `output/graphormer_split_metrics.json`
- `input/`, and `intermediate/samples_manifest.json`

**Tier 2, re-inference without regenerating.** Keep to re-evaluate a checkpoint or redraw a
per-door figure without an 11-hour rebuild: `checkpoints_hash/` (~140 MB per run),
`output/experiments/<arm>/checkpoint/` (~162 MB per ablation arm), `output/holdout/` (0.3 to
1.2 GB per run), and `intermediate/embedded_ifcs/` (7 to 98 MB, the latents themselves).

**Tier 3, regenerable bulk.** The sample arrays (79.8 GB, and provably droppable per above)
and `output/graph_*.json` (36 GB across the five runs, regenerable from `Data_base_1.ifc` and
the fixed seed via `scripts/main.py` menu `1.2` then `3.2`).

## History

**2026-09-10.** Eleven run folders were deleted, taking `data/` from 352 GB to 126 GB.

Nine went after a reference audit across every `.py`, `.md`, `.json` and `.txt` outside
`data/` found no mention of any of them: `2026JUL101857` (540 K), `2026JUL102149` (1.1 G),
`2026JUL091107` (4.0 G), `2026JUL100130 copy` (13 G), `2026JUL102258` (15 G),
`2026JUL102308` (15 G), `2026JUL122211GIN` (15 G), `2026JUL122211Gpmr` (18 G) and
`2026JUL101841` (62 G). `GIN` and `Gpmr` were verified as exact duplicates first: all three
`122211*` folders reported an identical holdout median of 6.1639 mm on 9,648 doors, and
`2026JUL122211` already contains both arms.

Two more went on a separate decision. `2026JUL140644` (75 G) was the newest run and referenced
by nothing, and was *not* a duplicate: its holdout median was 11.47 mm against the paper's
6.16 mm, a later rerun that did not reproduce the reported result. `2026JUL091725` (12 G)
appeared only as the example path in two usage docstrings, had no `output/experiments/` arms
and no holdout graphs, and so backed no number in the paper; those two docstrings now point at
`2026JUL122211`.

Two dangling references were repaired at the same time. `scripts/generate_graph_dataset.py`
hardcoded `data/runs/2026JUN301147`, absent for some time; it now takes the run folder as
`sys.argv[1]` and defaults to the most recently modified run. The six scripts in `scratch/`
all pointed at `2026JUL080055`, also absent, and now point at `2026JUL122211` — though they
remain non-runnable for a separate reason recorded in the root `README.md`.
