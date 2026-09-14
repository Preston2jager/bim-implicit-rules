# SDF_3 — learning implicit design rules from BIM geometry

Code and results for the paper *Learning Implicit Design Rules from BIM Geometry for
Automated Design Review: A Controlled Study of Graph Transformers over Neural Signed
Distance Function Embeddings* (Du, Hou and Zhang; submitted to *Buildings*).

A graph transformer is asked to predict where along its host wall an omitted door belongs,
from pose-conditioned neural SDF embeddings of the surrounding components and IFC topology
alone. No coordinates enter the model, identity-bearing categorical features are stripped,
and evaluation is restricted to rigid transforms of the plan that were withheld from
training.

## Layout

| Path | What is in it |
|---|---|
| `scripts/` | Entry points. `main.py` is the interactive pipeline; the rest are one-shot tools |
| `src/` | Library code: `DataPreprocessor`, `Embedding`, `EmbeddingHash`, `Graphormer`, plus `Tools/` renderers |
| `sdf_sampler/` | Rust SDF point sampler, std-only so it builds offline (`cargo build --release`) |
| `paper_section/` | One self-contained directory per experiment, `exp1` to `exp7`: extraction script, extracted numbers, figure script and figures |
| `experiments/` | Early run outputs kept because `paper_section/exp1_head_ablation` reads from them |
| `data/input/` | The base IFC model (`Data_base_1.ifc`) and sweep configuration, 544 KB. Everything else is generated |
| `data/runs/` | Run archives. **Not in this repository** — 126 GB. See [data/README.md](data/README.md), which also explains what makes it large |

## Environment

Python 3.12, dependencies pinned in `uv.lock`:

```bash
uv sync
```

JAX and Flax carry the Graphormer and the SDF auto-decoder; Torch and
`torch-geometric` carry the GINE baseline; `ifcopenshell` reads and writes the IFC files.

Every reported result was produced on one desktop workstation: RTX 4060 Ti (16 GB),
i5-13400, 125 GB RAM. No cluster, no cloud, no multi-GPU. The final 161-copy round takes
about 11.2 hours end to end, of which the SDF auto-decoder is 4.0 h and the Graphormer
5 h to 6.5 h.

## Running the pipeline

Everything resolves paths relative to the repository root, so **run from the repository
root**, not from `scripts/`:

```bash
python scripts/main.py
```

`main.py` first asks whether to initialise a new run folder under `data/runs/` or to open
an existing one, then offers the task menu. The options that matter:

| Menu | Task |
|---|---|
| `1.2` | Data preprocessing, wall variants (the parametric sweep) |
| `2.2` | Train SDF embeddings with the hash grid and write them back into the IFC files |
| `3.2` | Generate the combinatorial graphs |
| `4` | Train the Graphormer and run the full test evaluation |
| `5.2` | The whole thing: `1.2` to `2.2` to `3.2` to `4`, plus holdout and archive |
| `6` | Retrain the Graphormer on existing graphs, reusing the embeddings |
| `6.2` | Retrain the GINE baseline on the same graphs (Experiment 5) |
| `7` | Evaluate only, from an existing checkpoint |
| `8` | Single-rule ablation, three balanced runs (Experiment 7) |

`5.2` is the path that produced the final model. The other scripts:

| Script | Purpose |
|---|---|
| `scripts/generate_graph_dataset.py` | Standalone graph generation from a base IFC |
| `scripts/res.py` | Full evaluation of a checkpoint with integrated-gradients attribution |
| `scripts/paper_figures.py` | Publication figures for one run, into `<run>/output/paper_figures/` |
| `scripts/paper_figures_extra.py` | The neighbour-context and rotation diagnostics |

## Reproducing the paper's numbers

Each experiment is self-contained under `paper_section/`. The numbers behind every figure
and table are already extracted into that directory's `data/` as a small JSON file, so the
figures can be redrawn from a fresh clone with no run archives:

```bash
python paper_section/exp4_rotation_scaling/make_figures.py
```

`make_figures.py` renders into `figures/` as PNG and PDF. `extract_data.py` is how those JSON
files were produced: it re-reads the evaluation reports and holdout graphs under `data/runs/`,
so it only runs on a machine that holds the archives. Each directory carries its own
`README.md`, and [data/README.md](data/README.md) tabulates which run backs which experiment.

## Caveats worth knowing before you touch anything

- **`data/` and `experiments/` paths are hardcoded and cwd-relative.** Roughly twenty call
  sites contain the literal strings `data/runs/...`, `data/input/input_ifcs/Data_base_1.ifc`
  and `experiments/...`. Moving either directory breaks the pipeline silently; running a
  script from anywhere other than the repository root does the same.
- **The five entry points in `scripts/` carry a short repo-root bootstrap block** that puts
  the repository root on `sys.path`, because they live one level down and import `from src...`.
  Do not remove it.
- **[data/README.md](data/README.md) records which parts of a run the paper needs** and which
  are regenerable bulk, should the run archives ever be deposited.
