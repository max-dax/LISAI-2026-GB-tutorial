# From one source to a catalog — multi-source NPE for LISA galactic binaries

Tutorial material by **Max Dax**, building directly on Stephen Green's
[LISAAI-Hackathon-ESTEC](https://github.com/stephengreen/LISAAI-Hackathon-ESTEC) tutorial.

Stephen's Part 2a showed that one frozen narrow-prior NPE network plus a heterodyne
frequency scan solves a single-source problem on a prior twenty times wider, with no
retraining — search and parameter estimation in a single pass. This tutorial asks the
question that trick begs: LISA's data stream holds of order $10^4$ resolvable galactic
binaries at once. **Part 1** injects dozens of sources — loud, marginal, and genuinely
undetectable — into one record and turns the same frozen network, from the same
checkpoint file, into a catalog pipeline: per-tile Bayes factors as the detection
statistic, a noise-calibrated threshold, exact-likelihood posteriors for every
detection, and a selection function scored against the injected truth. **Part 2**
replaces the injection list with a single galactic population prior — 30,000 binaries
whose faint bulk *is* the noise floor — and retrains the network once to infer the
loudest source per tile: P–P-calibrated posteriors, a catalog without importance
sampling, honestly bimodal near-ties, and iterative subtraction down to the confusion
limit.

## Quick start (Colab)

Open a notebook and **Runtime → Run all**; the first cells fetch the modules and the
trained networks from this repository.

| | | |
|---|---|---|
| Part 1 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/max-dax/LISAI-2026-GB-tutorial/blob/main/part1_scan_catalog.ipynb) | the frozen single-source network turned into a catalog pipeline |
| Part 2 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/max-dax/LISAI-2026-GB-tutorial/blob/main/part2_confusion.ipynb) | the same scan under a full galactic population |

Neither notebook trains anything live: Part 1 reuses Stephen's frozen network, Part 2
loads a pre-trained population network (choose its size with `MODEL` in the fetch
cell). Full runs take roughly five (Part 1) and fifteen (Part 2) minutes on a CPU
runtime, much less on a T4 (**Runtime → Change runtime type → T4 GPU**).

## Running on a laptop

With [uv](https://docs.astral.sh/uv/) installed:

```
git clone https://github.com/max-dax/LISAI-2026-GB-tutorial.git
cd LISAI-2026-GB-tutorial
uv run jupyter lab
```

Without uv: a fresh virtual environment (Python ≥ 3.10),
`pip install glasflow corner torch matplotlib scipy jupyter`, and open the notebook.

## Contents

| | |
|---|---|
| `part1_scan_catalog.ipynb` | Part 1 — a 40-binary record, the $W$-independent scan, detection, the catalog, and the resolvability limit |
| `part2_confusion.ipynb` | Part 2 — a 30,000-binary population, loudest-in-tile NPE, P–P calibration, a catalog without importance sampling, bimodal near-ties, iterative subtraction |
| `gb_catalog.py` | Part 1 machinery: injection settings, the band-limited scan, detection/matching, per-candidate posteriors; `python gb_catalog.py` runs a self-test |
| `gb_population.py` | Part 2 machinery: the population prior, band-local training-data generation, the population network; `python gb_population.py` runs a self-test |
| `train_confusion_flow.py` | trains the Part 2 network (documented CLI; CUDA auto-detected); the notebook only loads its checkpoint |
| `gb_simulator.py`, `gb_wide.py` | vendored **verbatim** from Stephen Green's repository (MIT) — the simulator and the conditioning/scan primitives |
| `checkpoints/part1_flow_v2.pt` | Stephen's frozen Part 1 network, byte for byte; nothing in this tutorial retrains it |
| `checkpoints/part2_population_flow_{s,m}.pt` | Part 2 population networks (neural spline flows), trained by `train_confusion_flow.py`: `s` (6x128, 8 MB) and `m` (8x256, 31 MB) ship in the repo; `l` (10x512, 116 MB, the best validation loss) is a [release asset](https://github.com/max-dax/LISAI-2026-GB-tutorial/releases). Select with `MODEL` in the Part 2 notebook's fetch cell |

## Provenance and license

MIT. `gb_simulator.py`, `gb_wide.py` and the checkpoint are © Stephen Green
(unmodified copies from
[LISAAI-Hackathon-ESTEC](https://github.com/stephengreen/LISAAI-Hackathon-ESTEC));
everything else © Max Dax. See `LICENSE`.
