# Reliability-First Queue Risk for GPU Clusters

> Calibration, SLOs, and Reproducible Operational Integration.
> ISS26 (Improving Scientific Software Conference 2026) companion artifact.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper artifact DOI](https://img.shields.io/badge/paper%20DOI-pending-lightgrey)](https://zenodo.org/communities/iss26)
[![Slides DOI](https://img.shields.io/badge/slides%20DOI-10.5281%2Fzenodo.19687976-blue)](https://doi.org/10.5281/zenodo.19687976)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/espirado/Reliability-First-Queue-Risk/HEAD?labpath=notebooks%2Freproducibility.ipynb)

This repository is the reproducible software artifact for the ISS26 paper
*Reliability-First Queue Risk for GPU Clusters: Calibration, SLOs, and Reproducible
Operational Integration*. It contains the full calibration-aware pipeline, the
benchmark results across four cluster environments (Amazon EKS, AWS ParallelCluster
Slurm with DCGM, Alibaba GPU traces, Google Borg), and a single Jupyter notebook
that regenerates every figure in the paper from sample data included in the repo.

## ISS26 companion materials

| Artifact | Identifier | Status |
| --- | --- | --- |
| This paper artifact (code, data, notebook, tex) | *DOI pending* | GitHub release [v1.0.0](https://github.com/espirado/Reliability-First-Queue-Risk/releases/tag/v1.0.0), Zenodo deposit pending ISS26 curator approval |
| Conference talk slides (VGAC: Predictive Queue Intelligence for GPU Cluster Observability) | [10.5281/zenodo.19687976](https://doi.org/10.5281/zenodo.19687976) | Submitted to the [ISS26 Proceedings](https://zenodo.org/communities/iss26) community |

## What is in here

```
paper2_repo/
├── tex/                  IEEE-format paper source (ieee_paper2.tex)
├── notebooks/            Single reproducibility notebook: every figure end-to-end
├── code/                 Pipeline modules (features, models, evaluation, ops)
├── artifacts/            Full results JSON / CSV with bootstrap CIs
├── figures/              PDF/PNG figures cited from the paper
├── data/samples/         Subsampled CSVs so the notebook runs from clone only
├── environment.yml       Conda environment (Python 3.11, pinned)
├── requirements.txt      pip alternative
├── CITATION.cff          Citation metadata
├── .zenodo.json          Zenodo community=iss26 metadata
└── LICENSE               MIT
```

## One-command reproduction

```bash
git clone https://github.com/espirado/Reliability-First-Queue-Risk.git
cd Reliability-First-Queue-Risk
conda env create -f environment.yml
conda activate reliability-first-queue-risk
jupyter lab notebooks/reproducibility.ipynb
```

or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/reproducibility.ipynb
```

The notebook loads `data/samples/*.csv`, retrains the five model families,
applies isotonic calibration, computes AUROC / AUPRC / ECE / MCE / Brier with
95% bootstrap CIs, assigns each model to one of four operational tiers
(warn / advisory / suggest / gate), computes PSI drift, and plots the
temporal-ECE trajectory — i.e., regenerates every figure in the paper.

Expected runtime: under 5 minutes on a laptop. No GPU required.

## How the artifact maps to the paper

| Paper section | File in this repo |
| --- | --- |
| § Methodology — Feature engineering | `code/features/feature_engineering.py` |
| § Methodology — Model zoo | `code/models/train_models.py` |
| § Methodology — Calibration | `code/models/calibrate.py` (invoked from notebook) |
| § Methodology — Tier qualification | `code/evaluation/tier_qualification.py` |
| § Methodology — Drift / recalibration | `code/ops/recalibration_loop.py` |
| § Benchmark Results (Table 2) | `artifacts/all_5_models_results.csv` + `artifacts/bootstrap_confidence_intervals.csv` |
| § DCGM on Slurm (Table 3) | `artifacts/paper2_paper3_experiments.json` (key `model_results`) |
| § Drift and Recalibration (Fig 5) | `artifacts/cross_domain_analysis.json` (keys `psi_top_features`, `temporal_ece`) |
| § Statistical significance | `artifacts/mcnemar_5_models.csv` |
| § VIII Controlled scaling experiment (Tables 4–7) | `code/simulation/slurm_simulator.py`, `code/simulation/run_scaling_study.py`, `artifacts/slurm_sim_*.json`, `artifacts/slurm_real_benchmark.json` |

Every claim in the paper cites the corresponding file in this repo; the
reproducibility notebook walks through them in the same order as the paper.

## Reproducing the controlled scaling experiment

The paper's §VIII (Controlled Scaling Experiment via Trace-Calibrated
Discrete-Event Simulation) is a **reproducibility harness** for the
calibration pipeline, not additional empirical evidence about real GPU
clusters or A100/H100 hardware. It runs the full pipeline at 10,000 jobs
under three controlled scenarios (baseline, heavy load, profile-A
telemetry shift) on a trace-calibrated discrete-event Slurm scheduler
model, and exists to confirm three software-engineering properties of the
pipeline: (a) the bootstrap variance from the real N=555 trace is
statistically conservative; (b) a median-threshold heuristic is
structurally insufficient even under model-favorable conditions; and
(c) the PSI / temporal-ECE monitoring SLIs fire under controlled
input-distribution shift even when discrimination remains intact (a unit
test for the monitoring design from §VI of the paper).

The experiment is calibrated on the real trace's IAT, service-time,
job-type, and telemetry distributions and enforces Slurm conservative
backfill. It is seeded and deterministic. AUROC values that approach
1.0 in the controlled experiment are a property of the harness's
full-observability conditions and are explicitly *not* claims about
real-cluster predictability — see §VIII.A in the paper for the full
scope-and-limits statement.

```bash
# From the repo root:
python code/simulation/run_scaling_study.py \
    --real data/samples/slurm_full_sample.csv \
    --n-jobs 10000 \
    --out-dir artifacts
```

Runtime: ~90 seconds on a laptop. Output artifacts:

```
artifacts/slurm_sim_baseline.csv         # full v100/t4 baseline trace
artifacts/slurm_sim_heavy.csv            # 2x arrival rate stress variant
artifacts/slurm_sim_a100.csv             # A100/H100 thermal profile
artifacts/slurm_sim_metrics.json         # Table 4 (AUROC/ECE/tier × scenario × model)
artifacts/slurm_sim_heuristic.json       # Table 5 input (heuristic baseline)
artifacts/slurm_sim_cross_profile.json   # Table 6 (calibration collapse: v100→A100)
artifacts/slurm_sim_leadtime.json        # Decision lead time for true alerts
artifacts/slurm_sim_fairness.json        # Table 7 (per-job-type parity)
artifacts/slurm_real_benchmark.json      # Apples-to-apples N=555 real benchmark
```

The notebook's §9 loads these artifacts and renders the paper tables
unchanged. Reviewers who want to reproduce at scale but on a different
real-trace calibration source can swap `--real` for any CSV containing
`submit_epoch_ms`, `job_type`, `gpu_nodes_alloc`, `wait_seconds`, and the
DCGM columns.

## Data access

Sample data (subsampled, anonymized) is included in `data/samples/` and is
sufficient to run the notebook end-to-end. For full-scale reproduction with
the source traces:

- **Alibaba v2020**: public at https://github.com/alibaba/clusterdata (trace `cluster-trace-gpu-v2020`).
- **Alibaba v2023**: public at https://github.com/alibaba/clusterdata (trace `cluster-trace-gpu-v2023`).
- **Google Borg 2019**: public at https://github.com/google/cluster-data.
- **EKS / Slurm**: collected on our own infrastructure; schema documented in
  `code/features/feature_engineering.py`. Access on request under a data-use
  agreement (contact the authors).

## Citation

See `CITATION.cff` for machine-readable metadata. Text citation:

> A. Espira, T. Dhole, B. Nagar, and S. Kumar,
> "Reliability-First Queue Risk for GPU Clusters: Calibration, SLOs, and
> Reproducible Operational Integration," in *Proc. Improving Scientific
> Software Conference (ISS26)*, Boulder, CO, USA, April 2026.
> DOI: `10.5281/zenodo.XXXXXXX`.

## License

MIT (see `LICENSE`). Data samples are released under the same license.
