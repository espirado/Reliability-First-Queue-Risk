# Paper 2 Pipeline — Architecture

This document describes the calibration-aware queue-risk pipeline that the
ISS26 paper *Reliability-First Queue Risk for GPU Clusters* documents and
benchmarks. It is a scientific-software artifact: every component is a
standalone Python module exercised by `notebooks/reproducibility.ipynb`.

## 1. End-to-end data flow

```
+-----------+      +------------+      +-------------+      +-------------+
| Cluster   |      | Submit-time|      | Feature     |      | Model zoo   |
| traces    | ---> | capture    | ---> | engineering | ---> | + isotonic  |
| (4 envs)  |      | (5s poll)  |      | (38 cols)   |      | calibration |
+-----------+      +------------+      +-------------+      +-------------+
                                                                   |
                                                                   v
+--------------+   +-----------+   +-------------+    +----------------+
| Sliding-     |<--| Recalib.  |<--| SLO monitor |<---| 4-tier         |
| window       |   | trigger   |   | (ECE, MCE,  |    | qualification  |
| recalibrator |-->| (PSI/ECE) |   |  Brier)     |    | (gate/.../warn)|
+--------------+   +-----------+   +-------------+    +----------------+
```

The pipeline is acyclic on a per-job basis but circular under operations: as
soon as a job's true wait time is observed, it feeds the rolling SLO monitor
and may trigger recalibration of the calibrator (not the base model) in a
sliding window.

## 2. Layer-by-layer

### 2.1 Submit-time capture

Identical to the methodology in the VGAC PEARC '26 paper: 5 s polling,
phase-transition detection (`Pending → Running → Succeeded`), record cluster
state at the moment of submission.

Code: `code/features/feature_engineering.py` (the cluster-state aggregation
helpers); the polling collector itself is environment-specific (Kubernetes
informer / Slurm `sacct` polling) and not redistributed.

### 2.2 Feature engineering

Two feature families:

- **Submit-time scheduler-agnostic features** — `pending_ratio`,
  `queue_depth_norm`, `fragmentation_score`, `congestion_score`,
  `total_pending`, `pending_gpus`, `running_gpus`, `gpu_nodes_alloc`,
  `gpu_nodes_total`, `job_type_encoded`.
- **DCGM telemetry features (Slurm + EKS only)** — per-cluster aggregates
  at submit time: `util_gpu_{mean,std,max}`, `util_mem_{mean,std,max}`,
  `temp_c_{mean,max}`, `power_w_{mean,max}`, `mem_used_mb_{mean,max}`,
  derived `power_efficiency` and `mem_pressure`.

Code: `code/features/feature_engineering.py`.

### 2.3 Label construction

Binary `label_long_wait = (wait_seconds > P90_per_dataset)`. P90 thresholds
are pre-computed and stored alongside features (`wait_threshold_p90` column
in the parquet).

### 2.4 Model zoo

Five model families with **identical hyperparameters across environments**
(deliberately — to test cross-scheduler portability of a single
configuration):

| Model | Configuration |
| --- | --- |
| Logistic Regression | L2, C = 1.0, `class_weight='balanced'` |
| Random Forest | `n_estimators=300, max_depth=12` |
| Gradient Boosting | `n_estimators=200, lr=0.05, max_depth=4` |
| XGBoost | matched shape, `scale_pos_weight` from train ratio |
| LightGBM | `num_leaves=63, lr=0.05` |

Code: `code/models/train_models.py`.

### 2.5 Calibration

Post-hoc isotonic regression on an out-of-fold validation split, wrapped in
`CalibratedClassifierCV(cv='prefit')` to preserve train/test discipline.
Platt scaling is run as a sensitivity check.

Code: imported from `sklearn.calibration` directly.

### 2.6 Evaluation metrics

| Metric | Role |
| --- | --- |
| AUROC | Ranking |
| AUPRC | Ranking under imbalance |
| Brier score (Murphy decomposition) | Joint, with reliability/resolution/uncertainty |
| ECE (15 equal-mass bins) | Calibration |
| MCE | Worst-bin calibration |
| Tail calibration at $\\theta \in \{0.5, 0.6, 0.7, 0.8, 0.9\}$ | Operational tier reliability |
| McNemar test (Holm-Bonferroni) | Pairwise model comparison |

Code: `code/evaluation/`.

### 2.7 Tier qualification

Four operational tiers, each with an SLO triple:

| Tier | Action | ECE max | MCE max | Tail-gap max |
| --- | --- | --- | --- | --- |
| 1 — Warn | Informational | 0.10 | --- | --- |
| 2 — Advisory | Surface in UI | 0.07 | 0.20 | --- |
| 3 — Suggest | Auto-suggest reroute | 0.05 | 0.10 | 0.035 |
| 4 — Gate | Admission control | 0.03 | 0.06 | 0.020 |

Code: `code/evaluation/tier_qualification.py`.

### 2.8 Drift detection

PSI on submit-time features over sliding windows; weekly rolling ECE on
labelled outcomes. Trigger rule:

> recalibrate when `PSI > 0.1` OR `weekly_ECE > 0.07`.

Code: integrated into the rolling recalibration loop in `code/ops/recalibration_loop.py`.

### 2.9 Sliding-window recalibrator

Refits the **calibrator only** (not the base model) on a rolling window of
labelled outcomes. Defaults: window = 7-14 days per namespace/queue, refit
cadence = on trigger (rate-limited to once per 24 h). Falls back to the prior
calibrator if the new fit's bootstrap-CI lower bound on tier qualification
gets *worse*.

Code: `code/ops/recalibration_loop.py`.

## 3. Companion artifact: the simulator

`code/simulation/slurm_simulator.py` implements a trace-driven Slurm
scaling-study harness used in the controlled scaling experiments. Outputs
under `artifacts/slurm_sim_*.csv` and `slurm_sim_*.json`.

## 4. Testing assumptions

The scientific-software question — *does this pipeline behave correctly?* —
is answered by:

1. The notebook regenerating every figure in the paper from sample data.
2. The bootstrap CIs in `artifacts/bootstrap_confidence_intervals.csv`.
3. Pairwise McNemar tests (`artifacts/mcnemar_5_models.csv`).
4. The cross-environment PSI / temporal-ECE evidence
   (`artifacts/cross_domain_analysis.json`).

If any of these disagree with the paper, that is a bug — please file an
issue.

## 5. Out of scope

- **Online learning** — the calibrator is updated online; the base classifier
  is batch-trained.
- **Feature collection beyond what DCGM exposes** — eBPF / SDC root-causing
  is referenced in the literature but not part of this artifact.
- **Cross-tenant cost models** — the SLO triples are fixed; per-tenant
  costing is future work.
