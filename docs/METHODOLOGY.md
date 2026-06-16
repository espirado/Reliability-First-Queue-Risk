# Paper 2 — Claim ↔ Code ↔ Artifact Map

A line-by-line trace from every empirical claim in the ISS26 paper to the
code that produces it and the artifact that records it. Use this as the
primary reproducibility-audit document.

Paths are relative to the repository root.

## 1. Five-model benchmark across four environments (Tables IV, V)

| Claim | Code | Artifact |
| --- | --- | --- |
| LR / RF / GB / XGB / LightGBM trained on EKS-Dec, Slurm, Alibaba, Borg with isotonic calibration | `code/models/train_models.py` driven by the notebook | `artifacts/all_5_models_results.csv` |
| 95 % bootstrap CIs (B = 1000 percentile) for AUROC / AUPRC / Brier / ECE | `code/evaluation/` (bootstrap helpers) | `artifacts/bootstrap_confidence_intervals.csv` |
| Pairwise McNemar with Holm-Bonferroni correction | `code/evaluation/` | `artifacts/mcnemar_5_models.csv` |
| Test-fold sizes per environment: EKS-Dec n = 103, Slurm n = 167, Alibaba n = 853, Borg n = 4 000 | `code/models/train_models.py` (StratifiedKFold) | `data/samples/*_sample.csv` and the rebuilt `all_5_models_results.csv` |
| Positive rates: Slurm 9.9 %, EKS-Dec 34.0 %, Alibaba 10.0 %, Borg 9.8 % | Computed at load time | Same |

## 2. Slurm submit-only vs DCGM-enriched (Section V.B, Tables V–VIII)

| Claim | Code | Artifact |
| --- | --- | --- |
| Submit-only GB qualifies for Tier 4 (Gate); RF for Tier 3 | `code/evaluation/tier_qualification.py` | `artifacts/paper2_paper3_experiments.json`, key `submit_only` |
| DCGM-enriched models hold Tier 4 with tighter ECE | Same | Same, key `dcgm_enriched` |
| RF + DCGM misses gate ECE threshold by 1.4× | Same | Same |
| Tail calibration at $\theta \in \{0.5, ..., 0.9\}$ within tier-specific tail-gap bounds | Same | Same, key `tail_calibration` |

The paper claim that "ECE multiplier was 1.4×" (not 3.4×) is verified directly
against the JSON; this was a correction made in the camera-ready (see
`submission/RESPONSE_TO_REVIEWERS.md`).

## 3. EKS-Dec calibration claims (Section V.B)

| Claim | Code | Artifact |
| --- | --- | --- |
| EKS-Dec models meet ECE prereq.; MCE exceeds Gate threshold; therefore models qualify for *Advisory* tier with full evaluation reserved as Gate-tier evidence | `code/evaluation/tier_qualification.py` | `artifacts/all_5_models_results.csv`, EKS-Dec rows + `artifacts/paper2_paper3_experiments.json` |

## 4. Cross-domain analysis (Section V.C)

| Claim | Code | Artifact |
| --- | --- | --- |
| PSI on `pending_ratio` reaches 12.4 across EKS-Dec → Slurm | `code/data_analysis.py` (PSI helpers) | `artifacts/cross_domain_analysis.json`, key `psi_per_feature` |
| Temporal ECE windows: 8 windows, recalibration trigger at window 7 (ECE = 0.0145) | Same | Same, key `temporal_ece` |

## 5. Feature importance (Section IV.B, Table III)

| Claim | Code | Artifact |
| --- | --- | --- |
| LR standardised coefficients, GB / RF feature importances on the 555-job Slurm sample | `code/models/train_models.py` and the inline helper used for the camera-ready (`scripts/feature_importance.py` if regenerating) | `artifacts/feature_importance.json` |
| Top features: `pending_ratio`, `pending_gpus`, `temp_c_max` | Same | Same |

## 6. Drift detection and recalibration (Section IV.C)

| Claim | Code | Artifact |
| --- | --- | --- |
| Sliding-window recalibration triggered on PSI > 0.1 OR weekly_ECE > 0.07 | `code/ops/recalibration_loop.py` | `artifacts/cross_domain_analysis.json` and the temporal-ECE traces |
| Window length 7–14 days per namespace / queue; rate-limited to one refit per 24 h | Same | --- |
| Rollback if bootstrap-CI lower bound on tier qualification regresses | Same | --- |

## 7. Controlled scaling study (Section VI)

| Claim | Code | Artifact |
| --- | --- | --- |
| Scaling experiment on 10 000-job synthetic Slurm extension | `code/simulation/run_scaling_study.py` | `artifacts/slurm_sim_*.csv` and `artifacts/slurm_sim_*.json` |
| Bootstrap variance estimates remain within 95 % CIs of the 555-job real benchmark | Cross-check in the notebook | `artifacts/slurm_real_benchmark.json` |

## 8. Operational case study (Section VII)

| Claim | Code | Artifact |
| --- | --- | --- |
| Tier 4 (gate) admission webhook integration | `code/ops/recalibration_loop.py` (the policy hooks) and the companion Sentinel repo | --- |
| Decision lead time, goodput tax | Discussion-only in the paper | --- |

## 9. Reproducibility contract

`notebooks/reproducibility.ipynb` reproduces every table and figure in the
paper from `data/samples/*.csv` and the artifacts above. The notebook is the
canonical entry point; this file lists what each cell verifies. Numerical
disagreement larger than the third decimal place is a bug — please file
an issue.
