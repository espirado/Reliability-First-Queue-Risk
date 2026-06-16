# Paper 2 — Data Dictionary

Schema documentation for every CSV / JSON in `data/samples/` and `artifacts/`.
SI units throughout unless otherwise stated.

## 1. `data/samples/eks_dec_sample.csv`

EKS-Dec submit-time features. One row per Pod create event. Sampled
deterministically from the full 1.2 M-row trace.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `pending_ratio` | float | ratio | Pending GPUs / total cluster GPUs at submit time |
| `queue_depth_norm` | int | jobs | Pending-job count, normalised |
| `fragmentation_score` | float | [0,1] | Heuristic fragmentation across GPU nodes |
| `congestion_score` | float | [0,1] | Composite queue-vs-utilisation congestion proxy |
| `pending_gpus` | int | GPUs | Sum of GPU requests across pending jobs |
| `total_pending` | int | jobs | Total pending jobs at submit time |
| `running_gpus` | int | GPUs | Sum of GPU allocations across running jobs |
| `gpu_nodes_alloc` | int | nodes | GPU nodes with at least one allocation |
| `gpu_nodes_total` | int | nodes | Total GPU nodes online |
| `label_long_wait` | int | 0/1 | 1 iff observed wait > P90 wait for this environment |
| `raw_score` | float | [0,1] | Pre-calibration model score |

## 2. `data/samples/slurm_sample.csv` and `slurm_full_sample.csv`

Slurm + DCGM samples. Schema is the EKS-Dec schema **plus** the DCGM
aggregates:

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `util_gpu_mean` | float | % | Cluster-mean SM utilisation at submit time |
| `util_gpu_std` | float | % | Cross-GPU std-dev of SM utilisation |
| `util_mem_mean` | float | % | Cluster-mean memory bus utilisation |
| `temp_c_max` | float | °C | Hottest GPU at submit time |
| `power_w_mean` | float | W | Mean GPU power draw |

The Slurm full-sample (`slurm_full_sample.csv`) ships all 38 features used in
training. The compact sample (`slurm_sample.csv`) ships the subset used by
the notebook for the Table V–VIII reproduction.

## 3. `data/samples/alibaba_sample.csv` and `borg_sample.csv`

Submit-time features for the public Alibaba 2020/2023 GPU traces and Google
Borg 2019. Same schema as EKS-Dec, no DCGM. `pending_ratio` is reconstructed
by trace replay (counting concurrent pending jobs at each row's submit
timestamp).

## 4. `data/samples/benchmark_5models.csv`

Pre-computed long-form benchmark table consumed directly by the notebook
when sample retraining is skipped:

| Column | Description |
| --- | --- |
| `env` | One of `eks_dec`, `slurm`, `alibaba`, `borg` |
| `model` | One of `lr`, `rf`, `gb`, `xgb`, `lgbm` |
| `auroc`, `auprc`, `brier`, `ece`, `mce` | Metrics |
| `n_test`, `seed` | Test fold size, RNG seed (= 42) |

## 5. `data/samples/bootstrap_cis.csv`

Bootstrap 95 % CIs (B = 1000 percentile) for the metrics in
`benchmark_5models.csv`.

| Column | Description |
| --- | --- |
| `env`, `model` | Identifiers |
| `metric` | `auroc`/`auprc`/`brier`/`ece` |
| `point` | Point estimate |
| `lower_95`, `upper_95` | Percentile CI bounds |

## 6. `data/samples/mcnemar_5models.csv`

Pairwise McNemar tests with Holm-Bonferroni correction.

| Column | Description |
| --- | --- |
| `env`, `model_a`, `model_b` | Comparison |
| `b_only`, `a_only` | Discordant counts |
| `chi2`, `p_raw`, `p_holm` | Statistic and corrected p |

## 7. `data/samples/drift_metrics.json`

Rolling drift trace.

```
{
  "window_index": [...],
  "psi":          [...],
  "rolling_ece":  [...],
  "rolling_brier":[...],
  "trigger":      [false, ..., true, ...]
}
```

A `trigger == true` value indicates the recalibration rule fired
(`PSI > 0.1` OR `weekly_ECE > 0.07`).

## 8. `artifacts/all_5_models_results.csv`

Same schema as `benchmark_5models.csv` but computed on the *full* trace
data; these are the headline numbers in the paper. Use this for cross-reference
when the notebook is run on samples.

## 9. `artifacts/feature_importance.json`

```
{
  "sample_n": 555,
  "positive_rate": 0.099,
  "features": [...],
  "lr_standardized_coefficients": { "<feature>": <float>, ... },
  "gb_feature_importances":       { "<feature>": <float>, ... },
  "rf_feature_importances":       { "<feature>": <float>, ... },
  "note": "Computed on data/samples/slurm_sample.csv (n=555). ..."
}
```

This is the artifact backing Table III of the camera-ready (added in the
revision cycle in response to reviewer feedback).

## 10. `artifacts/paper2_paper3_experiments.json`

Top-level keys:

- `submit_only`     — Slurm 5-model submit-only metrics + tier qualification.
- `dcgm_enriched`   — Same with the DCGM features added.
- `tier_qualification` — Per-model tier label assignment with the rules in
  `code/evaluation/tier_qualification.py`.
- `tail_calibration` — ECE evaluated at thresholds {0.5, ..., 0.9}.
- `false_action_rates` — FAR per tier.

## 11. `artifacts/cross_domain_analysis.json`

Top-level keys:

- `psi_per_feature` — PSI values across each environment pair.
- `temporal_ece` — `{ "windows": [...], "ece": [...], "trigger": [...] }`.
- `transfer_matrix` — `{ "<src>_to_<dst>": { "auroc_gap": ..., "ece_ratio": ... } }`.

`psi_per_feature.pending_ratio` reaches 12.4 across `eks_dec → slurm`.

## 12. `artifacts/slurm_sim_*.csv` and `slurm_sim_*.json`

Outputs of the controlled scaling study (`code/simulation/run_scaling_study.py`).
File-name suffix indicates which baseline / variant:

| Suffix | Meaning |
| --- | --- |
| `slurm_sim_baseline.csv` | FIFO baseline |
| `slurm_sim_heavy.csv` | High-load run |
| `slurm_sim_a100.csv` | A100 GPU mix |
| `slurm_sim_metrics.json` | Aggregate metrics |
| `slurm_sim_fairness.json` | Fairness numbers |
| `slurm_sim_cross_profile.json` | Cross-profile transfer |
| `slurm_sim_heuristic.json` | Heuristic-baseline numbers |
| `slurm_sim_leadtime.json` | Decision-lead-time numbers |

## 13. Notes on missing values and de-duplication

- EKS-Dec rows are de-duplicated on `(pod_uid, transition_timestamp)` to
  remove double-emission caused by Kubernetes informer re-deliveries.
- Slurm rows are not de-duplicated (`job_id` is unique).
- Missing DCGM aggregates are imputed with 0.0; absence of telemetry is
  itself signal (typically a cold node).
- Alibaba / Borg `pending_ratio` is reconstructed from the trace itself, not
  a real queue-state snapshot; this is documented as a limitation in the
  paper.
