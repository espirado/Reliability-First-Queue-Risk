# Paper 2 — Reproducibility Guide

Step-by-step instructions for a third-party reviewer to reproduce every
empirical claim in the ISS26 paper *Reliability-First Queue Risk for GPU
Clusters* from a fresh clone.

## 1. Environment

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate reliability-first-queue-risk
python -m ipykernel install --user --name reliability-first-queue-risk \
  --display-name "Python (reliability-first-queue-risk)"
```

### pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name reliability-first-queue-risk \
  --display-name "Python (reliability-first-queue-risk)"
```

Verify:

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('ok')"
```

## 2. Run the reproducibility notebook

```bash
jupyter lab notebooks/reproducibility.ipynb
```

Run all cells. The notebook is structured to mirror the paper:

| Step | Paper section | Output |
| --- | --- | --- |
| 0 | Setup | Paths, helpers, RNG seed |
| 1 | §III Data | Loads four sample CSVs |
| 2 | §IV Methodology | Re-runs feature engineering on the Slurm sample |
| 3 | §V.A Five-model benchmark | Reproduces Table IV (AUROC / AUPRC / Brier / ECE) |
| 4 | §V.B Slurm submit-only vs DCGM | Reproduces Tables V–VIII |
| 5 | §V.B Tier qualification | Reproduces Table VIII |
| 6 | §V.C Cross-domain PSI / temporal ECE | Reproduces Figure 5 + drift trigger |
| 7 | §IV.B Feature importance | Reproduces Table III + Figure 6 |
| 8 | All figures | Regenerates `figures/*.pdf` and `figures/*.png` |

End-to-end runtime: approximately **2–4 minutes** on a 2024-era laptop with
16 GB RAM.

## 3. Run experiments from the command line

```bash
python code/models/train_models.py --env eks_dec --out artifacts/eks_dec_run.json
python code/models/train_models.py --env slurm   --out artifacts/slurm_run.json
python code/models/train_models.py --env alibaba --out artifacts/alibaba_run.json
python code/models/train_models.py --env borg    --out artifacts/borg_run.json
python code/evaluation/tier_qualification.py \
  --inputs artifacts/eks_dec_run.json artifacts/slurm_run.json \
           artifacts/alibaba_run.json artifacts/borg_run.json \
  --out artifacts/all_5_models_results.csv
```

The output is deterministic for the seed shipped in `code/models/train_models.py`
(`SEED = 42`).

## 4. Build the paper PDF

```bash
cd tex
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  bash -lc 'pdflatex ieee_paper2.tex && pdflatex ieee_paper2.tex'
```

Or with a local TeX Live: `cd tex && pdflatex ieee_paper2.tex` twice.

The bibliography is `\bibitem`-embedded, so no `bibtex` invocation is needed.

## 5. Controlled scaling study

```bash
python code/simulation/run_scaling_study.py --jobs 10000 \
  --out artifacts/slurm_sim_metrics.json
```

This reproduces the controlled extension to 10 000 jobs cited in §VI.

## 6. Verifying claims numerically

After running the notebook, compare printed numbers against:

```bash
cat artifacts/all_5_models_results.csv | column -ts,
cat artifacts/bootstrap_confidence_intervals.csv | column -ts,
cat artifacts/mcnemar_5_models.csv | column -ts,
jq . artifacts/cross_domain_analysis.json | head -30
jq . artifacts/paper2_paper3_experiments.json | head -30
jq . artifacts/feature_importance.json | head -20
```

The mapping from each claim to the artifact that backs it is in
`docs/METHODOLOGY.md`.

## 7. Full data access

The repository ships with **anonymised samples** under `data/samples/`. Full
trace data lives outside the repository:

| Environment | Source | Notes |
| --- | --- | --- |
| EKS-Dec (1.2 M jobs) | Private collection (Saint Peter's University AWS account) | Schema in `docs/DATA_DICTIONARY.md`; full data not redistributed |
| Slurm + DCGM (555 jobs) | AWS ParallelCluster lab | Same |
| Alibaba GPU 2020 / 2023 | <https://github.com/alibaba/clusterdata> | Public |
| Google Borg 2019 | <https://github.com/google/cluster-data> | Public |

The notebook degrades gracefully on the samples and prints the full-trace
headline numbers (from `artifacts/`) for cross-reference.

## 8. Hardware

No GPU is required to reproduce the paper. The DCGM features in the Slurm
sample are pre-computed aggregates; no live DCGM endpoint is consulted at
training time.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ImportError: lightgbm` | Wrong Python | Use Python 3.11; `pip install lightgbm==4.3.*` |
| `KeyError: 'wait_threshold_p90'` | Stale sample | `git pull` and reload the CSVs |
| Bootstrap step is slow | Large B | Reduce `B = 1000` to `B = 200` for a quick smoke test |
| McNemar p-values do not match | RNG state drift | Restart kernel; the seed is fixed at top of step 0 |

If something else breaks, please open a GitHub issue with the full traceback
and your environment versions (`python -V; pip freeze | head`).
