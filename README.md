# Reliability-First Queue Risk for GPU Clusters

**Calibration, SLOs, and Operational Integration**

This repository contains the research artifacts for our paper on reliability-first queue risk prediction for GPU clusters.

## Overview

Building on our [calibration benchmark work](https://github.com/espirado/Calibrated-Queue-Delay-Prediction), this paper focuses on the **operational integration** of calibrated predictions into real scheduling systems.

### Key Contributions

1. **Reliability-First Formulation**: We treat calibration quality (ECE, MCE, Brier decomposition) as a first-class objective, not an afterthought.

2. **Operational Stack**:
   - Reliability monitors for production deployment
   - Sliding-window recalibration for drift detection
   - SLO-based triggers and automated rollbacks

3. **Scheduler Integration**:
   - Kubernetes admission webhook patterns
   - Slurm job submit plugin design
   - Per-namespace threshold configuration

4. **Cross-Cluster Analysis**: 
   - Transfer limits between clusters
   - Per-cluster recalibration strategies
   - Slice-aware thresholds

## Repository Structure

```
.
├── tex/                    # LaTeX source for the paper
│   └── ieee_paper2.tex
├── code/
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and calibration
│   └── evaluation/        # Metrics and analysis
├── data/
│   ├── raw/               # Original data files
│   ├── processed/         # Cleaned datasets
│   └── samples/           # Sample data for testing
├── artifacts/
│   ├── calibration/       # Calibration artifacts
│   ├── checkpoints/       # Model checkpoints
│   └── metrics/           # Evaluation metrics
├── figures/               # Paper figures
├── notebooks/             # Jupyter notebooks for analysis
└── reviews/               # Peer review responses
```

## Relationship to Paper 1

| Paper 1 (Benchmark) | Paper 2 (This Work) |
|---------------------|---------------------|
| Calibration methods comparison | Production integration patterns |
| Multi-dataset evaluation | SLO-driven operations |
| Design rules derivation | Reliability monitoring |
| Negative results (Borg) | Cross-cluster transfer |

## Author

**Andrew Espira** (ORCID: [0009-0002-9196-8094](https://orcid.org/0009-0002-9196-8094))

Department of Data Science, Saint Peter's University

## Status

🚧 **Work in Progress** - This paper builds on the benchmark results from Paper 1 and focuses on production deployment patterns.

## Related Work

- [Paper 1: Calibrated Queue Delay Prediction](https://github.com/espirado/Calibrated-Queue-Delay-Prediction)
- [VGAC: GPU Cluster Observability Platform](https://github.com/espirado/vgac)

## License

MIT License - see LICENSE file for details.
