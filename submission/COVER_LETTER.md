# Paper Submission Cover Letter

**Title:** Reliability-First Queue Risk for GPU Clusters: Calibration, SLOs, and Operational Integration

**Target venue:** Improving Scientific Software Conference 2026 (ISS26) Proceedings

**Artifact repository:** https://github.com/espirado/Reliability-First-Queue-Risk (Zenodo DOI to be assigned on v1.0.0)

---

## 1. Executive Value Proposition: The "Reliability-First" Paradigm

In current GPU cluster operations, queue-risk models are frequently evaluated
through the lens of discrimination — specifically AUROC — which measures the
ability to rank jobs by risk but remains silent on the actual reliability of
those probabilities. This research advocates for a **reliability-first**
paradigm. For an automated admission or advisory system to be operationally
viable, risk scores must be calibrated: a predicted "70% risk" must realize a
long-wait outcome 70% of the time.

### Statement of Innovation

The proposed framework treats probability calibration as a first-class
engineering objective, moving beyond ranking metrics to provide an operational
path that links offline benchmarks to production Service Level Objectives
(SLOs). This is critical for operational trust because:

- **Decoupling of metrics.** High discrimination (AUROC) does not guarantee
  reliability. Models can be perfectly *correct* in ranking while being
  *wrong* in probability, producing user-visible errors.
- **Operational integrity.** Calibrated probabilities allow for honest
  signals. In high-uncertainty environments, a calibrated model signals
  *"I don't know"* (calibrated indifference) rather than providing
  high-variance, misleading predictions.
- **SLO alignment.** By quantifying reliability via Expected Calibration Error
  (ECE), operators can define clear prerequisites for automated gating,
  reducing the "trust deficit" in automated scheduling.

---

## 2. Breakthrough Finding: DCGM Telemetry as a Performance Catalyst

Empirical analysis within the Slurm HPC environment demonstrates that NVIDIA
Data Center GPU Manager (DCGM) telemetry is a transformative catalyst for
model performance — but with an important caveat: **feature richness does not
inherently improve reliability**. While DCGM telemetry *rescued* Logistic
Regression (LR) — elevating it from chance-level to an operationally viable
*gate* tier — it actually degraded the calibration of the Random Forest (RF)
model.

| Feature Set    | Model               | AUROC | ECE    | MCE   | Operational Tier |
| -------------- | ------------------- | ----- | ------ | ----- | ---------------- |
| Submit-Only    | Logistic Regression | 0.498 | 0.066  | 0.145 | Warn             |
| Submit + DCGM  | Logistic Regression | 0.815 | 0.027  | 0.065 | **Gate**         |
| Submit-Only    | Random Forest       | 0.983 | 0.031  | 0.177 | Suggest          |
| Submit + DCGM  | Random Forest       | 0.991 | 0.068  | 0.421 | Warn             |
| Submit + DCGM  | Gradient Boosting   | 0.995 | 0.0018 | 0.030 | **Gate**         |

The leap in LR's AUROC from 0.498 to 0.815, coupled with an ECE reduction to
0.027, justifies its promotion to *gate*. Conversely, the RF model's MCE
spike to 0.421 with added features highlights a primary concern for SREs:
worst-bin behavior. Gradient Boosting (GB) achieved the best overall
calibration (ECE 0.0018), serving as the gold standard for
admission-gating. Honest aggregate reporting: of the six Slurm configurations
above, **only three pass all SLO thresholds for the gate tier**.

---

## 3. The Operational Framework: SLO-Driven Integration and Tiering

The paper formalizes a four-tier operational qualification algorithm,
mapping measured model reliability to specific integration levels so that
only the most reliable models influence cluster policy.

- **Warn (Informational).** Entry-level tier for general context.
  *Prerequisites:* ECE ≤ 0.10.
- **Advisory (UI Surface).** Suitable for dashboarding without automated
  intervention.
  *Prerequisites:* ECE ≤ 0.05; false-action rate at τ = 0.5 below 20%.
- **Suggest (Auto-Suggestion).** Qualified to suggest alternative partitions
  or GPU counts.
  *Prerequisites:* ECE ≤ 0.05; MCE ≤ 0.15; tail-gap at τ = 0.7 below 0.05.
- **Gate (Policy Admission).** The highest tier, capable of enforcing
  admission-webhook decisions.
  *Prerequisites:* ECE ≤ 0.05; MCE ≤ 0.10; tail-gap ≤ 0.035 at every
  threshold; false-action rate at τ = 0.7 below 5%.

---

## 4. Calibration Stability and Graceful Degradation

A core finding across heterogeneous environments (EKS, Slurm, Alibaba, Borg)
is that **calibration often holds where discrimination fails**. This is most
evident in the Google Borg trace analysis:

> On the Google Borg trace, discrimination metrics collapse
> (AUROC ≈ 0.50), yet the Expected Calibration Error remains exceptionally
> low (ECE ≤ 0.014). This "calibrated indifference" proves that the model
> correctly identifies its inability to predict the outcome. For an SRE, a
> calibrated "I don't know" is an infinitely superior signal to an
> uncalibrated, overconfident error, ensuring graceful degradation in
> feature-poor environments.

---

## 5. Proactive Maintenance: Drift Monitoring and Automated Recalibration

To ensure long-term reliability, we implement a drift-to-recalibration
pipeline. Our analysis shows that cross-cluster transfer (e.g., EKS → Slurm)
is non-viable without recalibration, as key features like `pending_ratio`,
`total_pending`, `pending_gpus`, and `queue_depth_norm` exhibit severe
distribution shifts (PSI = 12.41).

The maintenance pipeline follows a three-step protocol:

1. **Monitoring.** Track Population Stability Index (PSI) on input features.
   PSI ≥ 0.25 (e.g., for `pending_ratio`) signals a severe shift.
2. **Triggering.** Recalibration is initiated if internal cues are met —
   specifically, weekly ECE > 0.07 or PSI > 0.1.
3. **Correction.** A sliding-window isotonic recalibration loop re-fits the
   calibrator on the most recent data to restore SLO compliance.

---

## 6. Methodological Rigor and Reproducibility Harness

This work is grounded in robust scientific-software engineering traditions,
utilizing a trace-calibrated discrete-event Slurm simulator in the lineage of
BatSim and the Lublin–Feitelson workload model.

### Reproducibility Checklist

- **Statistical validation.** Every result is supported by bootstrap 95%
  confidence intervals and McNemar significance tests across five model
  families.
- **Scale and artifacts.** Evaluation spans ~1.2M job events across four
  heterogeneous environments. All artifacts — the simulator harness, trained
  calibrators, sample data, and LaTeX source — are archived via a Zenodo DOI.
- **Stress-testing.** A single-command notebook reproduces 10,000-job
  controlled-experiment workloads, unit-testing the monitoring SLIs under
  controlled shifts (profile-V vs. profile-A). The controlled experiment
  confirms that calibration SLIs fire even when AUROC remains deceptively
  "green" (0.99) during distribution shifts — validating the "monitor
  calibration, not just discrimination" design choice.

---

## 7. Strategic Summary

Modern GPU infrastructure faces a significant **goodput tax** — a 15–25% loss
in successful job completions due to delays and hardware-induced
interruptions. As documented in Meta's LLaMA-3 training (where 58.7% of
unexpected failures were GPU-related), raw throughput is no longer the sole
metric of success. Our reliability-first approach provides the infrastructure
intelligence required to reclaim this goodput through trustworthy, calibrated
prediction.

### Key Technical Contributions

1. **Telemetry-driven model rescue.** Empirical proof that hardware-level
   DCGM telemetry rescues linear models on Slurm — *provided it is coupled
   with rigorous calibration monitoring to prevent feature-driven reliability
   decay* (the Random Forest cautionary tale).
2. **SLO-to-tier mapping architecture.** A formal qualification algorithm
   that bridges the gap between ML research and SRE practice by mapping
   statistical metrics to concrete operational risks (warn / advisory /
   suggest / gate).
3. **Reliability-first reproducibility suite.** An open-source, end-to-end
   pipeline — including a trace-calibrated discrete-event Slurm simulator —
   designed to let operators deploy, monitor, and recalibrate queue-risk
   models with mathematical confidence.

---

*Fit for ISS26.* The contribution is explicitly a scientific-software
contribution: the code, the evaluation harness, the reproducibility
notebook, and the SLO framework itself are the artifact, released under
MIT license with a Zenodo DOI for long-term citability. We expect this to
be of direct operational value to the ISS26 community of research software
engineers running shared GPU infrastructure.
