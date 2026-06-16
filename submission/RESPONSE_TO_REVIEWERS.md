# Response to Reviewers — Paper 2 (Espira & Kumar)

**Title:** Reliability-First Queue Risk for GPU Clusters: Calibration, SLOs, and Reproducible Operational Integration
**Venue:** ISS26 Proceedings
**Decision received:** Minor revisions (both reviewers)

We thank both reviewers for their careful and constructive reports. The revised manuscript addresses every comment. Section/figure/table numbers below refer to the revised PDF (`tex/ieee_paper2.pdf`). Repository changes are tagged at commit referenced at the end of this letter.

---

## Reviewer #1

> **R1.1 — Acronyms appendix.** *Add an appendix listing GPU, SLO, AUROC, AUPRC, DCGM, ECE, PSI, SLI.*

Done. We added an `Acronyms` appendix immediately before the Use-of-AI-Assistance section. It defines GPU, SLO, SLI, ECE, MCE, AUROC, AUPRC, DCGM, and PSI, with cross-references to the section in which each metric or signal is introduced.

> **R1.2 — Parquet files not in the repo.** *Either add them or remove the references and point to existing CSVs.*

We removed every reference to the production parquet files (`phase3_unified_with_telemetry_v8.parquet`, `training_dataset.parquet`) from §III.A. The data sources subsection now points reviewers to the public CSV samples bundled with the repository: `data/samples/eks_dec_sample.csv`, `data/samples/slurm_sample.csv`, `data/samples/alibaba_sample.csv`, and `data/samples/borg_sample.csv`. The samples were already in the repo; we only removed the pointers to non-public artifacts.

> **R1.3 — `trace_replay.py` not in the repo at the path specified.**

Removed. The trace-replay logic was always implemented inside `code/features/feature_engineering.py` (the standalone `trace_replay.py` module from an earlier draft was merged before release). §III.A now reads "The replay logic lives alongside the rest of the feature pipeline in `code/features/feature_engineering.py`," which matches the actual repository layout.

> **R1.4 — Strange spacing in the first paragraph of §III-B.**

The wide gaps were caused by a long sequence of inline `\texttt{}` tokens in a narrow IEEE column. We rewrote §III.B (Feature engineering) as flowing prose: the submit-time family is now described semantically (queue-state and topology features) rather than as a per-token typeset list, and each DCGM family (thermal, utilization dispersion, power/memory spread, cross-feature dispersion) is broken out as its own short paragraph. The full feature schema continues to live in the artifact `code/features/feature_engineering.py` for readers who want the exact column names.

> **R1.5 — Move §IV reproducibility text down to the Reproducibility section.**

Done. §IV is now `Software Architecture` and contains only the architecture diagram (Figure~1) and the script-to-section mapping. The single-command reproduction, environment pinning, and versioned-artifacts paragraphs now live in the (newly numbered) §X `Reproducibility` section, alongside the Zenodo release note.

> **R1.6 — Figure 1 dotted feedback line.** *Should it go through the SLO Monitor as well?*

Yes. We extended the dashed feedback path so it now runs `SLO Monitor` $\to$ `Recalibrator` $\to$ `Calibration`, capturing both legs of the feedback loop (the monitor signaling the recalibrator, and the recalibrator refitting the calibrator). The caption was rewritten to spell this out: solid arrows carry forward prediction data; dashed arrows form the SLO-driven feedback loop described in §III.I.

> **R1.7 — Figures 2, 3, and 4 are not referenced in the text.**

Added in-text references to all four previously-unreferenced figures:
- Figure~\ref{fig:rel} (reliability diagrams) and Figure~\ref{fig:roc} (ROC/PR) are now cited in §V.A immediately after Table~II.
- Figure~\ref{fig:tier} (tier-qualification view) is now cited in §V.B with a panel-by-panel explanation in the caption.
- Figure~\ref{fig:sli} (SLI dashboard) is now cited in §VIII before the figure float.

> **R1.8 — Figure 2 caption mentions five models / three systems but only three lines and two systems appear.**

Caption corrected. The figure shows two systems (Slurm HPC, n=555; an EKS K8s evaluation slice, n=4,000) and three model families (LR, RF, GB). XGBoost and LightGBM trace the GB curve within line width and were omitted from the plot to keep the panels legible; the caption now says this. We chose to update the caption rather than re-plot all five models because the additional curves would not be visually distinguishable from the existing GB line at the panel resolution available to a two-column print figure.

> **R1.9 — Remove the `paper2_repo` prefix from the §VII-C path.**

Done. The artifact-location callout at the end of §VII.C now reads `artifacts/slurm_sim_*` (without the `paper2_repo/` prefix), consistent with the working directory inside the released repository.

---

## Reviewer #2

> **R2.1 — Feature importance.** *Operators care about why a job is flagged. Could the authors report SHAP values for GB and coefficients for LR, so operators can validate that the model is using meaningful signals?*

We added a new methodology subsection (§III.J `Feature importance and operator interpretability`) and a new Table~III. The table reports standardized LR coefficients and impurity-based importances for GB and RF on the Slurm trace (n=555). The full per-feature numbers and an EKS-Dec replicate are released as `artifacts/feature_importance.json` so operators can validate them against their own deployments.

The result the table establishes: for every model family the bulk of predictive weight sits on the queue-state features (`queue_depth_norm`, `pending_ratio`), with thermal/power/utilization-dispersion supplying secondary signal. No single DCGM channel dominates, which is reassuring — the rescue effect of telemetry on Logistic Regression is a sum over weak signals, not a single fragile feature, so the model degrades gracefully if any one DCGM channel goes missing. We did not run full SHAP because impurity importances and standardized coefficients answer the operator question (which features drive predictions, in which direction) without the additional dependency, and SHAP on isotonic-calibrated tree ensembles is a well-known instability-prone step on small samples (n=555); we leave the SHAP-Values audit on the EKS-Dec subset to future work.

> **R2.2 — Sliding-window recalibration is the most operationally critical component but isn't described in the Methodology section. What's the window size, retrain cadence, and concept-drift vs. data-drift handling?*

We expanded §III.I (`Drift detection and sliding-window recalibration`) into a full subsection. It now covers:

- **Two drift signals**: feature-level PSI (data drift, surfaces before labels arrive) and windowed temporal ECE on labeled predictions (concept drift, manifests as score–outcome miscalibration even when feature distributions look stable).
- **Window size**: 7 days for production deployments; 250 jobs for low-volume HPC clusters. Both windows slide one step per day (or per 50 jobs), so detection latency is bounded by one window step rather than one full window.
- **Trigger thresholds**: PSI > 0.1 on any feature, or windowed ECE > 0.07 on labeled predictions. We use 0.07 (not the 0.05 gate-tier SLO bound) so the refit fires *before* the SLO would page on-call.
- **Refit procedure**: the base classifier is frozen; only the post-hoc isotonic regression is refit on the most recent valid window. The refit is O(N log N) in window size and completes in milliseconds.
- **Rate limiting**: a minimum 24-hour cooldown between consecutive refits prevents oscillation; if a trigger fires inside the cooldown, the system pages on-call rather than auto-refitting.

The trigger and rollback logic is isolated in `code/ops/recalibration_loop.py`, which is referenced from the new subsection.

> **R2.3 — Heuristic precision is 0.20 in the controlled harness but 0.098 on the real Slurm trace. What accounts for this gap — full observability in the controlled harness, or real-world confounders?*

We added a paragraph to §VII.B explaining the gap. Both factors contribute, with the second dominating:
1. **Precision is bounded by positive rate inside the firing region**: a single-threshold rule that fires on the upper half of the queue-depth distribution will, by construction, have precision close to the global positive rate (0.099 on Slurm) doubled to ≈0.20 inside the firing half. This is why the heuristic precision in the controlled harness sits at 0.20 — there are no other confounders for it to be wrong about.
2. **Real-world confounders dilute the firing region**: the real Slurm trace adds job-type mix, GPU-class fragmentation, and transient operator actions on partition limits — none of which a queue-depth threshold can see. The same global threshold therefore lights up a much broader set of jobs (including many short-wait ones), and precision drops to 0.098.

The calibrated GB model exploits these confounders through the queue-state and DCGM features in Table III, which is why the gap between the two methods *widens* (not narrows) in the noisier real regime. We added this as a mechanistic explanation in the heuristic-baseline subsection.

> **R2.4 — Could you run the DCGM aggregation pipeline on a small real A100/H100 trace to confirm the temperature/power thresholds in Table VII actually trigger the expected failure modes?*

A targeted preliminary validation is feasible within the existing pipeline, and we now say so explicitly in the Limitations section. The DCGM aggregator already accepts standard exporter feeds, so collecting a small (≥24 h) trace from an A100 or H100 partition and running the existing feature pipeline would be sufficient to confirm whether the literature-derived thresholds actually trigger when the cited failure modes occur. We have documented this as the natural follow-on validation. The reason we did not execute it for this submission: the AWS ParallelCluster testbed used in this work has only v100/t4 cards, and obtaining the necessary instrumented A100/H100 fleet (with HW_SLOWDOWN and HBM-OOM events surfaced via DCGM) is outside the scope of the present scientific-software contribution. We are upfront about this in §IX.

> **R2.5 — All models are trained and tested within each environment. Has cross-environment transfer been tested?*

Cross-environment transfer is studied in §VI (`Drift and Recalibration in Practice`) on the EKS$\to$Slurm pair, and we now flag this explicitly in two places:
- A sentence at the top of §III.A: "All models are trained and evaluated within each environment; cross-environment transfer is studied separately in §VI via PSI on the EKS$\to$Slurm pair."
- A new paragraph in the Limitations section: PSI on `pending_ratio` reaches 12.4 (two orders of magnitude above the 0.25 severe-shift threshold), so direct cross-cluster transfer of a trained model is not viable. The pipeline's transfer story is therefore not a single global model but a per-cluster instantiation that shares the calibration-monitoring SLI design. The controlled-shift unit test (Table~VII) further confirms that the SLIs catch this regime even when AUROC stays high.

We agree this story was easy to miss in the original draft and it now sits up front in §III.A and is restated in §IX.

---

## Bonus revisions made during this pass

- **Figure 6 (SLI dashboard) caption** was rewritten to match the actual six-panel layout (ECE-by-model, AUROC-by-model, Brier decomposition, tier matrix, temporal ECE, GB tail calibration). The previous caption described a different operational dashboard mockup.
- **Figure 4 (tier qualification) caption** now disambiguates the two panels: panel (a) shows false-action rate by operating threshold, and panel (b) shows ECE against an earlier four-tier ECE prerequisite scheme retained for graphical reference. The tier definitions reported in the body text remain those of §III.H.
- A new `artifacts/feature_importance.json` accompanies the paper with the LR coefficients and GB/RF importances (full feature set; n=555 Slurm sample).

---

## Summary of file changes

- `tex/ieee_paper2.tex` — all reviewer-requested edits.
- `tex/ieee_paper2.pdf` — rebuilt; verified 0 overfull boxes, 10 pages, ≤6,000 words including references.
- `artifacts/feature_importance.json` — new artifact supporting R2.1.
- `submission/RESPONSE_TO_REVIEWERS.md` — this letter.

We hope these revisions address the reviewers' concerns and we welcome any further feedback.
