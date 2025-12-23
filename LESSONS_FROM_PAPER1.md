# Lessons from Paper 1: What Went Wrong & How to Fix in Paper 2

## Executive Summary

Paper 1 achieved its core goal—demonstrating submit-time prediction is possible—but reviewer feedback exposed critical weaknesses. Paper 2 must address these to achieve NVIDIA/Meta-level quality.

---

## 🔴 Critical Issues (Blockers)

### 1. Cross-Domain Transfer Failure
**Paper 1 Problem:**
- Alibaba → EKS: 0.286 AUC (worse than random!)
- EKS → Alibaba: 0.532 AUC (barely better than random)
- This was buried with one sentence of discussion

**Root Cause:**
- Feature distributions are completely different (EKS: 2-4 cores, Alibaba: 48-192 cores)
- Label definitions differ (EKS: >5min, Alibaba: top 10% unknown threshold)
- EKS-Nov had 99.9% zero CPU/memory requests (system pods)

**Paper 2 Solution:**
- [ ] Document exact label thresholds in both datasets
- [ ] Report JS-divergence for all features across datasets
- [ ] Try domain adaptation techniques:
  - Per-dataset normalization
  - Adversarial domain adaptation (DANN)
  - Multi-source domain adaptation
- [ ] Dedicate 2-3 pages to cross-domain analysis
- [ ] If transfer still fails, be honest and explain why

### 2. Dataset Inconsistencies
**Paper 1 Problem:**
- Different EKS counts in Table 1, Section VI.A, Table 8
- "December 2025" dates (typo)
- Borg: claimed 15,430 jobs but used 12,271

**Paper 2 Solution:**
- [ ] Single source of truth: Table 1 with exact counts
- [ ] All references point to Table 1
- [ ] Include filtering steps: "Starting from N, we removed X incomplete records, Y duplicates, leaving Z"
- [ ] Triple-check all dates before submission

### 3. GPU Features Missing (Title Mismatch)
**Paper 1 Problem:**
- Title: "GPU Cluster Scheduling"
- Features: Only `req_gpu` (submit-time)
- No runtime GPU telemetry

**Paper 2 Solution:**
- [ ] Title explicitly mentions "Runtime Telemetry"
- [ ] 15+ GPU-specific features from DCGM:
  - Temperature, power, utilization
  - ECC errors, XID events
  - Memory fragmentation
  - Thermal throttling
- [ ] Ablation study: submit-time vs submit-time + telemetry

---

## 🟡 High Priority Issues

### 4. Statistical Rigor
**Paper 1 Problem:**
- McNemar's test without contingency table
- Table 5 had identical F1/Precision/Recall (suspicious)
- No confidence intervals on key metrics

**Paper 2 Solution:**
- [ ] Report contingency tables for McNemar's
- [ ] Use 4 decimal places for AUROC/AUPRC
- [ ] Bootstrap 95% CIs for all headline metrics
- [ ] DeLong test for AUC comparisons
- [ ] Friedman test + Nemenyi for multi-model comparison

### 5. Calibration Analysis Shallow
**Paper 1 Problem:**
- ECE reported without binning details
- Isotonic regression concerns not addressed
- No reliability diagrams in paper

**Paper 2 Solution:**
- [ ] Specify: 10 equal-width bins, 15-bin sensitivity analysis
- [ ] Include reliability diagrams as figures
- [ ] Compare: Platt, Isotonic, Temperature scaling, Beta calibration
- [ ] Report Brier decomposition: reliability + resolution + uncertainty

### 6. Production Section Hand-Wavy
**Paper 1 Problem:**
- Vague deployment descriptions
- "MacBook Pro M2" mentioned (unprofessional)
- No real latency/throughput measurements

**Paper 2 Solution:**
- [ ] Remove all laptop references → "commodity hardware"
- [ ] Actual latency benchmarks: P50, P95, P99
- [ ] Memory footprint measurements
- [ ] Throughput: predictions/second
- [ ] Failure modes and error handling

---

## 🟢 Medium Priority Issues

### 7. Related Work Gaps
**Paper 1 Problem:**
- Missing GPU scheduling papers (Gandiva, Tiresias, Pollux)
- No positioning against recent work

**Paper 2 Solution:**
- [ ] Add GPU scheduling section with Gandiva, Tiresias, Pollux, AntMan
- [ ] Compare our approach explicitly
- [ ] Cite recent arXiv papers (2023-2024)

### 8. Figures/Tables Quality
**Paper 1 Problem:**
- Placeholder figures
- Dashboard screenshot in paper (unprofessional)
- Some figures not generated from code

**Paper 2 Solution:**
- [ ] All figures from reproducible notebooks
- [ ] Remove dashboard screenshots
- [ ] Consistent style (matplotlib rcParams)
- [ ] Vector graphics (PDF/SVG) not PNG

### 9. Terminology Inconsistency
**Paper 1 Problem:**
- "long-wait" vs "long_wait"
- "submit-time" vs "submission-time"
- Inconsistent capitalization

**Paper 2 Solution:**
- [ ] Style guide before writing
- [ ] Automated linting for terminology
- [ ] Consistent hyphenation throughout

---

## 📋 Paper 2 Pre-Submission Checklist

### Data Quality
- [ ] All datasets have documented filtering steps
- [ ] No impossible values (negative times, future dates)
- [ ] Feature distributions analyzed and reported
- [ ] Missing value handling documented

### Statistical Claims
- [ ] Every claim has p-value or CI
- [ ] Effect sizes reported, not just significance
- [ ] Multiple comparison correction applied
- [ ] Contingency tables for paired tests

### Reproducibility
- [ ] Code repository cleaned and documented
- [ ] Requirements.txt with exact versions
- [ ] Sample data for verification
- [ ] Claims verification script passes

### Writing Quality
- [ ] No overclaims ("consistently" → "often")
- [ ] Limitations discussed honestly
- [ ] No laptop/casual references
- [ ] Consistent terminology throughout

### Figures
- [ ] All figures generated from code
- [ ] High resolution (300+ DPI)
- [ ] Colorblind-friendly palette
- [ ] Legends readable at print size

---

## 🎯 Paper 2 Differentiation Strategy

| Aspect | Paper 1 | Paper 2 |
|--------|---------|---------|
| Features | Submit-time only | Submit-time + GPU telemetry |
| Datasets | 3 (with issues) | 2 (clean, well-documented) |
| Cross-domain | Mentioned, failed | Analyzed deeply, explained |
| Calibration | Post-hoc | Online + drift detection |
| Production | Vague | Real measurements |
| Reproducibility | Limited | Complete package |

---

## 📚 Key Papers to Cite

1. **GPU Scheduling**
   - Gandiva (OSDI'18) - Time-slicing
   - Tiresias (NSDI'19) - LAS scheduling
   - Pollux (OSDI'21) - Adaptive scheduling
   - AntMan (OSDI'20) - Memory sharing

2. **Calibration**
   - Guo et al. (ICML'17) - Modern calibration
   - Naeini et al. (AAAI'15) - ECE definition
   - Kumar et al. (NeurIPS'19) - Verified calibration

3. **Production ML**
   - Borg (EuroSys'15) - Google's scheduler
   - Fuxi (VLDB'14) - Alibaba's scheduler
   - Twine (OSDI'20) - Meta's workloads

4. **Domain Adaptation**
   - DANN (JMLR'16) - Adversarial DA
   - CORAL (ECCV'16) - Correlation alignment







