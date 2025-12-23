# Paper 2: Extensive Data Collection Plan

## 🎯 Goal
Collect **production-grade telemetry data** that enables:
1. Training models with GPU runtime features
2. Validating cross-domain transfer improvements
3. Demonstrating production reliability
4. Supporting all claims with statistical rigor

---

## 📊 Data Requirements

### Minimum Dataset Size
| Dataset | Jobs | Duration | Purpose |
|---------|------|----------|---------|
| Training | 50,000+ | 1 week | Model learning |
| Validation | 10,000+ | 2 days | Hyperparameter tuning |
| Test | 10,000+ | 2 days | Final evaluation |
| Transfer Test | 5,000+ | 1 day | Cross-domain validation |

### Required Feature Coverage

#### Submit-Time Features (12 features)
- [x] `req_gpu` - GPU request count
- [x] `req_cpu_m` - CPU request (millicores)
- [x] `req_mem_mb` - Memory request (MB)
- [x] `priority_class` - K8s priority class
- [x] `priority_value` - Numeric priority
- [x] `has_node_selector` - Placement constraint
- [x] `has_affinity` - Affinity rules
- [x] `num_tolerations` - Tolerations count
- [x] `container_count` - Multi-container jobs
- [x] `namespace` - Team/namespace
- [x] `owner_kind` - Job/Deployment/etc
- [x] `preemption_policy` - Preemption behavior

#### Queue State Features (10 features)
- [x] `pending_ratio` - Queue saturation
- [x] `gpu_pending` - GPU jobs waiting
- [x] `cpu_pending` - CPU jobs waiting
- [x] `total_pending` - All pending jobs
- [x] `gpu_utilization` - Cluster GPU usage %
- [x] `cpu_utilization` - Cluster CPU usage %
- [x] `mem_utilization` - Cluster memory usage %
- [x] `avg_wait_seconds` - Running average wait
- [x] `max_wait_seconds` - Current max wait
- [x] `p90_wait_seconds` - 90th percentile wait

#### GPU Telemetry Features (15 features) - **NEW IN PAPER 2**
- [x] `gpu_temp_avg` - Average temperature (°C)
- [x] `gpu_temp_max` - Max temperature (°C)
- [x] `gpu_power_avg` - Average power draw (W)
- [x] `gpu_power_max` - Max power draw (W)
- [x] `gpu_util_avg` - Average SM utilization (%)
- [x] `gpu_mem_util_avg` - Average memory utilization (%)
- [x] `gpu_mem_used_avg` - Average memory used (MB)
- [x] `ecc_sbe_count` - Single-bit ECC errors
- [x] `ecc_dbe_count` - Double-bit ECC errors
- [x] `xid_error_count` - XID errors (last hour)
- [x] `retired_pages` - Retired memory pages
- [x] `throttling_events` - Thermal throttling count
- [x] `pcie_replay_count` - PCIe replay errors
- [x] `healthy_gpu_ratio` - % of healthy GPUs
- [x] `fragmentation_score` - GPU fragmentation (0-1)

#### Capacity Features (8 features)
- [x] `total_gpus` - Cluster total GPUs
- [x] `allocatable_gpus` - Schedulable GPUs
- [x] `gpu_nodes` - Number of GPU nodes
- [x] `cpu_nodes` - Number of CPU nodes
- [x] `effective_capacity` - After health/fragmentation adjustment
- [x] `usable_for_1_gpu` - Single-GPU job capacity
- [x] `usable_for_4_gpu` - 4-GPU job capacity
- [x] `largest_contiguous_block` - Max allocatable block

#### Outcome Labels
- [x] `wait_seconds` - Actual wait time
- [x] `label_long_wait` - Binary (>P90 threshold)
- [x] `scheduling_result` - Success/Fail/Timeout

---

## 🔧 Collection Infrastructure

### S3 Storage
```
s3://queue-risk-mvp-us-east-1-data-9ed20351/
└── paper2-data/
    ├── phase1-baseline/
    │   ├── snapshots/       # 30-second interval snapshots
    │   ├── jobs/            # Individual job records
    │   └── events/          # K8s events
    ├── phase2-load-testing/
    │   ├── low-load/
    │   ├── medium-load/
    │   └── high-load/
    └── phase3-production/
        ├── week1/
        └── week2/
```

### Collection Script
Location: `/Users/andrewespira/Downloads/st_peters/vgac/scripts/paper2_data_collector.py`

Features:
- 30-second snapshot interval
- Full DCGM metrics extraction
- Job lifecycle tracking (submit → schedule → complete)
- Automatic S3 upload
- Data validation checks

---

## 📅 Collection Phases

### Phase 1: Baseline Collection (Days 1-3)
**Goal**: Establish normal cluster behavior

| Hour | Workload Pattern | Expected Jobs |
|------|------------------|---------------|
| 0-8 | Low load (10% utilization) | 500 |
| 8-16 | Normal load (50% utilization) | 2,000 |
| 16-24 | Peak load (80% utilization) | 3,000 |

**Workload Mix**:
- 60% single-GPU training jobs (30-120 min)
- 20% multi-GPU training (2-4 GPUs, 1-4 hours)
- 10% inference jobs (1 GPU, 5-30 min)
- 10% CPU-only jobs (data preprocessing)

### Phase 2: Load Testing (Days 4-6)
**Goal**: Capture queue dynamics under stress

| Scenario | Pending Ratio | Duration | Purpose |
|----------|---------------|----------|---------|
| Low Load | 0-10% | 8 hours | Baseline behavior |
| Medium Load | 30-50% | 8 hours | Normal congestion |
| High Load | 70-90% | 8 hours | Stress conditions |
| Burst | 0→100%→0 | 4 hours | Sudden spikes |
| Fragmentation | Specific patterns | 4 hours | GPU blocking |

**Key Metrics to Capture**:
- Time to queue saturation
- Recovery time after burst
- Fragmentation impact on large jobs

### Phase 3: Production Simulation (Days 7-14)
**Goal**: Realistic mixed workloads

| Pattern | Description |
|---------|-------------|
| Daily cycle | Low overnight, peak midday |
| Weekly pattern | Heavy Mon-Thu, light Fri-Sun |
| Random bursts | Unpredictable spikes |
| Team conflicts | Namespace competition |
| Priority preemption | High-priority interrupts |

---

## ✅ Data Quality Checks

### Per-Snapshot Validation
```python
def validate_snapshot(snapshot: dict) -> bool:
    checks = [
        snapshot.get("timestamp") is not None,
        snapshot.get("total_pods", 0) >= 0,
        len(snapshot.get("dcgm_sample", [])) > 0,  # Must have GPU telemetry
        snapshot.get("queue_state", {}).get("pending_ratio") is not None,
    ]
    return all(checks)
```

### Per-Job Validation
```python
def validate_job(job: dict) -> bool:
    checks = [
        job.get("wait_seconds") is not None,
        job.get("req_gpu") is not None,
        job.get("creation_timestamp") is not None,
        job.get("start_timestamp") is not None or job.get("phase") == "Pending",
    ]
    return all(checks)
```

### Dataset-Level Checks
- [ ] No duplicate job IDs
- [ ] Timestamps are monotonically increasing
- [ ] All required features have <5% missing values
- [ ] Label distribution matches expected (10-15% long-wait)
- [ ] GPU telemetry coverage >90%

---

## 🚀 Start Collection

```bash
# Start the enhanced data collector
cd /Users/andrewespira/Downloads/st_peters/vgac
python scripts/paper2_data_collector.py \
    --phase baseline \
    --duration 72h \
    --interval 30s \
    --s3-prefix paper2-data/phase1-baseline
```

---

## 📈 Expected Outputs

After collection, we should have:

| Artifact | Location | Size |
|----------|----------|------|
| Raw snapshots | `s3://.../paper2-data/*/snapshots/` | ~500MB |
| Processed jobs | `paper2/data/processed/jobs.parquet` | ~100MB |
| Feature matrix | `paper2/data/processed/features.parquet` | ~50MB |
| Telemetry series | `paper2/data/processed/gpu_telemetry.parquet` | ~200MB |
| Sample dataset | `paper2/data/samples/sample_1000.csv` | ~5MB |







