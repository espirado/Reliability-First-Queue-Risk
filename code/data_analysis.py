#!/usr/bin/env python3
"""
Combined EKS + Slurm Telemetry Analysis for Paper 2 & Paper 4

This script analyzes the collected telemetry data and maps it to:
- Paper 2: GPU Cluster Queue Prediction with Runtime Telemetry
- Paper 4: Calibration-Aware GPU Scheduling Layer (ASPLOS Workshop)

Data Sources:
- EKS Telemetry: /Volumes/data-esp/hpc_telemetry/eks_telemetry/
- Slurm Queue State: /Volumes/data-esp/hpc_telemetry/2026-02-06/queue_state/
- Slurm GPU Metrics: /Volumes/data-esp/hpc_telemetry/2026-02-06/gpu/
"""

import json
import gzip
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime

# Data paths
EKS_PATH = Path("/Volumes/data-esp/hpc_telemetry/eks_telemetry/2026-02-05")
SLURM_QUEUE_PATH = Path("/Volumes/data-esp/hpc_telemetry/2026-02-06/queue_state")
SLURM_GPU_PATH = Path("/Volumes/data-esp/hpc_telemetry/2026-02-06/gpu")

def load_eks_telemetry():
    """Load EKS telemetry snapshots."""
    records = []
    for f in sorted(EKS_PATH.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
                records.append({
                    "timestamp": data.get("timestamp"),
                    "pending_jobs": data.get("summary", {}).get("pending_jobs", 0),
                    "running_jobs": data.get("summary", {}).get("running_jobs", 0),
                    "total_jobs": data.get("summary", {}).get("total_jobs", 0),
                    "pending_ratio": data.get("summary", {}).get("pending_ratio", 0),
                    "total_cpus": data.get("summary", {}).get("total_cpus", 0),
                    "alloc_cpus": data.get("summary", {}).get("alloc_cpus", 0),
                    "cpu_utilization": data.get("summary", {}).get("cpu_utilization", 0),
                    "total_nodes": data.get("summary", {}).get("total_nodes", 0),
                    "idle_nodes": data.get("summary", {}).get("idle_nodes", 0),
                    "queue_state_count": len(data.get("queue_state", [])),
                    "node_state_count": len(data.get("node_state", [])),
                    "source": "eks"
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.DataFrame(records)

def load_slurm_queue_state():
    """Load Slurm queue state snapshots."""
    records = []
    for f in sorted(SLURM_QUEUE_PATH.glob("*.jsonl")):
        try:
            with open(f) as fp:
                for line in fp:
                    data = json.loads(line)
                    records.append({
                        "timestamp": data.get("timestamp"),
                        "epoch_ms": data.get("epoch_ms"),
                        "job_id": data.get("job_id"),
                        "job_type": data.get("job_type"),
                        "phase": data.get("phase"),
                        "total_pending": data.get("total_pending", 0),
                        "total_running": data.get("total_running", 0),
                        "pending_gpus": data.get("pending_gpus", 0),
                        "running_gpus": data.get("running_gpus", 0),
                        "gpu_nodes_alloc": data.get("gpu_nodes_alloc", 0),
                        "gpu_nodes_total": data.get("gpu_nodes_total", 0),
                        "cluster": data.get("cluster"),
                        "source": "slurm"
                    })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.DataFrame(records)

def load_slurm_gpu_metrics():
    """Load Slurm GPU metrics from compressed files."""
    records = []
    for f in sorted(SLURM_GPU_PATH.glob("*.jsonl.gz")):
        try:
            with gzip.open(f, 'rt') as fp:
                for line in fp:
                    data = json.loads(line)
                    records.append({
                        "timestamp": data.get("ts"),
                        "epoch_ms": data.get("epoch_ms"),
                        "job_id": data.get("job_id"),
                        "job_type": data.get("job_type"),
                        "node": data.get("node"),
                        "gpu_id": data.get("gpu_id"),
                        "model": data.get("model"),
                        "util_gpu": data.get("util_gpu", 0),
                        "util_mem": data.get("util_mem", 0),
                        "mem_total_mb": data.get("mem_total_mb", 0),
                        "mem_used_mb": data.get("mem_used_mb", 0),
                        "mem_free_mb": data.get("mem_free_mb", 0),
                        "temp_c": data.get("temp_c", 0),
                        "power_w": data.get("power_w", 0),
                        "power_lim_w": data.get("power_lim_w", 0),
                        "source": "slurm_gpu"
                    })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.DataFrame(records)

def analyze_data():
    """Main analysis function."""
    print("=" * 60)
    print("TELEMETRY DATA ANALYSIS FOR PAPER 2 & PAPER 4")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading data...")
    eks_df = load_eks_telemetry()
    slurm_queue_df = load_slurm_queue_state()
    slurm_gpu_df = load_slurm_gpu_metrics()
    
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 EKS Telemetry:")
    print(f"   Snapshots: {len(eks_df)}")
    if len(eks_df) > 0:
        print(f"   Time range: {eks_df['timestamp'].min()} to {eks_df['timestamp'].max()}")
        print(f"   Pending range: {eks_df['pending_jobs'].min()} - {eks_df['pending_jobs'].max()}")
    
    print(f"\n📊 Slurm Queue State:")
    print(f"   Records: {len(slurm_queue_df)}")
    if len(slurm_queue_df) > 0:
        print(f"   Unique jobs: {slurm_queue_df['job_id'].nunique()}")
        print(f"   Job types: {slurm_queue_df['job_type'].value_counts().to_dict()}")
        print(f"   Pending range: {slurm_queue_df['total_pending'].min()} - {slurm_queue_df['total_pending'].max()}")
    
    print(f"\n📊 Slurm GPU Metrics:")
    print(f"   Records: {len(slurm_gpu_df)}")
    if len(slurm_gpu_df) > 0:
        print(f"   Unique jobs: {slurm_gpu_df['job_id'].nunique()}")
        print(f"   GPU model: {slurm_gpu_df['model'].iloc[0] if len(slurm_gpu_df) > 0 else 'N/A'}")
        print(f"   Temp range: {slurm_gpu_df['temp_c'].min()}°C - {slurm_gpu_df['temp_c'].max()}°C")
        print(f"   Power range: {slurm_gpu_df['power_w'].min():.1f}W - {slurm_gpu_df['power_w'].max():.1f}W")
    
    print(f"\n{'='*60}")
    print("PAPER 2 FEATURE COVERAGE")
    print("=" * 60)
    
    # Paper 2 required features check
    paper2_features = {
        "Queue State Features": {
            "total_pending": len(slurm_queue_df) > 0,
            "total_running": len(slurm_queue_df) > 0,
            "pending_gpus": len(slurm_queue_df) > 0,
            "running_gpus": len(slurm_queue_df) > 0,
            "gpu_nodes_alloc": len(slurm_queue_df) > 0,
            "gpu_nodes_total": len(slurm_queue_df) > 0,
            "pending_ratio": len(eks_df) > 0,
        },
        "GPU Telemetry Features": {
            "util_gpu": len(slurm_gpu_df) > 0,
            "util_mem": len(slurm_gpu_df) > 0,
            "temp_c": len(slurm_gpu_df) > 0,
            "power_w": len(slurm_gpu_df) > 0,
            "mem_used_mb": len(slurm_gpu_df) > 0,
            "mem_total_mb": len(slurm_gpu_df) > 0,
        },
        "Job Features": {
            "job_id": len(slurm_queue_df) > 0,
            "job_type": len(slurm_queue_df) > 0 and slurm_queue_df['job_type'].notna().sum() > 0,
            "phase (before/after)": len(slurm_queue_df) > 0,
        }
    }
    
    for category, features in paper2_features.items():
        print(f"\n{category}:")
        for feat, available in features.items():
            status = "✅" if available else "❌"
            print(f"   {status} {feat}")
    
    # Gaps analysis
    print(f"\n{'='*60}")
    print("GAPS FOR PAPER 2")
    print("=" * 60)
    
    gaps = [
        ("wait_seconds", "Need to compute from before/after timestamps"),
        ("label_long_wait", "Need to derive from wait_seconds (>P90)"),
        ("priority_class", "Not in Slurm telemetry (Slurm uses numeric priority)"),
        ("namespace", "Not applicable in Slurm (use partition instead)"),
        ("ecc_errors", "Not captured (nvidia-smi query needed)"),
        ("xid_events", "Not captured (dmesg parsing needed)"),
        ("fragmentation_score", "Need to compute from gpu_nodes_alloc/total"),
    ]
    
    print("\nMissing/Derived Features:")
    for feat, note in gaps:
        print(f"   ⚠️  {feat}: {note}")
    
    print(f"\n{'='*60}")
    print("PAPER 4 (ASPLOS) RELEVANCE")
    print("=" * 60)
    
    print("""
Paper 4 focuses on calibration-aware scheduling. Key data needs:

1. ✅ Queue State at Submit Time (have: total_pending, running_gpus)
   - Can compute congestion metrics for calibration experiments

2. ✅ Job Outcomes (have: before/after phases per job)  
   - Can compute actual wait times
   - Can create binary labels for calibration

3. ✅ GPU Utilization (have: util_gpu, util_mem, power_w, temp_c)
   - Supports capacity-aware prediction claims

4. ⚠️  Need to compute:
   - ECE (Expected Calibration Error) from predictions
   - Brier score decomposition
   - Calibration curves with confidence intervals

5. ✅ SLI/SLO Context (from VGAC):
   - SLI: P(long_wait) probability prediction
   - SLO: ECE < 0.05 (calibration target)
   - SLO: Latency < 10ms for predictions
""")
    
    # Save summary to file
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "eks_snapshots": len(eks_df),
        "slurm_queue_records": len(slurm_queue_df),
        "slurm_gpu_records": len(slurm_gpu_df),
        "unique_slurm_jobs": int(slurm_queue_df['job_id'].nunique()) if len(slurm_queue_df) > 0 else 0,
        "job_types": slurm_queue_df['job_type'].value_counts().to_dict() if len(slurm_queue_df) > 0 else {},
        "pending_range": {
            "min": int(slurm_queue_df['total_pending'].min()) if len(slurm_queue_df) > 0 else 0,
            "max": int(slurm_queue_df['total_pending'].max()) if len(slurm_queue_df) > 0 else 0
        }
    }
    
    return eks_df, slurm_queue_df, slurm_gpu_df, summary

if __name__ == "__main__":
    eks_df, slurm_queue_df, slurm_gpu_df, summary = analyze_data()
    print(f"\n{'='*60}")
    print("SUMMARY JSON")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
