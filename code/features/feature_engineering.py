#!/usr/bin/env python3
"""
Feature Engineering for Paper 2: Reliability-First Queue Risk
Computes derived features from raw Slurm telemetry data.

Key derivations:
- wait_seconds: Actual queue wait time per job
- label_long_wait: Binary label (1 if wait > P90 threshold)
- fragmentation_score: GPU allocation efficiency
- Various queue state aggregations
"""

import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ============================================================
# DATA PATHS
# ============================================================
TELEMETRY_ROOT = Path("/Volumes/data-esp/hpc_telemetry")
SLURM_DATA = TELEMETRY_ROOT / "2026-02-06"
QUEUE_STATE = SLURM_DATA / "queue_state"
GPU_METRICS = SLURM_DATA / "gpu"

OUTPUT_DIR = Path("/Users/andrewespira/Downloads/st_peters/Reliability-First-Queue-Risk/data")


def load_queue_state_data():
    """Load all queue state JSONL files into a DataFrame."""
    records = []
    
    for f in sorted(QUEUE_STATE.glob("*.jsonl")):
        with open(f) as fp:
            for line in fp:
                try:
                    rec = json.loads(line.strip())
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    df = pd.DataFrame(records)
    if 'epoch_ms' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['epoch_ms'], unit='ms')
    return df


def load_gpu_metrics():
    """Load all GPU metrics files into a DataFrame."""
    records = []
    
    for f in sorted(GPU_METRICS.glob("job_*.jsonl.gz")):
        try:
            with gzip.open(f, 'rt') as fp:
                for line in fp:
                    try:
                        rec = json.loads(line.strip())
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            continue
    
    df = pd.DataFrame(records)
    if 'epoch_ms' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['epoch_ms'], unit='ms')
    return df


def compute_wait_times(queue_df):
    """
    Compute actual wait times by matching before/after phases.
    
    Returns DataFrame with:
    - job_id
    - submit_time (before phase timestamp)
    - start_time (after phase timestamp)  
    - wait_seconds
    - wait_minutes
    """
    before = queue_df[queue_df['phase'] == 'before'].copy()
    after = queue_df[queue_df['phase'] == 'after'].copy()
    
    # Group by job_id, get first before and first after
    before_times = before.groupby('job_id').agg({
        'epoch_ms': 'min',
        'total_pending': 'first',
        'pending_gpus': 'first',
        'job_type': 'first',
    }).rename(columns={'epoch_ms': 'submit_epoch_ms'})
    
    after_times = after.groupby('job_id').agg({
        'epoch_ms': 'max',
        'total_running': 'last',
        'running_gpus': 'last',
    }).rename(columns={'epoch_ms': 'start_epoch_ms'})
    
    # Merge
    wait_df = before_times.join(after_times, how='inner')
    wait_df['wait_ms'] = wait_df['start_epoch_ms'] - wait_df['submit_epoch_ms']
    wait_df['wait_seconds'] = wait_df['wait_ms'] / 1000
    wait_df['wait_minutes'] = wait_df['wait_seconds'] / 60
    
    return wait_df.reset_index()


def compute_wait_labels(wait_df, percentile=90):
    """
    Create binary label for long wait based on percentile threshold.
    
    Following VGAC SLI/SLO framework:
    - SLI: P(long_wait) probability prediction
    - Label: 1 if wait > P90 threshold
    """
    threshold = wait_df['wait_seconds'].quantile(percentile / 100)
    wait_df['label_long_wait'] = (wait_df['wait_seconds'] > threshold).astype(int)
    wait_df['wait_threshold_p90'] = threshold
    
    print(f"\n📊 Wait Time Statistics:")
    print(f"   P50 (median): {wait_df['wait_seconds'].median():.1f}s")
    print(f"   P90 (threshold): {threshold:.1f}s")
    print(f"   P99: {wait_df['wait_seconds'].quantile(0.99):.1f}s")
    print(f"   Max: {wait_df['wait_seconds'].max():.1f}s")
    print(f"   Long wait jobs (>P90): {wait_df['label_long_wait'].sum()} / {len(wait_df)}")
    
    return wait_df


def compute_queue_state_features(queue_df):
    """
    Compute queue state features at submit time.
    
    Features for Paper 2:
    - pending_ratio: pending_gpus / total_gpus
    - queue_depth: total_pending
    - congestion_score: normalized queue pressure
    - fragmentation_score: 1 - (allocated / total) allocation efficiency
    """
    features = queue_df[queue_df['phase'] == 'before'].copy()
    
    # Pending ratio (GPU pressure)
    features['pending_ratio'] = features['pending_gpus'] / features['pending_gpus'].add(features['running_gpus']).replace(0, 1)
    
    # Queue depth normalized (0-1 based on observed max)
    max_pending = features['total_pending'].max()
    features['queue_depth_norm'] = features['total_pending'] / max(max_pending, 1)
    
    # Fragmentation score: how fragmented is the GPU allocation?
    # Lower allocated/total ratio when many GPUs available = more fragmentation
    features['allocation_ratio'] = features['gpu_nodes_alloc'] / features['gpu_nodes_total'].replace(0, 1)
    features['fragmentation_score'] = 1 - features['allocation_ratio']
    
    # Congestion score: combined metric
    features['congestion_score'] = (
        0.4 * features['pending_ratio'] +
        0.3 * features['queue_depth_norm'] +
        0.3 * (1 - features['allocation_ratio'])
    )
    
    return features


def compute_gpu_aggregates(gpu_df):
    """
    Aggregate GPU metrics per job.
    
    Features:
    - mean/std of utilization, temperature, power per job
    - max temperature seen during job
    - power efficiency ratio
    """
    if gpu_df.empty:
        return pd.DataFrame()
    
    aggs = gpu_df.groupby('job_id').agg({
        'util_gpu': ['mean', 'std', 'max'],
        'util_mem': ['mean', 'std', 'max'],
        'temp_c': ['mean', 'max'],
        'power_w': ['mean', 'max'],
        'mem_used_mb': ['mean', 'max'],
        'mem_total_mb': 'first',
    })
    
    # Flatten column names
    aggs.columns = ['_'.join(col).strip() for col in aggs.columns]
    
    # Power efficiency: utilization per watt
    aggs['power_efficiency'] = aggs['util_gpu_mean'] / aggs['power_w_mean'].replace(0, 1)
    
    # Memory pressure
    aggs['mem_pressure'] = aggs['mem_used_mb_mean'] / aggs['mem_total_mb_first'].replace(0, 1)
    
    return aggs.reset_index()


def build_training_dataset(wait_df, queue_features, gpu_aggs):
    """
    Build final training dataset merging all feature sources.
    
    Schema aligned with Paper 2 requirements.
    """
    # Start with wait times and labels
    df = wait_df.copy()
    
    # Merge queue state features at submit time
    queue_subset = queue_features[[
        'job_id', 'pending_ratio', 'queue_depth_norm', 
        'fragmentation_score', 'congestion_score',
        'total_pending', 'pending_gpus', 'running_gpus',
        'gpu_nodes_alloc', 'gpu_nodes_total'
    ]].drop_duplicates(subset=['job_id'], keep='first')
    
    df = df.merge(queue_subset, on='job_id', how='left')
    
    # Merge GPU aggregates
    if not gpu_aggs.empty:
        df = df.merge(gpu_aggs, on='job_id', how='left')
    
    # Job type encoding
    df['job_type_encoded'] = df['job_type'].map({
        'short': 0, 'medium': 1, 'long': 2, 'burst': 3, 'memheavy': 4
    }).fillna(-1).astype(int)
    
    return df


def main():
    print("=" * 60)
    print("FEATURE ENGINEERING FOR PAPER 2")
    print("=" * 60)
    
    # Load data
    print("\n📥 Loading queue state data...")
    queue_df = load_queue_state_data()
    print(f"   Loaded {len(queue_df)} queue state records")
    
    print("\n📥 Loading GPU metrics...")
    gpu_df = load_gpu_metrics()
    print(f"   Loaded {len(gpu_df)} GPU metric records")
    
    # Compute features
    print("\n⚙️  Computing wait times...")
    wait_df = compute_wait_times(queue_df)
    print(f"   Computed wait times for {len(wait_df)} jobs")
    
    print("\n⚙️  Computing wait labels (P90 threshold)...")
    wait_df = compute_wait_labels(wait_df)
    
    print("\n⚙️  Computing queue state features...")
    queue_features = compute_queue_state_features(queue_df)
    
    print("\n⚙️  Computing GPU aggregates...")
    gpu_aggs = compute_gpu_aggregates(gpu_df)
    
    # Build final dataset
    print("\n🔧 Building training dataset...")
    training_df = build_training_dataset(wait_df, queue_features, gpu_aggs)
    
    print(f"\n📊 Final Dataset:")
    print(f"   Rows: {len(training_df)}")
    print(f"   Columns: {len(training_df.columns)}")
    print(f"   Features: {list(training_df.columns)}")
    
    # Class balance
    if 'label_long_wait' in training_df.columns:
        pos_rate = training_df['label_long_wait'].mean()
        print(f"\n   Class balance: {pos_rate:.1%} positive (long wait)")
    
    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # CSV for easy inspection
    csv_path = OUTPUT_DIR / "training_dataset.csv"
    training_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved CSV: {csv_path}")
    
    # Parquet for efficient loading
    parquet_path = OUTPUT_DIR / "training_dataset.parquet"
    training_df.to_parquet(parquet_path, index=False)
    print(f"💾 Saved Parquet: {parquet_path}")
    
    # Summary statistics JSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'n_jobs': len(training_df),
        'n_features': len(training_df.columns),
        'features': list(training_df.columns),
        'wait_stats': {
            'p50_seconds': float(training_df['wait_seconds'].median()),
            'p90_seconds': float(training_df['wait_seconds'].quantile(0.9)),
            'p99_seconds': float(training_df['wait_seconds'].quantile(0.99)),
            'max_seconds': float(training_df['wait_seconds'].max()),
        },
        'class_balance': {
            'n_long_wait': int(training_df['label_long_wait'].sum()),
            'n_short_wait': int(len(training_df) - training_df['label_long_wait'].sum()),
            'positive_rate': float(training_df['label_long_wait'].mean()),
        },
        'job_types': training_df['job_type'].value_counts().to_dict(),
    }
    
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved Summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ Feature engineering complete!")
    print("=" * 60)
    
    return training_df


if __name__ == "__main__":
    df = main()
