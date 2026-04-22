#!/usr/bin/env python3
"""
Model Training and Evaluation for Paper 2
Trains calibration-focused models on Slurm HPC telemetry data.

Metrics aligned with VGAC/Paper 2 requirements:
- AUC-ROC, AUC-PR for discrimination
- ECE, MCE, Brier for calibration
- Tail calibration analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("/Users/andrewespira/Downloads/st_peters/Reliability-First-Queue-Risk/data")
OUTPUT_DIR = Path("/Users/andrewespira/Downloads/st_peters/Reliability-First-Queue-Risk/results")
FIGURES_DIR = Path("/Users/andrewespira/Downloads/st_peters/Reliability-First-Queue-Risk/figures")


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include right boundary for last bin
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_size = mask.sum() / len(y_true)
            ece += bin_size * abs(bin_accuracy - bin_confidence)
    
    return ece


def compute_mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            gap = abs(bin_accuracy - bin_confidence)
            mce = max(mce, gap)
    
    return mce


def brier_decomposition(y_true, y_prob):
    """
    Brier score decomposition: Reliability - Resolution + Uncertainty
    Following Murphy (1973) decomposition.
    """
    n = len(y_true)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    # Base rate (uncertainty)
    p_bar = y_true.mean()
    uncertainty = p_bar * (1 - p_bar)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        
        if mask.sum() > 0:
            n_k = mask.sum()
            o_k = y_true[mask].mean()  # Observed frequency
            f_k = y_prob[mask].mean()  # Forecast probability
            
            reliability += (n_k / n) * (o_k - f_k) ** 2
            resolution += (n_k / n) * (o_k - p_bar) ** 2
    
    brier = reliability - resolution + uncertainty
    
    return {
        'brier': brier,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty
    }


def tail_calibration_analysis(y_true, y_prob, thresholds=[0.5, 0.6, 0.7, 0.8]):
    """Analyze calibration in high-risk tail."""
    results = []
    
    for thresh in thresholds:
        mask = y_prob >= thresh
        if mask.sum() > 0:
            actual = y_true[mask].mean()
            predicted = y_prob[mask].mean()
            gap = abs(actual - predicted)
            results.append({
                'threshold': thresh,
                'count': int(mask.sum()),
                'actual': float(actual),
                'predicted': float(predicted),
                'gap': float(gap)
            })
    
    return results


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    """Train model and compute comprehensive metrics."""
    # Fit model
    model.fit(X_train, y_train)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        'model': model_name,
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'ece': compute_ece(y_test.values, y_prob),
        'mce': compute_mce(y_test.values, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Brier decomposition
    brier_dec = brier_decomposition(y_test.values, y_prob)
    metrics.update({
        'brier_reliability': brier_dec['reliability'],
        'brier_resolution': brier_dec['resolution'],
        'brier_uncertainty': brier_dec['uncertainty'],
    })
    
    # Tail calibration
    metrics['tail_calibration'] = tail_calibration_analysis(y_test.values, y_prob)
    
    return metrics, y_prob, model


def main():
    print("=" * 70)
    print("MODEL TRAINING FOR PAPER 2: SLURM HPC DATA")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading training dataset...")
    df = pd.read_parquet(DATA_DIR / "training_dataset.parquet")
    print(f"   Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Feature selection
    feature_cols = [
        # Queue state features
        'pending_ratio', 'queue_depth_norm', 'fragmentation_score', 'congestion_score',
        'total_pending_x', 'pending_gpus_x', 'running_gpus_x',
        
        # GPU telemetry aggregates
        'util_gpu_mean', 'util_gpu_std', 'util_gpu_max',
        'util_mem_mean', 'temp_c_mean', 'temp_c_max',
        'power_w_mean', 'mem_pressure', 'power_efficiency',
        
        # Job type
        'job_type_encoded'
    ]
    
    # Filter to available columns
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"\n   Using {len(available_features)} features: {available_features}")
    
    # Prepare data
    X = df[available_features].fillna(0)
    y = df['label_long_wait']
    
    print(f"\n   Class distribution: {y.value_counts().to_dict()}")
    print(f"   Positive rate: {y.mean():.1%}")
    
    # Train/test split (temporal would be better but we don't have enough data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }
    
    results = []
    best_model = None
    best_auc = 0
    best_probs = None
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Use scaled data for LR, original for tree-based
        if 'Logistic' in name:
            metrics, y_prob, trained_model = train_and_evaluate(
                X_train_scaled, X_test_scaled, y_train, y_test, name, model
            )
        else:
            metrics, y_prob, trained_model = train_and_evaluate(
                X_train, X_test, y_train, y_test, name, model
            )
        
        results.append(metrics)
        
        print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"   AUC-PR:  {metrics['auc_pr']:.3f}")
        print(f"   ECE:     {metrics['ece']:.3f}")
        print(f"   MCE:     {metrics['mce']:.3f}")
        print(f"   Brier:   {metrics['brier']:.3f}")
        print(f"   F1:      {metrics['f1']:.3f}")
        
        if metrics['auc_roc'] > best_auc:
            best_auc = metrics['auc_roc']
            best_model = trained_model
            best_probs = y_prob
    
    # Calibrated model
    print("\n🔧 Training Calibrated Logistic Regression (Isotonic)...")
    cal_lr = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=42),
        method='isotonic', cv=3
    )
    cal_metrics, cal_probs, _ = train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        'Calibrated LR (Isotonic)', cal_lr
    )
    results.append(cal_metrics)
    
    print(f"   AUC-ROC: {cal_metrics['auc_roc']:.3f}")
    print(f"   ECE:     {cal_metrics['ece']:.3f}")
    print(f"   Brier:   {cal_metrics['brier']:.3f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary_df = pd.DataFrame([
        {
            'Model': r['model'],
            'AUC-ROC': f"{r['auc_roc']:.3f}",
            'AUC-PR': f"{r['auc_pr']:.3f}",
            'ECE': f"{r['ece']:.3f}",
            'MCE': f"{r['mce']:.3f}",
            'Brier': f"{r['brier']:.3f}",
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))
    
    # Best model details
    best_result = max(results, key=lambda x: x['auc_roc'])
    print(f"\n🏆 Best Model: {best_result['model']}")
    print(f"\nBrier Decomposition:")
    print(f"   Reliability:  {best_result['brier_reliability']:.4f}")
    print(f"   Resolution:   {best_result['brier_resolution']:.4f}")
    print(f"   Uncertainty:  {best_result['brier_uncertainty']:.4f}")
    
    print(f"\nTail Calibration:")
    for tc in best_result['tail_calibration']:
        print(f"   ≥{tc['threshold']:.1f}: {tc['count']:3d} samples, "
              f"actual={tc['actual']:.2f}, pred={tc['predicted']:.2f}, gap={tc['gap']:.3f}")
    
    # Wait time statistics for paper
    print("\n" + "=" * 70)
    print("STATISTICS FOR PAPER")
    print("=" * 70)
    
    print(f"\nDataset Characteristics:")
    print(f"   Total jobs: {len(df)}")
    print(f"   Wait time P50: {df['wait_seconds'].median():.0f}s")
    print(f"   Wait time P90: {df['wait_seconds'].quantile(0.9):.0f}s (threshold)")
    print(f"   Wait time P99: {df['wait_seconds'].quantile(0.99):.0f}s")
    print(f"   Long-wait jobs: {df['label_long_wait'].sum()} ({df['label_long_wait'].mean():.1%})")
    
    print(f"\nQueue State at Submit (mean ± std):")
    print(f"   Pending jobs: {df['total_pending_x'].mean():.1f} ± {df['total_pending_x'].std():.1f}")
    print(f"   Pending GPUs: {df['pending_gpus_x'].mean():.1f} ± {df['pending_gpus_x'].std():.1f}")
    print(f"   Running GPUs: {df['running_gpus_x'].mean():.1f} ± {df['running_gpus_x'].std():.1f}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-serializable format
    results_json = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'tail_calibration'}
        r_copy['tail_calibration'] = r['tail_calibration']
        results_json.append(r_copy)
    
    output_path = OUTPUT_DIR / "model_evaluation_slurm.json"
    with open(output_path, 'w') as f:
        json.dump({
            'evaluation_time': datetime.now().isoformat(),
            'dataset': 'Slurm HPC (ParallelCluster)',
            'n_samples': len(df),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'features': available_features,
            'results': results_json,
        }, f, indent=2)
    print(f"\n💾 Saved results: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ Model training complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
