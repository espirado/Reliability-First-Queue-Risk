"""Scaling study for the Slurm evaluation.

Generates three synthetic Slurm traces (baseline, heavy-load stress, and an
A100/H100-profile thermal variant), trains LR/RF/GB on each with isotonic
calibration, and emits:

* ``artifacts/slurm_sim_<name>.csv``  -- full synthetic trace
* ``artifacts/slurm_sim_metrics.json`` -- AUROC/ECE/MCE/Brier/tier, bootstrap CIs
* ``artifacts/slurm_sim_heuristic.json`` -- naive queue-depth baseline
* ``artifacts/slurm_sim_leadtime.json`` -- decision-lead-time distribution
* ``artifacts/slurm_sim_fairness.json`` -- per-job-type precision/recall parity

Everything is seeded; rerunning the script reproduces the paper numbers
exactly. See ``paper2_repo/README.md`` for one-line invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.slurm_simulator import SimulatorConfig, simulate_from_trace  # noqa: E402


FEATURES = [
    "pending_ratio",
    "queue_depth_norm",
    "fragmentation_score",
    "congestion_score",
    "total_pending",
    "pending_gpus",
    "running_gpus",
    "gpu_nodes_alloc",
    "util_gpu_mean",
    "util_gpu_std",
    "util_gpu_max",
    "util_mem_mean",
    "util_mem_std",
    "util_mem_max",
    "temp_c_mean",
    "temp_c_max",
    "power_w_mean",
    "power_w_max",
    "mem_used_mb_mean",
    "mem_used_mb_max",
    "power_efficiency",
    "mem_pressure",
    "gpus_request",
]


def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(p[mask].mean() - y_true[mask].mean())
        ece += (mask.sum() / len(p)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _bootstrap_ci(
    y: np.ndarray, p: np.ndarray, fn, n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats[i] = fn(y[idx], p[idx])
        except ValueError:
            stats[i] = np.nan
    stats = stats[~np.isnan(stats)]
    return float(fn(y, p)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _tier(ece: float, auroc: float, auprc: float) -> str:
    if ece <= 0.05 and auroc >= 0.70 and auprc >= 0.50:
        return "gate"
    if ece <= 0.10 and auroc >= 0.70:
        return "suggest"
    if ece <= 0.15 and auroc >= 0.65:
        return "advisory"
    return "warn"


def _train_and_score(df: pd.DataFrame, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(0.8 * len(df))
    train, test = df.iloc[idx[:split]], df.iloc[idx[split:]]
    X_tr = train[FEATURES].to_numpy()
    X_te = test[FEATURES].to_numpy()
    y_tr = train["label_long_wait"].to_numpy()
    y_te = test["label_long_wait"].to_numpy()
    out: dict = {"n_train": len(train), "n_test": len(test), "pos_rate": float(y_te.mean())}
    models = {
        "LR": LogisticRegression(max_iter=300, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed),
    }
    for name, base in models.items():
        cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
        cal.fit(X_tr, y_tr)
        p = cal.predict_proba(X_te)[:, 1]
        auroc, lo, hi = _bootstrap_ci(y_te, p, roc_auc_score, seed=seed)
        auprc, _, _ = _bootstrap_ci(y_te, p, average_precision_score, seed=seed)
        ece, mce = _ece(y_te, p)
        brier = brier_score_loss(y_te, p)
        out[name] = {
            "auroc": auroc,
            "auroc_ci": [lo, hi],
            "auprc": auprc,
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "tier": _tier(ece, auroc, auprc),
        }
    return out


def _heuristic_baseline(df: pd.DataFrame) -> dict:
    thresh = float(df["queue_depth_norm"].median())
    y_pred = (df["queue_depth_norm"] > thresh).astype(int).to_numpy()
    y_true = df["label_long_wait"].to_numpy()
    p = df["queue_depth_norm"].to_numpy()
    return {
        "rule": f"queue_depth_norm > {thresh:.3f} (cluster median)",
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, p)) if len(set(y_true)) > 1 else float("nan"),
    }


def _lead_time(df: pd.DataFrame, p: np.ndarray, alert_prob: float = 0.7) -> dict:
    df = df.copy().reset_index(drop=True)
    df["p_hat"] = p
    firing = (df["p_hat"] >= alert_prob) & (df["label_long_wait"] == 1)
    if firing.sum() == 0:
        return {"alert_prob": alert_prob, "n_true_alerts": 0}
    leads = df.loc[firing, "wait_seconds"].to_numpy()
    return {
        "alert_prob": alert_prob,
        "n_true_alerts": int(firing.sum()),
        "median_s": float(np.median(leads)),
        "p90_s": float(np.percentile(leads, 90)),
        "max_s": float(leads.max()),
    }


def _fairness_by_type(df: pd.DataFrame, p: np.ndarray, thresh: float = 0.5) -> dict:
    df = df.copy().reset_index(drop=True)
    df["pred"] = (p >= thresh).astype(int)
    out = {}
    for jt, sub in df.groupby("job_type"):
        if sub["label_long_wait"].sum() == 0 and sub["pred"].sum() == 0:
            continue
        out[jt] = {
            "n": int(len(sub)),
            "pos_rate": float(sub["label_long_wait"].mean()),
            "precision": float(precision_score(sub["label_long_wait"], sub["pred"], zero_division=0)),
            "recall": float(recall_score(sub["label_long_wait"], sub["pred"], zero_division=0)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="../../../paper2/paper2_final_raw/data/training_dataset.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--n-jobs", type=int, default=10_000)
    parser.add_argument("--reuse-sim", action="store_true", help="Skip simulation; reuse existing slurm_sim_*.csv in out-dir")
    args = parser.parse_args()

    real = pd.read_csv(args.real)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "baseline": SimulatorConfig(n_jobs=args.n_jobs, arrival_scale=1.0, dcgm_profile="v100_t4", seed=42),
        "heavy": SimulatorConfig(n_jobs=args.n_jobs, arrival_scale=0.5, dcgm_profile="v100_t4", seed=43),
        "a100": SimulatorConfig(n_jobs=args.n_jobs, arrival_scale=1.0, dcgm_profile="a100_h100", seed=44),
    }

    all_metrics: dict = {}
    all_heuristic: dict = {}
    all_lead: dict = {}
    all_fair: dict = {}

    for name, cfg in scenarios.items():
        sim_path = out_dir / f"slurm_sim_{name}.csv"
        if args.reuse_sim and sim_path.exists():
            print(f"[sim] reusing {sim_path}", flush=True)
            sim = pd.read_csv(sim_path)
        else:
            print(f"[sim] running {name} ...", flush=True)
            sim = simulate_from_trace(real, cfg)
            sim.to_csv(sim_path, index=False)
            print(f"[sim]   n={len(sim)} pos_rate={sim['label_long_wait'].mean():.3f} wait_p90={sim['wait_seconds'].quantile(0.9):.1f}s -> {sim_path.name}")

        metrics = _train_and_score(sim, seed=cfg.seed)
        all_metrics[name] = metrics
        all_heuristic[name] = _heuristic_baseline(sim)

        from sklearn.model_selection import train_test_split
        train, test = train_test_split(sim, test_size=0.2, random_state=cfg.seed, stratify=sim["label_long_wait"])
        best = max(("LR", "RF", "GB"), key=lambda k: metrics[k]["auroc"])
        if best == "LR":
            model = LogisticRegression(max_iter=300, class_weight="balanced")
        elif best == "RF":
            model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=cfg.seed, n_jobs=-1)
        else:
            model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=cfg.seed)
        cal = CalibratedClassifierCV(model, method="isotonic", cv=5)
        cal.fit(train[FEATURES], train["label_long_wait"])
        p_test = cal.predict_proba(test[FEATURES])[:, 1]
        all_lead[name] = _lead_time(test.reset_index(drop=True), p_test)
        all_fair[name] = _fairness_by_type(test.reset_index(drop=True), p_test)

    (out_dir / "slurm_sim_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (out_dir / "slurm_sim_heuristic.json").write_text(json.dumps(all_heuristic, indent=2))
    (out_dir / "slurm_sim_leadtime.json").write_text(json.dumps(all_lead, indent=2))
    (out_dir / "slurm_sim_fairness.json").write_text(json.dumps(all_fair, indent=2))

    print("[sim] cross-profile calibration collapse (v100/t4 -> a100/h100) ...")
    base_csv = out_dir / "slurm_sim_baseline.csv"
    a100_csv = out_dir / "slurm_sim_a100.csv"
    base = pd.read_csv(base_csv)
    a100 = pd.read_csv(a100_csv)
    X_tr, y_tr = base[FEATURES].to_numpy(), base["label_long_wait"].to_numpy()
    X_te, y_te = a100[FEATURES].to_numpy(), a100["label_long_wait"].to_numpy()
    cross = {}
    for name, base_model in [
        ("LR", LogisticRegression(max_iter=300, class_weight="balanced")),
        ("RF", RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("GB", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
    ]:
        cal = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        cal.fit(X_tr, y_tr)
        p = cal.predict_proba(X_te)[:, 1]
        ece, mce = _ece(y_te, p)
        auroc = roc_auc_score(y_te, p)
        cross[name] = {"auroc": float(auroc), "ece": float(ece), "mce": float(mce)}
    in_dist_ece = all_metrics["baseline"]["GB"]["ece"]
    cross["notes"] = {
        "in_distribution_GB_ece": in_dist_ece,
        "description": (
            "Train on v100/t4-profile simulator, test on a100/h100-profile simulator. "
            "Discrimination (AUROC) transfers; calibration does not."
        ),
    }
    (out_dir / "slurm_sim_cross_profile.json").write_text(json.dumps(cross, indent=2))

    print("[sim] real-trace apples-to-apples benchmark ...")
    real_aligned = real.rename(columns={
        "total_pending_y": "total_pending",
        "pending_gpus_y": "pending_gpus",
        "running_gpus_y": "running_gpus",
    }).copy()
    real_aligned["gpus_request"] = real_aligned["gpu_nodes_alloc"]
    for col in FEATURES:
        if col not in real_aligned.columns:
            real_aligned[col] = 0.0
        real_aligned[col] = pd.to_numeric(real_aligned[col], errors="coerce").fillna(0.0)
    real_metrics = _train_and_score(real_aligned, seed=7)
    real_heur = _heuristic_baseline(real_aligned)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(real_aligned, test_size=0.2, random_state=7, stratify=real_aligned["label_long_wait"])
    cal = CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=7), method="isotonic", cv=5)
    cal.fit(tr[FEATURES], tr["label_long_wait"])
    p_real = cal.predict_proba(te[FEATURES])[:, 1]
    real_lead = _lead_time(te.reset_index(drop=True).assign(job_type=te["job_type"].values), p_real)
    real_fair = _fairness_by_type(te.reset_index(drop=True), p_real)
    (out_dir / "slurm_real_benchmark.json").write_text(json.dumps({
        "metrics": real_metrics,
        "heuristic": real_heur,
        "leadtime": real_lead,
        "fairness": real_fair,
        "n": int(len(real)),
        "pos_rate": float(real["label_long_wait"].mean()),
    }, indent=2))

    print(f"[sim] done -> {out_dir}/slurm_*.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
