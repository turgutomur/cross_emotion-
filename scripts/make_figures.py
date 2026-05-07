#!/usr/bin/env python3
"""
Generate paper figures for the cross-emotion project.

Reads result CSVs from <output-dir>/results/, the bootstrap p-value table,
and (for Figure 4) model checkpoints to run inference on WASSA-21.

Usage (on Colab after all training seeds are complete):
    python scripts/make_figures.py \\
        --output-dir /content/drive/MyDrive/cross_emotion_data/outputs \\
        --figures-dir /content/drive/MyDrive/cross_emotion_data/outputs/figures \\
        --bootstrap-csv /content/drive/MyDrive/cross_emotion_data/outputs/bootstrap/pairwise_pvalues.csv \\
        --config configs/default.yaml

All four figures are saved as both .pdf (paper) and .png (preview).
Figures whose data are missing are skipped with a warning; the rest proceed.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on Colab and headless servers
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Constants ────────────────────────────────────────────────────────────────

EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Fixed display order per protocol.  Slot 0 is always the CE baseline.
LODO_METHOD_ORDER = [
    "source_only", "dann", "cdan",
    "source_only_focal", "dann_focal", "cdan_focal",
]
MIXED_METHOD_ORDER = [
    "mixed", "dann", "cdan",
    "mixed_focal", "dann_focal", "cdan_focal",
]

METHOD_DISPLAY: Dict[str, str] = {
    "source_only":       "CE",
    "mixed":             "CE",
    "dann":              "DANN",
    "cdan":              "CDAN",
    "source_only_focal": "CE+Focal",
    "mixed_focal":       "CE+Focal",
    "dann_focal":        "DANN+Focal",
    "cdan_focal":        "CDAN+Focal",
}

# 4-panel order for Figure 1
PANELS: List[Tuple[str, str]] = [
    ("mixed", "mixed"),
    ("lodo",  "goemotions"),
    ("lodo",  "isear"),
    ("lodo",  "wassa21"),
]

PANEL_TITLES: Dict[Tuple[str, str], str] = {
    ("mixed", "mixed"):      "Mixed → Mixed",
    ("lodo",  "goemotions"): "LODO → GoEmotions",
    ("lodo",  "isear"):      "LODO → ISEAR",
    ("lodo",  "wassa21"):    "LODO → WASSA-21",
}

# Visual style
COLOR_NOFOCAL = "#4C72B0"
COLOR_FOCAL   = "#DD8452"

# Regex to parse per-epoch log lines emitted by Trainer._train_epoch.
# Log format (trainer.py):
#   epoch=01  train_loss=0.1234  val_macro_f1=0.5678  [...]
#       task_loss=0.1234  domain_loss=0.5678  lambda=0.1234
_EPOCH_RE = re.compile(
    r"epoch=(\d+).*?domain_loss=([\d.eE+\-]+).*?lambda=([\d.eE+\-]+)"
)


# ─── Shared helpers ───────────────────────────────────────────────────────────

CellData = Dict[str, List]  # keys: seeds, val_f1s, test_f1s (parallel lists)


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved → {out}")
    plt.close(fig)


def _parse_csv_stem(stem: str) -> Optional[Tuple[str, str, str]]:
    """Parse (method, protocol, target) from a results CSV filename stem."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    target   = parts[-1]
    protocol = parts[-2]
    if protocol not in ("lodo", "mixed"):
        return None
    method = "_".join(parts[:-2])
    return method, protocol, target


def discover_results(output_dir: Path) -> Dict[Tuple[str, str, str], CellData]:
    """Recursively scan output_dir for */results/*.csv; return per-cell F1 lists.

    Mirrors the discovery logic in run_bootstrap.py so both scripts agree on
    which files belong to which (method, protocol, target) cell.  Duplicate
    CSV files for the same cell are merged by taking the union of seeds.
    """
    cells: Dict[Tuple[str, str, str], CellData] = {}

    if not output_dir.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return cells

    candidates = [
        p for p in sorted(output_dir.rglob("*.csv"))
        if p.parent.name == "results"
    ]

    for csv_path in candidates:
        parsed = _parse_csv_stem(csv_path.stem)
        if parsed is None:
            logger.warning(f"Could not parse cell from: {csv_path.name} — skipping")
            continue
        method, protocol, target = parsed
        key = (method, protocol, target)

        seeds, val_f1s, test_f1s = [], [], []
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if "seed" not in row:
                        continue
                    seeds.append(int(row["seed"]))
                    val_f1s.append(float(row.get("val_macro_f1_aggregate", "nan")))
                    test_f1s.append(float(row.get("test_macro_f1_aggregate", "nan")))
        except Exception as exc:
            logger.warning(f"Could not read {csv_path}: {exc}")
            continue

        if key in cells:
            existing = set(cells[key]["seeds"])
            for s, v, t in zip(seeds, val_f1s, test_f1s):
                if s not in existing:
                    cells[key]["seeds"].append(s)
                    cells[key]["val_f1s"].append(v)
                    cells[key]["test_f1s"].append(t)
        else:
            cells[key] = {"seeds": seeds, "val_f1s": val_f1s, "test_f1s": test_f1s}

    logger.info(f"Discovered {len(cells)} result cells under {output_dir}")
    return cells


def load_pvalues(bootstrap_csv: Path) -> Dict[Tuple[str, str, str, str], float]:
    """Load pairwise bootstrap p-values. Keys stored in both A→B and B→A directions."""
    pv: Dict[Tuple[str, str, str, str], float] = {}
    if not bootstrap_csv.exists():
        logger.warning(f"Bootstrap CSV not found: {bootstrap_csv}")
        return pv
    with open(bootstrap_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["p_value"])
            except (ValueError, KeyError):
                continue
            p, t, ma, mb = row["protocol"], row["target"], row["method_a"], row["method_b"]
            pv[(p, t, ma, mb)] = v
            pv[(p, t, mb, ma)] = v  # symmetric storage
    return pv


# ─── Figure 1 — main_results_bars ─────────────────────────────────────────────

def fig_main_results_bars(
    cells: Dict[Tuple[str, str, str], CellData],
    pvalues: Dict[Tuple[str, str, str, str], float],
    figures_dir: Path,
) -> None:
    """4-panel grouped bar chart: test macro-F1 per method, per protocol."""
    fig, axes = plt.subplots(
        1, 4, figsize=(18, 5), sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "Test Macro-F1 by Method and Protocol",
        fontsize=13, fontweight="bold",
    )

    y_min, y_max = 0.20, 0.80

    for ax, (protocol, target) in zip(axes, PANELS):
        method_order = MIXED_METHOD_ORDER if protocol == "mixed" else LODO_METHOD_ORDER
        baseline = method_order[0]
        n_methods = len(method_order)
        x_pos = np.arange(n_methods)

        for i, method in enumerate(method_order):
            key = (method, protocol, target)
            data = cells.get(key)

            if data is None or len(data["test_f1s"]) == 0:
                ax.bar(
                    i, 0.01, width=0.6,
                    color="lightgray", hatch="///",
                    edgecolor="black", linewidth=0.8,
                )
                ax.text(
                    i, y_min + 0.015, "n/a",
                    ha="center", va="bottom", fontsize=7, color="#888888",
                )
                continue

            n_seeds = len(data["test_f1s"])
            mean_f1 = float(np.mean(data["test_f1s"]))
            std_f1  = float(np.std(data["test_f1s"])) if n_seeds > 1 else 0.0

            is_focal = method.endswith("_focal")
            color  = COLOR_FOCAL if is_focal else COLOR_NOFOCAL
            hatch  = "///" if n_seeds < 3 else ""

            ax.bar(
                i, mean_f1, width=0.6,
                color=color, alpha=0.88,
                hatch=hatch,
                edgecolor="black" if hatch else "white",
                linewidth=0.8,
            )
            if n_seeds > 1:
                ax.errorbar(
                    i, mean_f1, yerr=std_f1,
                    fmt="none", color="black", capsize=3.5, linewidth=1.1,
                )

            # "n=1" annotation for under-seeded cells
            if n_seeds < 3:
                label_y = min(mean_f1 + std_f1 + 0.018, y_max - 0.04)
                ax.text(
                    i, label_y, f"n={n_seeds}",
                    ha="center", va="bottom", fontsize=7, color="#444444",
                )

            # Significance star vs CE baseline
            if method != baseline:
                pv = pvalues.get((protocol, target, baseline, method))
                if pv is not None and pv < 0.05:
                    star_y = mean_f1 + std_f1 + (0.05 if n_seeds < 3 else 0.03)
                    star_y = min(star_y, y_max - 0.02)
                    ax.text(
                        i, star_y, "★",
                        ha="center", va="bottom",
                        fontsize=11, color="black",
                    )

        panel_title = PANEL_TITLES.get((protocol, target), f"{protocol}/{target}")
        ax.set_title(panel_title, fontsize=10, fontweight="semibold", pad=6)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [METHOD_DISPLAY.get(m, m) for m in method_order],
            rotation=38, ha="right", fontsize=8,
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.45, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Test Macro-F1", fontsize=10)

    legend_handles = [
        mpatches.Patch(color=COLOR_NOFOCAL, alpha=0.88, label="No focal loss"),
        mpatches.Patch(color=COLOR_FOCAL,   alpha=0.88, label="Focal loss"),
        mpatches.Patch(
            facecolor="lightgray", hatch="///", edgecolor="black",
            label="n < 3 seeds",
        ),
        plt.Line2D(
            [0], [0], marker="★", color="black", linestyle="none",
            markersize=9, label="p < 0.05 vs CE baseline",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.10), ncol=4,
        fontsize=9, frameon=True, framealpha=0.9,
    )

    logger.info("Generating Figure 1: main_results_bars")
    _save_figure(fig, figures_dir, "main_results_bars")


# ─── Figure 2 — val_test_gap ──────────────────────────────────────────────────

def fig_val_test_gap(
    cells: Dict[Tuple[str, str, str], CellData],
    figures_dir: Path,
) -> None:
    """3-panel grouped bar chart showing val vs test F1 with gap annotations (LODO only)."""
    lodo_targets = ["goemotions", "isear", "wassa21"]
    method_order = LODO_METHOD_ORDER
    n_methods = len(method_order)

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Validation → Test Macro-F1 Gap (LODO)\n"
        "Gap is method-invariant for GoEmotions/ISEAR; focal loss shrinks it on WASSA-21",
        fontsize=12, fontweight="bold",
    )

    bar_w = 0.35
    x_pos = np.arange(n_methods)
    COLOR_VAL  = "#A8C8E8"  # light blue — validation
    COLOR_TEST = "#2B5EA7"  # dark blue — test

    for ax, target in zip(axes, lodo_targets):
        val_means, test_means = [], []

        for method in method_order:
            data = cells.get((method, "lodo", target))
            if data and len(data["val_f1s"]) > 0:
                val_means.append(float(np.nanmean(data["val_f1s"])))
                test_means.append(float(np.nanmean(data["test_f1s"])))
            else:
                val_means.append(float("nan"))
                test_means.append(float("nan"))

        # Draw bars (skip NaN gracefully via masking)
        for i, (vm, tm) in enumerate(zip(val_means, test_means)):
            if np.isnan(vm):
                continue
            ax.bar(i - bar_w / 2, vm, bar_w, color=COLOR_VAL, edgecolor="white", linewidth=0.8)
            ax.bar(i + bar_w / 2, tm, bar_w, color=COLOR_TEST, edgecolor="white", linewidth=0.8, alpha=0.9)

            # Gap annotation above the pair
            gap = vm - tm
            annotation_y = max(vm, tm) + 0.015
            ax.text(
                i, annotation_y,
                f"Δ{gap:+.2f}",
                ha="center", va="bottom", fontsize=7.5,
                color="#333333", fontweight="semibold",
            )

        target_display = {"goemotions": "GoEmotions", "isear": "ISEAR", "wassa21": "WASSA-21"}
        ax.set_title(
            f"LODO → {target_display.get(target, target)}",
            fontsize=10, fontweight="semibold", pad=5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [METHOD_DISPLAY.get(m, m) for m in method_order],
            rotation=38, ha="right", fontsize=8,
        )
        ax.set_ylabel("Macro-F1", fontsize=9)
        ax.set_ylim(0.15, 0.85)
        ax.grid(axis="y", linewidth=0.4, alpha=0.45, linestyle="--")
        ax.set_axisbelow(True)

    # Shared legend for val / test
    legend_handles = [
        mpatches.Patch(color=COLOR_VAL,  label="Validation Macro-F1"),
        mpatches.Patch(color=COLOR_TEST, label="Test Macro-F1"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.10), ncol=2,
        fontsize=9, frameon=True, framealpha=0.9,
    )

    logger.info("Generating Figure 2: val_test_gap")
    _save_figure(fig, figures_dir, "val_test_gap")


# ─── Figure 3 — dann_lambda_dynamics ─────────────────────────────────────────

def _sigmoid_lambda(p: np.ndarray, lambda_max: float = 0.5, gamma: float = 10.0) -> np.ndarray:
    """Standard DANN sigmoid schedule: λ = λ_max · (2/(1+exp(−γp)) − 1)."""
    return lambda_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def _parse_dann_log(log_path: Path) -> Tuple[List[int], List[float], List[float]]:
    """Extract (epoch, domain_loss, lambda) from a trainer seed log file."""
    epochs, domain_losses, lambdas = [], [], []
    if not log_path.exists():
        return epochs, domain_losses, lambdas
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                domain_losses.append(float(m.group(2)))
                lambdas.append(float(m.group(3)))
    return epochs, domain_losses, lambdas


def fig_dann_lambda_dynamics(output_dir: Path, figures_dir: Path) -> None:
    """2-panel figure: (A) analytical λ schedule; (B) domain_loss over epochs."""
    fig, (ax_lambda, ax_dloss) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True,
    )
    fig.suptitle("DANN Domain-Adversarial Dynamics", fontsize=12, fontweight="bold")

    # ── Panel A: Analytical sigmoid schedule ──────────────────────────────────
    p_vals = np.linspace(0, 1, 300)
    ax_lambda.plot(
        p_vals, _sigmoid_lambda(p_vals),
        color="#2B5EA7", linewidth=2.2, label="λ (sigmoid, γ=10, λ_max=0.5)",
    )
    ax_lambda.axhline(0.5, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)
    ax_lambda.set_xlabel("Training progress p = step / total_steps", fontsize=9)
    ax_lambda.set_ylabel("Lambda (λ)", fontsize=9)
    ax_lambda.set_title("(A) Lambda Schedule", fontsize=10, fontweight="semibold")
    ax_lambda.set_xlim(0, 1)
    ax_lambda.set_ylim(-0.02, 0.58)
    ax_lambda.legend(fontsize=8)
    ax_lambda.grid(linewidth=0.4, alpha=0.4, linestyle="--")

    # ── Panel B: Empirical domain_loss from training logs ─────────────────────
    lodo_targets   = ["goemotions", "isear", "wassa21"]
    target_display = {"goemotions": "GoEmotions", "isear": "ISEAR", "wassa21": "WASSA-21"}
    colors_b = ["#E64B35", "#4DBBD5", "#00A087"]
    any_log_found = False

    for target, color in zip(lodo_targets, colors_b):
        experiment_name = f"dann_lodo_{target}"
        log_path = output_dir / "logs" / experiment_name / "seed_42.log"
        epochs, domain_losses, _ = _parse_dann_log(log_path)
        if not epochs:
            logger.warning(f"  No DANN log found: {log_path}")
            continue
        any_log_found = True
        ax_dloss.plot(
            epochs, domain_losses,
            color=color, linewidth=2.0, marker="o", markersize=3.5,
            label=target_display[target],
        )

    if not any_log_found:
        ax_dloss.text(
            0.5, 0.5, "No DANN log files found\n(run training to populate)",
            ha="center", va="center", transform=ax_dloss.transAxes,
            fontsize=10, color="gray",
        )

    ax_dloss.set_xlabel("Epoch", fontsize=9)
    ax_dloss.set_ylabel("Domain Loss (mean per update step)", fontsize=9)
    ax_dloss.set_title(
        "(B) Domain Loss — DANN × LODO Targets (seed 42)",
        fontsize=10, fontweight="semibold",
    )
    ax_dloss.legend(fontsize=8)
    ax_dloss.grid(linewidth=0.4, alpha=0.4, linestyle="--")

    logger.info("Generating Figure 3: dann_lambda_dynamics")
    _save_figure(fig, figures_dir, "dann_lambda_dynamics")


# ─── Figure 4 — confusion_matrix_wassa ────────────────────────────────────────

def _run_wassa_inference(
    output_dir: Path,
    method: str,
    config_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load best checkpoint for method on LODO-wassa21 (seed 42) and run inference.

    Returns (y_pred, y_true) as int arrays, or None if the checkpoint is missing.
    All heavy imports are deferred so other figures can render even without GPU.
    """
    experiment_name = f"{method}_lodo_wassa21"
    ckpt_path = output_dir / "checkpoints" / experiment_name / "seed_42" / "best.pt"

    if not ckpt_path.exists():
        # Try recursive search under output_dir in case runs were sharded.
        candidates = list(output_dir.rglob(f"checkpoints/{experiment_name}/seed_42/best.pt"))
        if candidates:
            ckpt_path = candidates[0]
        else:
            logger.warning(f"  Checkpoint not found for {method}/lodo/wassa21: {ckpt_path}")
            return None

    import torch
    import yaml
    from src.data.builder import DataConfig, build_datasets
    from src.data.protocols import build_lodo_protocol
    from src.data.torch_dataset import EmotionCollator, EmotionTorchDataset
    from src.models.backbone import BackboneConfig, DebertaBackbone
    from src.models.classifier import EmotionClassifier
    from torch.utils.data import DataLoader

    with open(config_path, encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    data_cfg = cfg.get("data", {})
    data_config = DataConfig(
        isear_csv=Path(data_cfg.get("isear_csv", "data/raw/isear.csv")),
        wassa_dir=Path(data_cfg.get("wassa_dir", "data/raw/wassa21/")),
        goemotions_strict_single_ekman=bool(
            data_cfg.get("goemotions_strict_single_ekman", True)
        ),
        isear_split_seed=int(data_cfg.get("isear_split_seed", 42)),
        isear_train_frac=float(data_cfg.get("isear_train_frac", 0.8)),
        isear_val_frac=float(data_cfg.get("isear_val_frac", 0.1)),
        min_text_length=int(data_cfg.get("min_text_length", 10)),
        goemotions_min_text_length=int(data_cfg.get("goemotions_min_text_length", 3)),
    )
    per_domain = build_datasets(data_config)
    split = build_lodo_protocol(per_domain, "wassa21")
    test_examples = split.test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg_dict = cfg.get("model", {})
    backbone_cfg = BackboneConfig.from_dict(model_cfg_dict)
    backbone = DebertaBackbone(backbone_cfg)
    tokenizer = backbone.get_tokenizer()

    model = EmotionClassifier(
        backbone=backbone,
        num_labels=backbone_cfg.num_labels,
        head_dropout=backbone_cfg.dropout,
        loss_fn=None,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt["model_state"]
    # Drop focal-loss buffer keys (may be absent in non-focal checkpoints).
    filtered_state = {k: v for k, v in raw_state.items() if "loss_fn." not in k}
    model.load_state_dict(filtered_state, strict=True)
    model.to(device).eval()

    max_length = int(model_cfg_dict.get("max_length", 256))
    batch_size = int(cfg.get("training", {}).get("batch_size", 16))
    dataset   = EmotionTorchDataset(test_examples)
    collator  = EmotionCollator(tokenizer=tokenizer, max_length=max_length)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(**batch)
            all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    return np.array(all_preds, dtype=int), np.array(all_labels, dtype=int)


def _plot_cm_panel(
    ax: plt.Axes,
    cm: np.ndarray,
    title: str,
    labels: List[str],
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Draw a single normalised confusion matrix panel."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=8)

    # Annotate cells
    thresh = (vmin + vmax) / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if val > thresh else "black",
            )

    return im


def fig_confusion_matrix_wassa(
    output_dir: Path,
    figures_dir: Path,
    config_path: Path,
) -> None:
    """Side-by-side row-normalised confusion matrices: source_only vs source_only_focal on WASSA-21."""
    from sklearn.metrics import confusion_matrix as sk_cm

    methods = ["source_only", "source_only_focal"]
    titles  = [
        "(A) CE — source only",
        "(B) CE + Focal loss",
    ]
    results: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    for method in methods:
        logger.info(f"  Running WASSA-21 inference for: {method}")
        try:
            results[method] = _run_wassa_inference(output_dir, method, config_path)
        except Exception as exc:
            logger.warning(f"  Inference failed for {method}: {exc}")
            results[method] = None

    if all(v is None for v in results.values()):
        logger.warning("Figure 4 skipped: no inference results available.")
        return

    # Determine a shared colour scale [0, vmax] across both matrices.
    cms = {}
    present_labels: List[str] = EKMAN_LABELS  # fallback
    for method in methods:
        if results[method] is None:
            continue
        y_pred, y_true = results[method]
        present_ids = sorted(np.unique(y_true).tolist())
        present_labels = [EKMAN_LABELS[i] for i in present_ids]
        raw = sk_cm(y_true, y_pred, labels=present_ids).astype(float)
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cms[method] = raw / row_sums  # row-normalised (recall per class)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        "Row-Normalised Confusion Matrix — WASSA-21 Test Set (LODO, seed 42)",
        fontsize=12, fontweight="bold",
    )

    last_im = None
    for ax, method, title in zip(axes, methods, titles):
        if method not in cms:
            ax.axis("off")
            ax.text(
                0.5, 0.5, f"No data for\n{method}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray",
            )
            continue
        im = _plot_cm_panel(ax, cms[method], title, present_labels)
        last_im = im

    # Shared colour bar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("Recall (row-normalised)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Diagonal difference annotation: highlight which classes focal recovers
    if len(cms) == 2 and all(m in cms for m in methods):
        diff_diag = np.diag(cms["source_only_focal"]) - np.diag(cms["source_only"])
        for i, (label, delta) in enumerate(zip(present_labels, diff_diag)):
            if abs(delta) >= 0.03:
                sign = "↑" if delta > 0 else "↓"
                axes[1].text(
                    i, -0.55, f"{sign}{abs(delta):.2f}",
                    ha="center", va="top", fontsize=7.5,
                    color="#E64B35" if delta < 0 else "#00A087",
                    transform=axes[1].get_xaxis_transform(),
                )

    logger.info("Generating Figure 4: confusion_matrix_wassa")
    _save_figure(fig, figures_dir, "confusion_matrix_wassa")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paper figures from training results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/cross_emotion_data/outputs",
        help="Root that contains results/, logs/, checkpoints/, bootstrap/.",
    )
    p.add_argument(
        "--figures-dir",
        default=None,
        help="Where to write figures. Defaults to <output-dir>/figures.",
    )
    p.add_argument(
        "--bootstrap-csv",
        default=None,
        help="Path to pairwise_pvalues.csv. "
             "Defaults to <output-dir>/bootstrap/pairwise_pvalues.csv.",
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config (needed for Figure 4 inference data loading).",
    )
    p.add_argument(
        "--skip-fig4", action="store_true",
        help="Skip Figure 4 (confusion matrices). Useful when no GPU is available.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_dir  = Path(args.output_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else output_dir / "figures"
    bootstrap_csv = (
        Path(args.bootstrap_csv)
        if args.bootstrap_csv
        else output_dir / "bootstrap" / "pairwise_pvalues.csv"
    )
    config_path = Path(args.config)

    logger.info(f"Output root : {output_dir}")
    logger.info(f"Figures dir : {figures_dir}")
    logger.info(f"Bootstrap   : {bootstrap_csv}")

    cells   = discover_results(output_dir)
    pvalues = load_pvalues(bootstrap_csv)

    # Figure 1 — main results bar chart
    try:
        fig_main_results_bars(cells, pvalues, figures_dir)
    except Exception as exc:
        logger.error(f"Figure 1 failed: {exc}", exc_info=True)

    # Figure 2 — val / test gap (LODO)
    try:
        fig_val_test_gap(cells, figures_dir)
    except Exception as exc:
        logger.error(f"Figure 2 failed: {exc}", exc_info=True)

    # Figure 3 — lambda dynamics & domain loss
    try:
        fig_dann_lambda_dynamics(output_dir, figures_dir)
    except Exception as exc:
        logger.error(f"Figure 3 failed: {exc}", exc_info=True)

    # Figure 4 — confusion matrices (requires checkpoints + GPU)
    if args.skip_fig4:
        logger.info("Figure 4 skipped (--skip-fig4).")
    elif not config_path.exists():
        logger.warning(f"Figure 4 skipped: config not found at {config_path}.")
    else:
        try:
            fig_confusion_matrix_wassa(output_dir, figures_dir, config_path)
        except Exception as exc:
            logger.error(f"Figure 4 failed: {exc}", exc_info=True)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
