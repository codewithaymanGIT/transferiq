"""
Regenerate the README charts in docs/charts/ from saved evaluation outputs.

Inputs are the reports the training scripts already write -- nothing here
refits a model or types a number by hand:

  docs/model-comparison.md     <- scripts/train/train_baseline_models.py
  docs/temporal-validation.md  <- scripts/train/temporal_validation.py

Optional input (not produced by any script yet, so the chart is skipped
when it is missing rather than faked):

  data/processed/heldout_predictions.csv
      columns: actual, predicted, low_bound, high_bound  (EUR millions)

Usage:
    python scripts/docs/make_charts.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
OUT = DOCS / "charts"

# 8in x 100dpi = 800px wide, per the README readability target.
WIDTH_IN = 8
DPI = 100
BASELINE_COLOR = "#9aa0a6"
MODEL_COLOR = "#3b6ea8"
HIGHLIGHT_COLOR = "#b5651d"


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
        }
    )


def _read_table(md_path: Path, heading: str | None) -> list[dict[str, str]]:
    """Return the first markdown table after `heading` (or the first table in
    the file when heading is None) as a list of row dicts keyed by header."""
    text = md_path.read_text(encoding="utf-8")
    if heading is not None:
        idx = text.find(heading)
        if idx == -1:
            raise ValueError(f"Heading {heading!r} not found in {md_path}")
        text = text[idx:]
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("|"))
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows = []
    for ln in lines[start + 2 :]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows


def _pooled_out_of_time(md_path: Path) -> tuple[float, float, int]:
    text = md_path.read_text(encoding="utf-8")
    r2 = re.search(r"\*\*R2: (-?[\d.]+)\*\* across all (\d+) held-out", text)
    mae = re.search(r"\*\*MAE: ([\d.]+)m EUR\*\*", text)
    if not (r2 and mae):
        raise ValueError(f"Pooled out-of-time result not found in {md_path}")
    return float(r2.group(1)), float(mae.group(1)), int(r2.group(2))


def chart_model_comparison(cv_rows: list[dict[str, str]]) -> Path:
    names = [r["Model"] for r in cv_rows]
    r2 = [float(r["Mean R2"]) for r in cv_rows]
    r2_std = [float(r["R2 std dev"]) for r in cv_rows]
    mae = [float(r["Mean MAE (EUR m)"]) for r in cv_rows]
    colors = [BASELINE_COLOR if n == "Median baseline" else MODEL_COLOR for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.6), sharey=True)
    y = range(len(names))
    ax1.barh(y, r2, xerr=r2_std, color=colors, capsize=3, error_kw={"elinewidth": 1})
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_yticks(list(y), names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean R² (error bar = std across folds)")
    ax1.set_title("5-fold CV R² (higher is better)", fontsize=10)

    ax2.barh(y, mae, color=colors)
    for i, v in enumerate(mae):
        ax2.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=9)
    ax2.set_xlabel("Mean MAE, EUR millions")
    ax2.set_title("5-fold CV MAE (lower is better)", fontsize=10)
    ax2.set_xlim(0, max(mae) * 1.18)

    fig.suptitle("Models vs. a median-fee baseline", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    path = OUT / "model_comparison.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def chart_cv_vs_out_of_time(cv_rows: list[dict[str, str]], season_rows: list[dict[str, str]], pooled) -> Path:
    rf = next(r for r in cv_rows if r["Model"] == "Random Forest")
    cv_r2 = float(rf["Mean R2"])
    oot_r2, oot_mae, n_oot = pooled

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    labels = ["Random 5-fold CV", "Walk-forward\n(out-of-time)"]
    vals = [cv_r2, oot_r2]
    bars = ax1.bar(labels, vals, color=[MODEL_COLOR, HIGHLIGHT_COLOR], width=0.55)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("R²")
    ax1.set_title("Random Forest: same data, two validations", fontsize=10)

    seasons = [r["Test season"] for r in season_rows]
    s_r2 = [float(r["R2"]) for r in season_rows]
    n_test = [r["Test rows"] for r in season_rows]
    colors = [HIGHLIGHT_COLOR if v >= 0 else BASELINE_COLOR for v in s_r2]
    ax2.bar(range(len(seasons)), s_r2, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(range(len(seasons)), [s.replace("-20", "-") for s in seasons], rotation=30, ha="right")
    for i, (v, n) in enumerate(zip(s_r2, n_test)):
        ax2.text(i, v + (0.01 if v >= 0 else -0.03), f"n={n}", ha="center", fontsize=8)
    ax2.set_ylim(min(s_r2) - 0.06, max(s_r2) + 0.04)
    ax2.set_ylabel("R² on that season only")
    ax2.set_title(f"Per test season (pooled: R² {oot_r2:.3f}, n={n_oot})", fontsize=10)

    fig.tight_layout()
    path = OUT / "cv_vs_out_of_time.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def chart_shap(shap_rows: list[dict[str, str]]) -> Path:
    key = next(k for k in shap_rows[0] if k != "Feature")
    feats = [r["Feature"] for r in shap_rows]
    vals = [float(r[key]) for r in shap_rows]

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.8))
    ax.barh(range(len(feats)), vals, color=MODEL_COLOR)
    ax.set_yticks(range(len(feats)), feats)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (log-fee units)")
    ax.set_title("Random Forest feature impact, full training set (descriptive, not causal)", fontsize=10)
    fig.tight_layout()
    path = OUT / "shap_importance.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def chart_heldout(csv_path: Path) -> Path:
    import pandas as pd

    df = pd.read_csv(csv_path).sort_values("actual")
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.2))
    ax.vlines(df["actual"], df["low_bound"], df["high_bound"], color=BASELINE_COLOR, alpha=0.5, linewidth=1,
              label="Prediction interval")
    ax.scatter(df["actual"], df["predicted"], s=12, color=MODEL_COLOR, label="Predicted")
    lim = max(df["actual"].max(), df["predicted"].max())
    ax.plot([0, lim], [0, lim], color="black", linewidth=0.8, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual fee, EUR millions")
    ax.set_ylabel("Predicted fee, EUR millions")
    ax.set_title(f"Held-out predictions vs. actual fees (n={len(df)})", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = OUT / "predicted_vs_actual.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = DOCS / "model-comparison.md"
    temporal = DOCS / "temporal-validation.md"

    cv_rows = _read_table(comparison, "## Cross-validated results")
    shap_rows = _read_table(comparison, "## SHAP feature importance")
    season_rows = _read_table(temporal, "## Per-season breakdown")
    pooled = _pooled_out_of_time(temporal)

    written = [
        chart_model_comparison(cv_rows),
        chart_cv_vs_out_of_time(cv_rows, season_rows, pooled),
        chart_shap(shap_rows),
    ]

    heldout = REPO_ROOT / "data" / "processed" / "heldout_predictions.csv"
    if heldout.exists():
        written.append(chart_heldout(heldout))
    else:
        print(f"Skipped predicted-vs-actual chart: {heldout.relative_to(REPO_ROOT)} not found "
              "(no script saves held-out predictions yet).")

    for p in written:
        print(f"Wrote {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
