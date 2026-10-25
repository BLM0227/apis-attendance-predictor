"""Matplotlib figures (built from src/evidence.py), styled like the APIS data-story dashboard:
blue highlight, pale lavender-grey for context, dashed red reference line, no chart borders."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import evidence as ev
from .config import RECOMMENDED_MODEL

BLUE, MID, PALE, RED, INK, MUTED = "#1F5AA6", "#8FA6CC", "#D3D9E6", "#C0392B", "#1B2437", "#6B7488"
MODEL_COLORS = {"Logistic Regression": BLUE, "Decision Tree": MID, "Gaussian Naive Bayes": PALE}


def _clean(ax, grid_axis=None):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(PALE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#EEF1F7", lw=1)
        ax.set_axisbelow(True)


def _figure(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    return fig, ax


def confusion_figure(model: str):
    cm = np.array(ev.confusion[model])
    fig, ax = _figure(4.2, 3.6)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.1)
    labels = ["Not attending", "Attending"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    ax.set_title(model + (" ★" if model == RECOMMENDED_MODEL else ""), fontsize=10, color=INK, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def not_attending_metrics_figure():
    df = ev.test_not_attending
    metrics = ["Precision (not attending)", "Recall (not attending)", "F1 (not attending)", "AP (not attending)"]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = _figure(7.6, 3.8)
    for k, model in enumerate(ev.MODELS):
        vals = df.loc[df["Model"] == model, metrics].iloc[0].values
        bars = ax.bar(x + (k - 1) * w, vals, w, label=model, color=MODEL_COLORS[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [m.replace(" (not attending)", "") for m in metrics])
    ax.set_ylim(0, 112)
    ax.set_ylabel("%", color=MUTED)
    _clean(ax, "y")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    return fig


def generalization_figure():
    df = ev.generalization
    x = np.arange(len(df))
    w = 0.26
    fig, ax = _figure(7, 3.6)
    for off, col, color, lab in [(-w, "Training accuracy", PALE, "Training"),
                                 (0, "CV accuracy", MID, "5-fold CV"),
                                 (w, "Test accuracy", BLUE, "Test")]:
        bars = ax.bar(x + off, df[col], w, label=lab, color=color)
        for b, v in zip(bars, df[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, df["Model"])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color=MUTED)
    _clean(ax, "y")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    return fig


def threshold_figure():
    t = ev.threshold_oof
    sel = ev.SELECTED_THRESHOLD
    fig, ax = _figure(7, 3.8)
    for col, c in [("Precision (not attending)", MID), ("Recall (not attending)", BLUE),
                   ("F1 (not attending)", INK)]:
        ax.plot(t["Threshold"], t[col], marker="o", lw=2, label=col.replace(" (not attending)", ""), color=c)
    ax.axvline(0.50, color=RED, ls="--", lw=1.2)
    ax.text(0.505, 33, "default 0.50", fontsize=8, color=RED)
    if abs(sel - 0.50) > 1e-9:
        ax.axvline(sel, color=BLUE, ls=":", lw=1.2)
        ax.text(sel + 0.005, 96, f"selected {sel:.2f}", fontsize=8, color=BLUE)
    ax.set_xlabel("Threshold on P(not attending)", color=MUTED)
    ax.set_ylabel("%", color=MUTED)
    ax.set_ylim(30, 100)
    _clean(ax, "y")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def dotplot(df, label_col, value_col="Recall @selected", reference=None, w=6.6):
    """Dot plot like the dashboard's age chart. Lowest-recall group is highlighted (the concern).
    Labels show the number of actual not-attending individuals the recall is based on."""
    labels = [f"{a}  (n={n})" for a, n in zip(df[label_col], df["Not attending (n)"])]
    vals = list(df[value_col])
    ref = ev.F["te_rec_s"] if reference is None else reference
    ypos = np.arange(len(labels))[::-1]
    fig, ax = _figure(w, 0.55 * len(labels) + 1.4)
    lo = int(np.argmin(vals))
    for i, (y, v) in enumerate(zip(ypos, vals)):
        ax.plot([0, v], [y, y], color="#E3E8F1", lw=1.5, zorder=1)
        ax.scatter(v, y, s=70, color=BLUE if i == lo else MID, zorder=3)
        ax.text(v + 2, y, f"{v:.1f}%", va="center", fontsize=9, color=INK,
                fontweight="bold" if i == lo else "normal")
    ax.axvline(ref, color=RED, ls="--", lw=1.2)
    ax.text(ref, len(labels) - 0.35, f"All groups: {ref:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(labels) - 0.1)
    ax.set_xlabel("Recall for not-attending class (%)", color=MUTED)
    _clean(ax)
    fig.tight_layout()
    return fig


def region_figure(df, value_col="Recall @selected"):
    """Horizontal bars; lowest-recall region highlighted, dashed line for all regions."""
    labels, vals = list(df["Region"]), list(df[value_col])
    ref = ev.F["te_rec_s"]
    lo = int(np.argmin(vals))
    fig, ax = _figure(7.4, 0.33 * len(labels) + 1.2)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=[BLUE if i == lo else PALE for i in range(len(vals))], height=0.7)
    for yy, v, n in zip(y, vals, df["Not attending (n)"]):
        ax.text(v + 1, yy, f"{v:.1f}%  (n={n})", va="center", fontsize=8.5, color=INK)
    ax.axvline(ref, color=RED, ls="--", lw=1.2)
    ax.text(ref, len(labels) - 0.2, f"All regions: {ref:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.6, len(labels) + 0.4)
    ax.set_xlabel("Recall for not-attending class (%)", color=MUTED)
    _clean(ax)
    fig.tight_layout()
    return fig
