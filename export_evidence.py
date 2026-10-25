"""Export the saved evaluation evidence (deliverable 8): CSV tables and PNG figures in assets/evidence/.
Run after train_models.py:  python export_evidence.py"""
from pathlib import Path

import matplotlib.pyplot as plt

from src import evidence as ev
from src import plots
from src.config import ROOT

OUT = ROOT / "assets" / "evidence"


def main():
    if not ev.AVAILABLE:
        raise SystemExit("models/evidence.json not found. Run `python train_models.py` first.")
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"cv_main": ev.cv_main, "cv_average_precision": ev.cv_ap, "test_overall": ev.test_overall,
              "test_not_attending": ev.test_not_attending, "generalization": ev.generalization,
              "threshold_validation_oof": ev.threshold_oof, "threshold_test_informational": ev.threshold_test,
              "subgroup_age": ev.subgroup_age, "subgroup_region": ev.subgroup_region,
              "subgroup_sex": ev.subgroup_sex, "subgroup_residence": ev.subgroup_residence}
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    figs = {"test_not_attending_metrics": plots.not_attending_metrics_figure(),
            "generalization": plots.generalization_figure(), "threshold_tradeoff_validation": plots.threshold_figure(),
            "subgroup_age": plots.dotplot(ev.subgroup_age, "Age group"),
            "subgroup_region": plots.region_figure(ev.subgroup_region)}
    for m in ev.MODELS:
        figs["confusion_" + m.lower().replace(" ", "_")] = plots.confusion_figure(m)
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(tables)} tables and {len(figs)} figures to {OUT}")


if __name__ == "__main__":
    main()
