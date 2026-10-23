"""Export three readable, vector column groups from the existing learning plot."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))

from matplotlib.transforms import Bbox
from cifar10_lab import report

DESTINATION = ROOT / "tmp" / "pdfs" / "assets"
DESTINATION.mkdir(parents=True, exist_ok=True)


def save_column_groups(fig, name):
    if name != "learning_curves":
        raise ValueError(name)
    for ax in fig.axes:
        ax.tick_params(labelleft=True)
        ax.set_ylabel("Accuracy (%)")
    for ax in fig.axes[:3]:
        ax.legend(loc="lower right", fontsize=8)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    renderer = fig.canvas.get_renderer()
    boxes = []
    for column in range(3):
        axes = [fig.axes[row * 3 + column] for row in range(3)]
        box = Bbox.union([ax.get_tightbbox(renderer) for ax in axes])
        boxes.append(box.transformed(fig.dpi_scale_trans.inverted()).padded(0.05))
    for column, inches in enumerate(boxes):
        for index, ax in enumerate(fig.axes):
            ax.set_visible(index % 3 == column)
        fig.savefig(DESTINATION / f"learning-curves-{column + 1}.pdf", bbox_inches=inches)
    report.plt.close(fig)


def main():
    plan = json.loads((ROOT / "results" / "evaluation_plan.json").read_text())
    summaries = {name: json.loads((ROOT / "results" / "runs" / name / "summary.json").read_text())
                 for name in plan["models"]}
    report.save = save_column_groups
    report.plot_curves(summaries)


if __name__ == "__main__":
    main()
