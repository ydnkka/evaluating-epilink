from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
import seaborn as sns
from matplotlib.figure import Figure


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_figure(fig: "Figure", out_base: Path, formats: list[str]) -> None:
    for ext in formats:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)


def set_seaborn_paper_context(font_scale=1.5) -> None:
    sns.set_theme(
        context="paper",       # scales fonts/lines appropriately
        style="white",     # subtle horizontal grid
        font="sans-serif",
        font_scale=font_scale,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "grid.alpha": 0.3,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
        }

    )
