#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy", "datasets"]
# ///
"""
Profile the evaluation dataset loaded from HuggingFace (irfanjamil/BlicketEnv_Eval_Set).

Provides:
  - Counts per rule type
  - Counts per number of objects
  - Counts per (num_objects, num_blickets) combination
  - Counts per (num_blickets, rule_type) combination
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datasets import load_dataset

HF_REPO = "irfanjamil/BlicketEnv_Eval_Set"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_from_hf(repo: str = HF_REPO, split: str = "eval") -> list[dict]:
    """Load dataset from HuggingFace and normalise info to a dict."""
    ds = load_dataset(repo, split=split)
    examples = []
    for row in ds:
        record = dict(row)
        if isinstance(record["info"], str):
            record["info"] = json.loads(record["info"])
        examples.append(record)
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Counting helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_rule_types(examples: list[dict]) -> Counter:
    return Counter(r["info"]["rule_type"] for r in examples)


def count_num_objects(examples: list[dict]) -> Counter:
    return Counter(r["info"]["num_objects"] for r in examples)


def count_object_blicket_combos(examples: list[dict]) -> Counter:
    """Count (num_objects, num_blickets) pairs."""
    return Counter(
        (r["info"]["num_objects"], r["info"]["num_blickets"]) for r in examples
    )


def count_blicket_rule_combos(examples: list[dict]) -> Counter:
    """Count (num_blickets, rule_type) pairs."""
    return Counter(
        (r["info"]["num_blickets"], r["info"]["rule_type"]) for r in examples
    )


# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def print_table(rows: list[tuple], headers: list[str]) -> None:
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def print_summary(examples: list[dict]) -> None:
    print_header("DATASET SUMMARY")
    print(f"  Unique examples : {len(examples)}")

    rt = count_rule_types(examples)
    print(f"  Rule types      : {dict(rt)}")

    no = count_num_objects(examples)
    print(f"  Num-objects range : {min(no)} – {max(no)}")

    nb = Counter(r["info"]["num_blickets"] for r in examples)
    print(f"  Num-blickets range : {min(nb)} – {max(nb)}")


def print_rule_type_counts(counts: Counter) -> None:
    print_header("1. COUNTS PER RULE TYPE")
    rows = sorted(counts.items())
    print_table([(rt, cnt) for rt, cnt in rows], ["Rule Type", "Count"])


def print_num_objects_counts(counts: Counter) -> None:
    print_header("2. COUNTS PER NUMBER OF OBJECTS")
    rows = sorted(counts.items())
    print_table([(no, cnt) for no, cnt in rows], ["Num Objects", "Count"])


def print_object_blicket_combos(counts: Counter) -> None:
    print_header("3. COUNTS PER (NUM_OBJECTS, NUM_BLICKETS) COMBINATION")
    rows = sorted(counts.items())
    print_table(
        [(no, nb, cnt) for (no, nb), cnt in rows],
        ["Num Objects", "Num Blickets", "Count"],
    )


def print_blicket_rule_combos(counts: Counter) -> None:
    print_header("4. COUNTS PER (NUM_BLICKETS, RULE_TYPE) COMBINATION")
    rows = sorted(counts.items())
    print_table(
        [(nb, rt, cnt) for (nb, rt), cnt in rows],
        ["Num Blickets", "Rule Type", "Count"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "conjunctive": "#4C72B0",
    "disjunctive": "#DD8452",
    "bar":         "#4C72B0",
}


def _bar(ax, labels, values, title, xlabel, ylabel, color=PALETTE["bar"]):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_rule_type(ax, counts: Counter) -> None:
    labels = sorted(counts)
    values = [counts[l] for l in labels]
    colors = [PALETTE.get(l, PALETTE["bar"]) for l in labels]
    _bar(ax, labels, values, "Rule Type Distribution", "Rule Type", "Count")
    for bar, lbl in zip(ax.patches, labels):
        bar.set_color(PALETTE.get(lbl, PALETTE["bar"]))


def plot_num_objects(ax, counts: Counter) -> None:
    labels = sorted(counts)
    values = [counts[l] for l in labels]
    _bar(ax, labels, values, "Num-Objects Distribution", "Number of Objects", "Count")


def plot_object_blicket_heatmap(ax, counts: Counter) -> None:
    """Heatmap of (num_objects × num_blickets)."""
    all_no = sorted({k[0] for k in counts})
    all_nb = sorted({k[1] for k in counts})
    matrix = np.zeros((len(all_nb), len(all_no)), dtype=int)
    for (no, nb), cnt in counts.items():
        r = all_nb.index(nb)
        c = all_no.index(no)
        matrix[r, c] = cnt

    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(all_no)))
    ax.set_xticklabels(all_no, fontsize=8)
    ax.set_yticks(range(len(all_nb)))
    ax.set_yticklabels(all_nb, fontsize=8)
    ax.set_title("(Num Objects, Num Blickets) Heatmap", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Number of Objects", fontsize=9)
    ax.set_ylabel("Number of Blickets", fontsize=9)

    # Annotate cells
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if val > 0:
                ax.text(c, r, str(val), ha="center", va="center",
                        fontsize=8, color="white" if val > matrix.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Count")


def plot_blicket_rule_grouped(ax, counts: Counter) -> None:
    """Grouped bar chart: num_blickets × rule_type."""
    rule_types = sorted({k[1] for k in counts})
    num_blickets = sorted({k[0] for k in counts})

    x = np.arange(len(num_blickets))
    width = 0.35
    offsets = np.linspace(-width / 2 * (len(rule_types) - 1),
                           width / 2 * (len(rule_types) - 1),
                           len(rule_types))

    for rt, offset in zip(rule_types, offsets):
        values = [counts.get((nb, rt), 0) for nb in num_blickets]
        bars = ax.bar(x + offset, values, width,
                      label=rt, color=PALETTE.get(rt, "#888"),
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(val),
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(num_blickets, fontsize=9)
    ax.set_title("(Num Blickets, Rule Type) Distribution", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Number of Blickets", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(title="Rule Type", fontsize=8, title_fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def visualize(
    rule_counts: Counter,
    obj_counts: Counter,
    obj_blicket_counts: Counter,
    blicket_rule_counts: Counter,
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evaluation Dataset Profile", fontsize=14, fontweight="bold", y=1.01)

    plot_rule_type(axes[0, 0], rule_counts)
    plot_num_objects(axes[0, 1], obj_counts)
    plot_object_blicket_heatmap(axes[1, 0], obj_blicket_counts)
    plot_blicket_rule_grouped(axes[1, 1], blicket_rule_counts)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the BlicketEnv eval dataset from HuggingFace."
    )
    parser.add_argument(
        "--repo",
        default=HF_REPO,
        help=f"HuggingFace dataset repo (default: {HF_REPO})",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        default="environments/BlicketTest_CausalReasoning/datasets_profile/eval_profile.png",
        help="Save figure to FILE instead of displaying it (e.g. profile.png)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_from_hf(repo=args.repo)

    rule_counts        = count_rule_types(examples)
    obj_counts         = count_num_objects(examples)
    obj_blicket_counts = count_object_blicket_combos(examples)
    blicket_rule_counts = count_blicket_rule_combos(examples)

    print_summary(examples)
    print_rule_type_counts(rule_counts)
    print_num_objects_counts(obj_counts)
    print_object_blicket_combos(obj_blicket_counts)
    print_blicket_rule_combos(blicket_rule_counts)
    
    save_path = Path(args.save) if args.save else None
    visualize(rule_counts, obj_counts, obj_blicket_counts, blicket_rule_counts, save_path)


if __name__ == "__main__":
    main()
