#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Profile the training dataset by regenerating full_train_pool with the same
parameters used in load_environment() / build_new_eval_dataset.py:

  sample_balanced_configs(
      num_objects_range=(4, 10),
      n_conjunctive=333,
      n_disjunctive=167,
      blicket_range_fn=original_blicket_range,   # [2, n//2]
      exclude_keys=set(),
      seed=42,
  )

Each element is a dict with keys: n_obj, rule, blickets, blicket_indices.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Pool generation  (copied from build_new_eval_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────
NUM_EXAMPLES = 250
MAX_TRAIN = 500
MAX_N_CONJ = round(2 * MAX_TRAIN / 3)   # 333
MAX_N_DISJ = MAX_TRAIN - MAX_N_CONJ      # 167

def original_blicket_range(n_obj: int) -> tuple[int, int]:
    return 2, n_obj // 2


def sample_balanced_configs(
    num_objects_range: tuple[int, int],
    n_conjunctive: int,
    n_disjunctive: int,
    blicket_range_fn,
    exclude_keys: set[tuple],
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    lo, hi = num_objects_range
    seen: set[tuple] = set(exclude_keys)
    conjunctive: list[dict] = []
    disjunctive: list[dict] = []

    while len(conjunctive) < n_conjunctive or len(disjunctive) < n_disjunctive:
        n_obj = int(rng.integers(lo, hi + 1))
        min_b, max_b = blicket_range_fn(n_obj)
        if min_b > max_b:
            continue
        b = int(rng.integers(min_b, max_b + 1))
        rule = str(rng.choice(["disjunctive", "conjunctive"]))
        blicket_indices = tuple(sorted(
            rng.choice(n_obj, size=b, replace=False).tolist()
        ))
        key = (n_obj, rule, blicket_indices)
        if key in seen:
            continue
        if rule == "conjunctive" and len(conjunctive) >= n_conjunctive:
            continue
        if rule == "disjunctive" and len(disjunctive) >= n_disjunctive:
            continue
        seen.add(key)
        blickets = [0] * n_obj
        for idx in blicket_indices:
            blickets[idx] = 1
        cfg = {
            "n_obj": n_obj,
            "rule": rule,
            "blickets": blickets,
            "blicket_indices": list(blicket_indices),
        }
        if rule == "conjunctive":
            conjunctive.append(cfg)
        else:
            disjunctive.append(cfg)

    return conjunctive + disjunctive


def build_train_pool() -> list[dict]:
    MAX_TRAIN  = 500
    N_CONJ     = round(2 * MAX_TRAIN / 3)   # 333
    N_DISJ     = MAX_TRAIN - N_CONJ          # 167
    return sample_balanced_configs(
        num_objects_range=(4, 10),
        n_conjunctive=N_CONJ,
        n_disjunctive=N_DISJ,
        blicket_range_fn=original_blicket_range,
        exclude_keys=set(),
        seed=42,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Counting helpers  (adapted for pool field names)
# ──────────────────────────────────────────────────────────────────────────────

def count_rule_types(pool: list[dict]) -> Counter:
    return Counter(c["rule"] for c in pool)


def count_num_objects(pool: list[dict]) -> Counter:
    return Counter(c["n_obj"] for c in pool)


def count_object_blicket_combos(pool: list[dict]) -> Counter:
    return Counter((c["n_obj"], len(c["blicket_indices"])) for c in pool)


def count_blicket_rule_combos(pool: list[dict]) -> Counter:
    return Counter((len(c["blicket_indices"]), c["rule"]) for c in pool)


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


def print_summary(pool: list[dict]) -> None:
    print_header("TRAINING POOL SUMMARY")
    print(f"  Total examples  : {len(pool)}")

    rt = count_rule_types(pool)
    print(f"  Rule types      : {dict(rt)}")

    no = count_num_objects(pool)
    print(f"  Num-objects range  : {min(no)} – {max(no)}")

    nb = Counter(len(c["blicket_indices"]) for c in pool)
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
            bar.get_height() + 0.5,
            str(val),
            ha="center", va="bottom", fontsize=8,
        )


def plot_rule_type(ax, counts: Counter) -> None:
    labels = sorted(counts)
    _bar(ax, labels, [counts[l] for l in labels],
         "Rule Type Distribution", "Rule Type", "Count")
    for bar, lbl in zip(ax.patches, labels):
        bar.set_color(PALETTE.get(lbl, PALETTE["bar"]))


def plot_num_objects(ax, counts: Counter) -> None:
    labels = sorted(counts)
    _bar(ax, labels, [counts[l] for l in labels],
         "Num-Objects Distribution", "Number of Objects", "Count")


def plot_object_blicket_heatmap(ax, counts: Counter) -> None:
    all_no = sorted({k[0] for k in counts})
    all_nb = sorted({k[1] for k in counts})
    matrix = np.zeros((len(all_nb), len(all_no)), dtype=int)
    for (no, nb), cnt in counts.items():
        matrix[all_nb.index(nb), all_no.index(no)] = cnt

    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(all_no)))
    ax.set_xticklabels(all_no, fontsize=8)
    ax.set_yticks(range(len(all_nb)))
    ax.set_yticklabels(all_nb, fontsize=8)
    ax.set_title("(Num Objects, Num Blickets) Heatmap", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Number of Objects", fontsize=9)
    ax.set_ylabel("Number of Blickets", fontsize=9)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if val > 0:
                ax.text(c, r, str(val), ha="center", va="center",
                        fontsize=8, color="white" if val > matrix.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Count")


def plot_blicket_rule_grouped(ax, counts: Counter) -> None:
    rule_types   = sorted({k[1] for k in counts})
    num_blickets = sorted({k[0] for k in counts})
    x      = np.arange(len(num_blickets))
    width  = 0.35
    offsets = np.linspace(
        -width / 2 * (len(rule_types) - 1),
         width / 2 * (len(rule_types) - 1),
        len(rule_types),
    )
    for rt, offset in zip(rule_types, offsets):
        values = [counts.get((nb, rt), 0) for nb in num_blickets]
        bars = ax.bar(x + offset, values, width,
                      label=rt, color=PALETTE.get(rt, "#888"),
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
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
    fig.suptitle("Training Pool Profile  (seed=42, n_obj∈[4,10], blickets∈[2, n//2])",
                 fontsize=13, fontweight="bold", y=1.01)

    plot_rule_type(axes[0, 0], rule_counts)
    plot_num_objects(axes[0, 1], obj_counts)
    plot_object_blicket_heatmap(axes[1, 0], obj_blicket_counts)
    plot_blicket_rule_grouped(axes[1, 1], blicket_rule_counts)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the BlicketTest training pool (regenerated from seed=42)."
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        default="datasets_profile/train_profile.png",
        help="Save figure to FILE (default: datasets_profile/train_profile.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating training pool (seed=42, n_obj∈[4,10], blickets∈[2,n//2])...")
    pool = build_train_pool()

    n_conj = round(2 * NUM_EXAMPLES / 3)
    n_disj = NUM_EXAMPLES - n_conj
    train_pool = pool[:n_conj] + pool[MAX_N_CONJ:MAX_N_CONJ + n_disj]
    pool = train_pool
    print(f"  Pool size: {len(pool)}")

    rule_counts         = count_rule_types(pool)
    obj_counts          = count_num_objects(pool)
    obj_blicket_counts  = count_object_blicket_combos(pool)
    blicket_rule_counts = count_blicket_rule_combos(pool)

    print_summary(pool)
    print_rule_type_counts(rule_counts)
    print_num_objects_counts(obj_counts)
    print_object_blicket_combos(obj_blicket_counts)
    print_blicket_rule_combos(blicket_rule_counts)

    save_path = Path(args.save) if args.save else None
    visualize(rule_counts, obj_counts, obj_blicket_counts, blicket_rule_counts, save_path)


if __name__ == "__main__":
    main()
