# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "datasets"]
# ///
"""Build a new, fully disjoint eval dataset with:
  - 100 examples: 62 conjunctive + 38 disjunctive
  - num_objects ∈ [8, 15]
  - num_blickets ∈ [5, floor(0.7 * num_objects)]
  - Distinct from the full training pool (seed=42, n∈[4,10]) and both
    existing eval splits (seed=100, n∈[4,10] and n∈[11,15])

build_rows() is copied verbatim from BlicketTest_CausalReasoning.py to compute
optimal step counts and assemble the final prompt rows, then saves the dataset
to environments/BlicketTest_CausalReasoning/datasets/new_eval/.
"""

import hashlib
import json
import math
import os
from itertools import product

import numpy as np
from datasets import Dataset


# ── Copied verbatim from BlicketTest_CausalReasoning.py ──────────────────────

def build_system_prompt(num_objects: int, max_num_steps: int) -> str:
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\

You are an intelligent, curious agent. You are playing a game where you are in a room with \
{num_objects} different objects, and a machine. The objects are labeled as such: {object_list}. Some of these objects are blickets. \
You can't tell which object is a blicket just by looking at it. \
Blickets make the machine turn on following some hidden rule. \
This hidden rule may require all blickets to be on the machine for it turn on. \
Or, the hidden rule may require any of the the blickets to be on the machine for it to turn on.

Your goal is to determine exactly which objects are Blickets through exploration.
You have a maximum of {max_num_steps} steps to conduct the exploration phase so you must act efficiently. You can also exit this phase early if you think you understand the relationship between the
objects and the machine. After the exploration phase is done, you will be asked to list the blickets.

RULES:
- In each action, you can place exactly one object onto the machine or remove exactly one object off the machine.
- After each action, you will observe which objects are on the machine and whether the machine is ON or OFF.
- When you have gathered enough information to determine which objects are Blickets, you can exit the exploration phase to submit your answer.

RESPONSE FORMAT (strict — violations count as wasted steps):

Every response must contain exactly one <reasoning> block followed by exactly one <action> block. No other XML tags are permitted. The <action> tag must not appear inside the <reasoning> block.

During exploration, your response must be one of these three forms:

  Place an object on the machine:
    <reasoning>Your reasoning here.</reasoning>
    <action>put N on</action>

  Remove an object from the machine:
    <reasoning>Your reasoning here.</reasoning>
    <action>put N off</action>

  Exit exploration and move to the answer phase:
    <reasoning>Your reasoning for stopping here.</reasoning>
    <action>exit</action>

Where N is a single integer in ({object_list}). Do not include any text outside the XML tags.

During the answer phase, your response must be exactly:
    <reasoning>Your analysis of which objects are Blickets.</reasoning>
    <action>{{1, 3}}</action>

List the IDs of every object you believe is a Blicket inside curly braces, separated by commas. \
If you believe none of the objects are Blickets, use <action>{{}}</action>.

STRATEGY: Plan your experiments carefully to gather maximum information efficiently since you are limited by the number of actions you can take. \
Reason about what actions will give you the most information and what each observation tells you about the hidden rule and which objects might be Blickets."""


def build_initial_message(num_objects: int) -> str:
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\
You are in front of a Blicket-detecting machine with {num_objects} objects: {object_list}.
Currently, no objects are on the machine. The machine is OFF. Your task is to determine which objects \
are blickets.

Begin"""


def compute_optimal_steps(
    num_objects: int,
    blickets: list[int],
    rule_type: str,
    num_samples: int = 10,
    seed: int = 0,
) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    run_seeds = rng.integers(0, 2**31, size=num_samples)

    def predict(obj_states, blicket_bits, rule):
        active = [obj_states[i] for i in range(num_objects) if blicket_bits[i]]
        return int(any(active)) if rule == "disjunctive" else int(all(active))

    all_hypotheses = [
        (bits, rule)
        for bits in product((0, 1), repeat=num_objects)
        for rule in ("disjunctive", "conjunctive")
    ]

    init_state = tuple([0] * num_objects)
    init_machine = predict(init_state, blickets, rule_type)
    init_active = [
        (bits, rule) for bits, rule in all_hypotheses
        if predict(init_state, bits, rule) == init_machine
    ]

    max_iters = 2 ** (num_objects + 1)
    total_steps = 0

    for rs in run_seeds:
        run_rng = np.random.default_rng(int(rs))
        obj_states = list(init_state)
        active = list(init_active)
        visited = {init_state}
        steps = 0

        while steps < max_iters and len(active) > 1:
            candidates = []
            for i in range(num_objects):
                trial = list(obj_states)
                trial[i] = 1 - trial[i]
                on_count = sum(
                    1 for bits, rule in active if predict(trial, bits, rule) == 1
                )
                bal = min(on_count, len(active) - on_count)
                is_new = tuple(trial) not in visited
                candidates.append((i, bal, is_new))

            best_bal = max(b for _, b, _ in candidates)
            tied = [(i, is_new) for i, b, is_new in candidates if b == best_bal]

            unseen = [i for i, is_new in tied if is_new]
            if unseen:
                action = int(run_rng.choice(unseen))
            else:
                action = int(run_rng.choice([i for i, _ in tied]))

            obj_states[action] = 1 - obj_states[action]
            visited.add(tuple(obj_states))
            actual = predict(obj_states, blickets, rule_type)
            active = [
                (b, r) for b, r in active if predict(obj_states, b, r) == actual
            ]
            steps += 1

        total_steps += steps

    return total_steps / num_samples, len(all_hypotheses) - 1


def build_rows(configs: list[dict]) -> tuple[list[dict], int]:
    rows = []
    global_max_steps = 0
    for cfg in configs:
        seed_str = f"{cfg['n_obj']}_{cfg['rule']}_{cfg['blicket_indices']}"
        row_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        optimal_steps, hyps_elim = compute_optimal_steps(
            cfg["n_obj"], cfg["blickets"], cfg["rule"], num_samples=3, seed=row_seed
        )
        max_steps = max(1, math.ceil(1.5 * optimal_steps))
        global_max_steps = max(global_max_steps, max_steps)

        system_msg = {"role": "system", "content": build_system_prompt(cfg["n_obj"], max_steps)}
        user_msg = {"role": "user", "content": build_initial_message(cfg["n_obj"])}

        rows.append({
            "prompt": [system_msg, user_msg],
            "info": json.dumps({
                "num_objects": cfg["n_obj"],
                "num_blickets": len(cfg["blicket_indices"]),
                "max_num_steps": max_steps,
                "rule_type": cfg["rule"],
                "blickets": cfg["blickets"],
                "optimal_hypotheses_eliminated": hyps_elim,
            }),
        })
    return rows, global_max_steps

# ── Targets ───────────────────────────────────────────────────────────────────

N_CONJUNCTIVE = 62
N_DISJUNCTIVE = 38
SEED = 200  # distinct from training (42) and existing eval (100)

# ── Reproduce exclusion sets from training + existing eval ────────────────────

def sample_balanced_configs(
    num_objects_range: tuple[int, int],
    n_conjunctive: int,
    n_disjunctive: int,
    blicket_range_fn,           # callable(n_obj) -> (min_b, max_b) inclusive
    exclude_keys: set[tuple],
    seed: int,
) -> list[dict]:
    """General balanced config sampler with a pluggable blicket range function."""
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


# Original blicket range used in training and existing eval: [2, floor(n/2)]
def original_blicket_range(n_obj):
    return 2, n_obj // 2

# New blicket range: [5, floor(0.7 * n_obj)]
def new_blicket_range(n_obj):
    return 5, math.floor(0.7 * n_obj)


# --- Build training exclusion pool (same params as load_environment) ---
MAX_TRAIN = 500
MAX_N_CONJ = round(2 * MAX_TRAIN / 3)   # 333
MAX_N_DISJ = MAX_TRAIN - MAX_N_CONJ      # 167

print("Generating training pool for exclusion...")
full_train_pool = sample_balanced_configs(
    num_objects_range=(4, 10),
    n_conjunctive=MAX_N_CONJ,
    n_disjunctive=MAX_N_DISJ,
    blicket_range_fn=original_blicket_range,
    exclude_keys=set(),
    seed=42,
)
train_keys = {(c["n_obj"], c["rule"], tuple(c["blicket_indices"])) for c in full_train_pool}

# --- Build existing eval exclusion sets ---
print("Generating existing eval configs for exclusion...")
existing_eval_4_10 = sample_balanced_configs(
    num_objects_range=(4, 10),
    n_conjunctive=40,
    n_disjunctive=40,
    blicket_range_fn=original_blicket_range,
    exclude_keys=train_keys,
    seed=100,
)
existing_eval_11_15 = sample_balanced_configs(
    num_objects_range=(11, 15),
    n_conjunctive=10,
    n_disjunctive=10,
    blicket_range_fn=original_blicket_range,
    exclude_keys=set(),
    seed=100,
)

all_exclude_keys = (
    train_keys
    | {(c["n_obj"], c["rule"], tuple(c["blicket_indices"])) for c in existing_eval_4_10}
    | {(c["n_obj"], c["rule"], tuple(c["blicket_indices"])) for c in existing_eval_11_15}
)
print(f"  Excluding {len(all_exclude_keys)} existing configs total.")

# ── Sample new eval configs ───────────────────────────────────────────────────

print(f"\nSampling {N_CONJUNCTIVE} conjunctive + {N_DISJUNCTIVE} disjunctive new eval configs...")
new_configs = sample_balanced_configs(
    num_objects_range=(8, 15),
    n_conjunctive=N_CONJUNCTIVE,
    n_disjunctive=N_DISJUNCTIVE,
    blicket_range_fn=new_blicket_range,
    exclude_keys=all_exclude_keys,
    seed=SEED,
)
print(f"  Sampled {len(new_configs)} configs.")

# Sanity check: verify full disjointness
new_keys = {(c["n_obj"], c["rule"], tuple(c["blicket_indices"])) for c in new_configs}
overlap = new_keys & all_exclude_keys
assert len(overlap) == 0, f"Overlap found: {overlap}"
assert len(new_keys) == len(new_configs), "Duplicate configs within new eval set"
print("  Disjointness check passed.")

# ── Build rows via environment's build_rows() ─────────────────────────────────

print("\nBuilding dataset rows (computing optimal steps per config)...")
rows, global_max_steps = build_rows(new_configs)
print(f"  global_max_steps across all configs: {global_max_steps}")

# ── Save dataset ──────────────────────────────────────────────────────────────

out_dir = os.path.join(
    os.path.dirname(__file__),
    "environments", "BlicketTest_CausalReasoning", "datasets", "new_eval"
)
os.makedirs(out_dir, exist_ok=True)

# HuggingFace Dataset
dataset = Dataset.from_list(rows)
dataset.save_to_disk(out_dir)
print(f"\nDataset saved to: {out_dir}")
print(f"  {len(dataset)} rows")

# Also save a JSONL of configs (without prompt text) for inspection
configs_path = os.path.join(out_dir, "configs.jsonl")
with open(configs_path, "w") as f:
    for cfg, row in zip(new_configs, rows):
        info = json.loads(row["info"])
        f.write(json.dumps({
            "n_obj": cfg["n_obj"],
            "n_blickets": len(cfg["blicket_indices"]),
            "blicket_ratio": round(len(cfg["blicket_indices"]) / cfg["n_obj"], 4),
            "rule": cfg["rule"],
            "blickets": cfg["blickets"],
            "blicket_indices": cfg["blicket_indices"],
            "max_num_steps": info["max_num_steps"],
            "optimal_hypotheses_eliminated": info["optimal_hypotheses_eliminated"],
        }) + "\n")
print(f"  Config manifest: {configs_path}")

# ── Quick analysis ────────────────────────────────────────────────────────────

from collections import Counter

print("\n── Dataset Summary ──────────────────────────────────────────────────")
rules = Counter(c["rule"] for c in new_configs)
print(f"\nRule type:")
for r in ("conjunctive", "disjunctive"):
    print(f"  {r:15s}: {rules[r]:3d}  ({100*rules[r]/len(new_configs):.1f}%)")

obj_dist = Counter(c["n_obj"] for c in new_configs)
print(f"\nObject count distribution:")
for n in sorted(obj_dist):
    cnt = obj_dist[n]
    print(f"  n={n:2d}: {cnt:3d}  ({100*cnt/len(new_configs):.1f}%)")

ratios = [len(c["blicket_indices"]) / c["n_obj"] for c in new_configs]
print(f"\nBlicket/object ratio:")
print(f"  mean:   {np.mean(ratios):.3f}")
print(f"  median: {np.median(ratios):.3f}")
print(f"  min:    {np.min(ratios):.3f}  (n_blickets={min(len(c['blicket_indices']) for c in new_configs)})")
print(f"  max:    {np.max(ratios):.3f}  (n_blickets={max(len(c['blicket_indices']) for c in new_configs)})")

max_steps_vals = [json.loads(r["info"])["max_num_steps"] for r in rows]
print(f"\nmax_num_steps (from build_rows):")
print(f"  global max: {global_max_steps}")
print(f"  mean:       {np.mean(max_steps_vals):.1f}")
print(f"  median:     {np.median(max_steps_vals):.1f}")
print(f"  min:        {min(max_steps_vals)}")
print(f"  max:        {max(max_steps_vals)}")
