"""
Create stratified train/val/test CSVs for any number of binary-classification
tasks defined in TASKS.

    python data_split.py --root data/Dataset
    python data_split.py --root data/Dataset --tasks normal_vs_rd
"""
from __future__ import print_function

import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1.  Configure tasks here:  {task_name: {int_label: [relative dirs]}}
# ----------------------------------------------------------------------
TASKS = {
    "normal_vs_rd": {
        0: ["Normal"],                       # Normal  → 0
        1: [                                 # RD      → 1
            "Macula_Detached/Bilateral",
            "Macula_Detached/TD",
            "Macula_Intact/ND",
            "Macula_Intact/TD",
        ],
    },
    "macula_detached_vs_intact": {
        0: [                                 # Macula Detached → 0
            "Macula_Detached/Bilateral",
            "Macula_Detached/TD",
        ],
        1: [                                 # Macula Intact   → 1
            "Macula_Intact/ND",
            "Macula_Intact/TD",
        ],
    },
}

# ----------------------------------------------------------------------
# 2.  Helpers
# ----------------------------------------------------------------------
def collect_files(root, rel_dirs, label):
    """Return list of (file_path, label) for all .mp4 files under rel_dirs."""
    rows = []
    for rel_dir in rel_dirs:
        pattern = os.path.join(root, rel_dir, "*.mp4")
        rows.extend([(f, label) for f in glob.glob(pattern)])
    return rows


def split_and_save(df, out_prefix):
    """Stratified split (72 / 22.4 / 5.6) and write three CSVs."""
    train, temp = train_test_split(
        df, test_size=0.28, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.20, stratify=temp["label"], random_state=42
    )

    train.to_csv("{}_train.csv".format(out_prefix), index=False)
    val.to_csv("{}_val.csv".format(out_prefix), index=False)
    test.to_csv("{}_test.csv".format(out_prefix), index=False)

    print("[✓] {}  →  {} train / {} val / {} test".format(
        out_prefix, len(train), len(val), len(test))
    )

# ----------------------------------------------------------------------
# 3.  Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/Dataset",
                        help="Dataset root directory")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()),
                        default=list(TASKS.keys()),
                        help="Which tasks to split (default: all)")
    args = parser.parse_args()

    for task in args.tasks:
        cfg = TASKS[task]
        rows = []
        for label_int, dirs in cfg.items():
            rows.extend(collect_files(args.root, dirs, label_int))

        if not rows:
            print("[!] No videos found for task '{}'. Skipping.".format(task))
            continue

        df = pd.DataFrame(rows, columns=["path", "label"])
        split_and_save(df, os.path.join("data", task))


if __name__ == "__main__":
    main()
