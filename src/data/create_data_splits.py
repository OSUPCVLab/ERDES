"""
Create stratified train/val/test CSVs for any number of binary-classification
tasks defined in TASKS.

    python data_split.py --root data/Dataset
    python data_split.py --root data/Dataset --tasks non_rd_vs_rd
"""
from __future__ import print_function

import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Configure tasks here:  {task_name: {int_label: [relative dirs]}}
# ----------------------------------------------------------------------
TASKS = {
    "non_rd_vs_rd": {
        0: ["Non_RD"],                       # Non_RD  → 0
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

def collect_files(root, rel_dirs, label):
    """Return list of (file_path, label) for all .mp4 files under rel_dirs."""
    rows = []
    for rel_dir in rel_dirs:
        pattern = os.path.join(root, rel_dir, "*.mp4")
        rows.extend([(f, label) for f in glob.glob(pattern)])
    return rows


def split_and_save(df, task_name):
    """Stratified split (72 / 8 / 20) and write three CSVs."""
    train_val, test = train_test_split(
    df,
    test_size=0.20,    
    stratify=df["label"],
    random_state=42,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.10,
        stratify=train_val["label"],
        random_state=42,
    )

    splits_dir = os.path.join("data", "splits")
    out_dir = os.path.join(splits_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)

    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"[✓] {task_name:25s} → {len(train):4d} train / {len(val):4d} val / {len(test):4d} test (saved to ERDES/data/splits/{task_name})")

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
        split_and_save(df, task)


if __name__ == "__main__":
    main()
