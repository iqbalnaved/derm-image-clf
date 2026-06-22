#!/usr/bin/env python3
import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ----------------------------- Helpers -----------------------------

ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def build_stem_index(img_dir: Path, recursive: bool = True):
    """
    Scan img_dir and build a mapping: filename stem -> [full paths].
    This lets us resolve CSV filenames that lack an extension.
    """
    stem_to_paths = defaultdict(list)
    pattern = "**/*" if recursive else "*"
    for p in img_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            stem_to_paths[p.stem].append(p)
    return stem_to_paths

def resolve_src_path(stem_or_name: str, img_dir: Path, stem_index) -> Path | None:
    """
    Given a CSV entry that might be a stem (no extension) or a full filename,
    return a concrete Path on disk if found; else None.
    """
    # If entry already includes extension and exists, return it.
    candidate = img_dir / stem_or_name
    if candidate.is_file():
        return candidate

    # Otherwise, try to resolve by stem.
    stem = Path(stem_or_name).stem
    matches = stem_index.get(stem, [])
    if not matches:
        return None
    if len(matches) > 1:
        # Prefer jpg/jpeg if multiple matches exist; else deterministic first.
        jpgs = [m for m in matches if m.suffix.lower() in [".jpg", ".jpeg"]]
        return (jpgs[0] if jpgs else sorted(matches)[0])
    return matches[0]

def load_split(csv_path: Path, img_dir: Path, stem_index=None):
    """
    Return a DataFrame with columns:
      filename (str), label_name (str), src_path (Path)
    Assumes CSV: first col = filename (may be stem), remaining cols = one-hot class columns.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} must have at least 2 columns (filename + at least one class)")

    filename_col = df.columns[0]
    class_cols = list(df.columns[1:])

    class_vals = df[class_cols].values
    argmax_idx = class_vals.argmax(axis=1)
    row_sums = class_vals.sum(axis=1)

    # Keep rows with at least one positive (avoid all-zero one-hot)
    valid_mask = row_sums >= 1
    if (~valid_mask).any():
        print(f"[WARN] {(~valid_mask).sum()} rows in {csv_path} have no positive class (all zeros). Skipped.")
    df_valid = df[valid_mask].copy()

    labels = [class_cols[i] for i in argmax_idx[valid_mask]]
    out = pd.DataFrame({
        "filename": df_valid[filename_col].astype(str).values,
        "label_name": labels
    })

    # Build index if not provided
    if stem_index is None:
        stem_index = build_stem_index(img_dir, recursive=True)

    resolved = []
    missing = 0
    for name in out["filename"]:
        p = resolve_src_path(name, img_dir, stem_index)
        if p is None:
            missing += 1
        resolved.append(p)

    if missing:
        print(f"[WARN] {missing} files from {csv_path} could not be resolved on disk (likely missing or wrong names).")

    out["src_path"] = resolved
    out = out[out["src_path"].notnull()].reset_index(drop=True)
    return out, class_cols

def sample_balanced(df: pd.DataFrame, classes_to_use, n_samples, seed):
    """
    Round-robin (balanced) sampling across classes_to_use.
    Returns up to n_samples rows, skipping classes that exhaust early.
    """
    by_class = {c: df.index[df["label_name"] == c].tolist() for c in classes_to_use}
    for c in classes_to_use:
        random.Random(seed).shuffle(by_class[c])

    picked = []
    exhausted = set()
    i = 0
    while len(picked) < n_samples and len(exhausted) < len(classes_to_use):
        c = classes_to_use[i % len(classes_to_use)]
        if c in exhausted:
            i += 1
            continue
        if by_class[c]:
            picked.append(by_class[c].pop())
        else:
            exhausted.add(c)
        i += 1

    return df.loc[picked].copy()

def ensure_images_copied(rows: pd.DataFrame, dest_images_dir: Path, split_name: str):
    """
    Copy images to dest_images_dir, naming them: <split>__<original_name>.
    Keeps the original file extension on disk.
    Returns the list of copied filenames (as stored on disk).
    """
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    copied_names = []
    for _, r in rows.iterrows():
        src = Path(r["src_path"])
        if not src.is_file():
            print(f"[WARN] Missing file on disk: {src}")
            continue
        new_name = f"{split_name}__{src.name}"
        dst = dest_images_dir / new_name
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] Failed to copy {src} -> {dst}: {e}")
            continue
        copied_names.append(new_name)
    return copied_names

# ----------------------------- Main -----------------------------

def main():
    random.seed(SEED)

    # Build stem indices once per split (fast reuse)
    train_index = build_stem_index(TRAIN_DIR, recursive=True)
    val_index   = build_stem_index(VAL_DIR,   recursive=True)
    test_index  = build_stem_index(TEST_DIR,  recursive=True)

    # Load splits (resolve missing extensions)
    train_df, train_cols = load_split(TRAIN_CSV, TRAIN_DIR, train_index)
    val_df,   val_cols   = load_split(VAL_CSV,   VAL_DIR,   val_index)
    test_df,  test_cols  = load_split(TEST_CSV,  TEST_DIR,  test_index)

    # Validate positive class membership
    if POSITIVE_CLASS not in CLASSES:
        print(f"[ERROR] POSITIVE_CLASS='{POSITIVE_CLASS}' must be one of CLASSES: {CLASSES}")
        sys.exit(1)

    # Filter to requested classes
    train_df = train_df[train_df["label_name"].isin(CLASSES)].reset_index(drop=True)
    val_df   = val_df[val_df["label_name"].isin(CLASSES)].reset_index(drop=True)
    test_df  = test_df[test_df["label_name"].isin(CLASSES)].reset_index(drop=True)

    # Inform availability
    def available_str(df):
        counts = df["label_name"].value_counts().to_dict()
        return ", ".join([f"{k}:{v}" for k, v in counts.items()]) if counts else "none"
    print(f"[INFO] Available TRAIN per class: {available_str(train_df)}")
    print(f"[INFO] Available VAL   per class: {available_str(val_df)}")
    print(f"[INFO] Available TEST  per class: {available_str(test_df)}")

    # Round-robin sampling
    train_s = sample_balanced(train_df, CLASSES, N_TRAIN, SEED)
    val_s   = sample_balanced(val_df,   CLASSES, N_VAL,   SEED + 1)
    test_s  = sample_balanced(test_df,  CLASSES, N_TEST,  SEED + 2)

    # Binary labels (1 for POSITIVE_CLASS, 0 for others in CLASSES)
    def set_binary_label(df):
        return (df["label_name"] == POSITIVE_CLASS).astype(int)

    train_s["binary_label"] = set_binary_label(train_s)
    val_s["binary_label"]   = set_binary_label(val_s)
    test_s["binary_label"]  = set_binary_label(test_s)

    train_s["split"] = "train"
    val_s["split"]   = "val"
    test_s["split"]  = "test"

    # Copy files
    dest_images = DEST_ROOT / "images"
    copied_train = ensure_images_copied(train_s, dest_images, "train")
    copied_val   = ensure_images_copied(val_s,   dest_images, "val")
    copied_test  = ensure_images_copied(test_s,  dest_images, "test")

    # Build splits.csv (CSV-only normalization to .jpg in the image column)
    out_rows = []
    for df, copied in [(train_s, copied_train), (val_s, copied_val), (test_s, copied_test)]:
        for (_, r), fname in zip(df.iterrows(), copied):
            base = Path(fname).stem
            out_rows.append({
                "image": f"{base}.jpg",                  # <-- CSV shows .jpg
                "binary_label": int(r["binary_label"]),  # 0/1
                "split": r["split"]                      # train/val/test
            })

    out_df = pd.DataFrame(out_rows, columns=["image", "binary_label", "split"])
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(DEST_ROOT / "splits.csv", index=False)

    print(f"[DONE] Copied: train={len(copied_train)}, val={len(copied_val)}, test={len(copied_test)} → {dest_images}")
    print(f"[DONE] Wrote splits CSV → {DEST_ROOT / 'splits.csv'}")
    print("[NOTE] Physical files keep their real extensions; only CSV forces .jpg in the 'image' column.")

# ----------------------------- CONFIG -----------------------------
# Edit these to match your setup

# TRAIN_DIR   = Path("/path/to/train/images")
# VAL_DIR     = Path("/path/to/val/images")
# TEST_DIR    = Path("/path/to/test/images")

# TRAIN_CSV   = Path("/path/to/train.csv")
# VAL_CSV     = Path("/path/to/val.csv")
# TEST_CSV    = Path("/path/to/test.csv")

# CLASSES         = ["benign", "malignant"]   # classes to include
# POSITIVE_CLASS  = "malignant"               # which class is positive (binary_label=1)

# N_TRAIN = 200   # total number sampled for TRAIN (balanced via round-robin)
# N_VAL   = 50    # total for VAL
# N_TEST  = 50    # total for TEST

# DEST_ROOT = Path("/path/to/output_dir")     # will create DEST_ROOT/images/ and DEST_ROOT/splits.csv
# SEED = 42

# ====================== CONFIG ======================
TRAIN_DIR   = Path("/mnt/d/Naved/Data/HAM10000/dataverse_files/HAM10000_images")
VAL_DIR     = Path("/mnt/d/Naved/Data/HAM10000/dataverse_files/HAM10000_images")
TEST_DIR    = Path("/mnt/d/Naved/Data/HAM10000/dataverse_files/ISIC2018_Task3_Test_Images")

TRAIN_CSV   = Path("/mnt/d/Naved/Data/HAM10000/dataverse_files/ham10k_metadata_onehot.csv")
VAL_CSV     = Path("/mnt/d/Naved/Data/HAM10000/dataverse_files/ham10k_metadata_onehot.csv")
TEST_CSV    = Path("/mnt/d/Naved/Data/HAM10000/dataverse_files/ham10k_testset_onehot.csv")

CLASSES         = ["mel", "nv"]   # classes to include
POSITIVE_CLASS  = "mel"               # which one is positive (binary_label=1)

N_TRAIN = 2000
N_VAL   = 200
N_TEST  = 300

DEST_ROOT = Path("/mnt/d/Naved/Data/HAM2500/")
SEED = 42
# ====================================================

if __name__ == "__main__":
    main()

