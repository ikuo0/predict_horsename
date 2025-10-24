import json
import os
import re

import numpy as np

# ファイル名 feat_00000_xdata_アアストシヤーター.npy, feat_399352_xdata_ヴードゥーレディ.npy 5桁又は6桁の数値部分と末尾の馬名が可変
RE_X_FILENAME = re.compile(r"feat_(\d{5,6})_xdata_(.+)\.npy")

# ファイル名 feat_00000_ydata_アアストシヤーター.npy, feat_399352_ydata_ヴードゥーレディ.npy 5桁又は6桁の数値部分と末尾の馬名が可変
RE_Y_FILENAME = re.compile(r"feat_(\d{5,6})_ydata_(.+)\.npy")

FEATURE_DIR = "futurize/futurize_data"
OUT_DIR = "./feature_assemble/feature_assemble_data"

def enum_feature_files(feature_dir: str):
    """Enumerate all feature files in the given directory."""
    feature_files = []
    for root, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith(".npy"):
                feature_files.append(os.path.join(root, file))
    return feature_files

def extract_xdata_filenames(files: list[str]) -> list[str]:
    result = []
    for filename in files:
        match = RE_X_FILENAME.match(os.path.basename(filename))
        if match:
            result.append(filename)
    return result

def extract_ydata_filenames(files: list[str]) -> list[str]:
    result = []
    for filename in files:
        match = RE_Y_FILENAME.match(os.path.basename(filename))
        if match:
            result.append(filename)
    return result

def load_and_concatenate_features(x_filenames: list[str], y_filenames: list[str]):
    x_data_list = []
    y_data_list = []
    size = len(x_filenames)

    for i, x_file in enumerate(x_filenames):
        print(f"Loading x data from {x_file}, ({i}/{size})")
        x_data = np.load(x_file)
        x_data_list.append(x_data)

    for i, y_file in enumerate(y_filenames):
        print(f"Loading y data from {y_file}, ({i}/{size})")
        y_data = np.load(y_file)
        y_data_list.append(y_data)

    if not x_data_list or not y_data_list:
        raise ValueError("No feature files found.")

    # Concatenate along the last axis
    x_data_combined = np.concatenate(x_data_list, axis=0)
    y_data_combined = np.concatenate(y_data_list, axis=0)

    # feature report
    print(f"Combined x data shape: {x_data_combined.shape}")
    print(f"Combined y data shape: {y_data_combined.shape}")
    report_dict = {
        "x_data": {
            "shape": x_data_combined.shape,
            "dtype": str(x_data_combined.dtype),
            "min": float(np.min(x_data_combined)),
            "max": float(np.max(x_data_combined)),
            "mean": float(np.mean(x_data_combined)),
            "std": float(np.std(x_data_combined)),
        },
        "y_data": {
            "shape": y_data_combined.shape,
            "dtype": str(y_data_combined.dtype),
            "min": float(np.min(y_data_combined)),
            "max": float(np.max(y_data_combined)),
            "mean": float(np.mean(y_data_combined)),
            "std": float(np.std(y_data_combined)),
        },
    }
    report_filename = os.path.join(OUT_DIR, "feature_report.json")
    with open(report_filename, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Feature report saved to {report_filename}")

    x_filename = os.path.join(OUT_DIR, "feat_all_xdata.npz")
    y_filename = os.path.join(OUT_DIR, "feat_all_ydata.npz")
    print(f"Saving combined features to {x_filename} and {y_filename}")
    np.savez_compressed(x_filename, x_data_combined)
    np.savez_compressed(y_filename, y_data_combined)
    print(f"Saved combined x data to {x_filename}")
    print(f"Saved combined y data to {y_filename}")


def main():
    all_files = enum_feature_files(FEATURE_DIR)
    x_filenames = extract_xdata_filenames(all_files)
    y_filenames = extract_ydata_filenames(all_files)

    # # debug
    # x_filenames = x_filenames[:100]
    # y_filenames = y_filenames[:100]

    load_and_concatenate_features(x_filenames, y_filenames)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()

# python feature_assemble/feature_assemble.py
