"""
LogiEdge Dataset Generator (Task D1)
Generates a labelled training dataset by simulating sensor data in each mode:
  - Class 0 (Normal):   --anomaly none       20 minutes  ~120 windows
  - Class 1 (Warning):  --anomaly temp_drift  15 minutes  ~90 windows
  - Class 2 (Critical): --anomaly combined    15 minutes  ~90 windows

Runs the simulator internally (no MQTT needed) and extracts features offline.
Saves dataset as .npy files for training.

Usage:
    python generate_dataset.py
"""

import os
import sys
import numpy as np

# Add parent directory to path so we can import from data_pipeline
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from data_pipeline.preprocessing import (
    offline_extract_features, save_training_stats, FILTER_SIZE
)
from data_pipeline.simulator import (
    generate_temperature, generate_vibration,
    TEMP_SETPOINT, TEMP_NORMAL_STD, TEMP_DRIFT_RATE,
    VIB_NORMAL_MEAN, VIB_NORMAL_STD, VIB_ANOMALY_MEAN, VIB_ANOMALY_STD
)

# Dataset parameters from the assignment
DATASET_CONFIG = [
    {"class_id": 0, "label": "Normal",   "anomaly": "none",       "duration": 1200},  # 20 min
    {"class_id": 1, "label": "Warning",  "anomaly": "temp_drift", "duration": 900},   # 15 min
    {"class_id": 2, "label": "Critical", "anomaly": "combined",   "duration": 900},   # 15 min
]

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")
STATS_FILE = os.path.join(PROJECT_DIR, "data_pipeline", "training_stats.npy")


def simulate_sensors(anomaly_mode, duration):
    """
    Run sensor simulation internally without MQTT.
    Returns lists of temperature and vibration readings.
    """
    temp_readings = []
    vib_readings = []

    for tick in range(duration):
        # Temperature at 1 Hz
        temp = generate_temperature(tick, anomaly_mode)
        temp_readings.append(temp)

        # Vibration at 0.5 Hz (every 2 seconds)
        if tick % 2 == 0:
            vib = generate_vibration(anomaly_mode)
            vib_readings.append(vib)

    return temp_readings, vib_readings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_features = []
    all_labels = []
    normal_features = []  # For computing training stats

    print("=" * 60)
    print("LogiEdge Dataset Generator")
    print("=" * 60)

    for config in DATASET_CONFIG:
        class_id = config["class_id"]
        label = config["label"]
        anomaly = config["anomaly"]
        duration = config["duration"]

        print(f"\n--- Generating Class {class_id} ({label}) ---")
        print(f"    Anomaly mode: {anomaly}")
        print(f"    Duration: {duration}s ({duration // 60} minutes)")

        # Simulate sensor data
        temp_readings, vib_readings = simulate_sensors(anomaly, duration)
        print(f"    Temperature readings: {len(temp_readings)}")
        print(f"    Vibration readings: {len(vib_readings)}")

        # Extract features
        features = offline_extract_features(temp_readings, vib_readings)
        print(f"    Feature windows extracted: {len(features)}")

        if len(features) == 0:
            print(f"    [WARNING] No features extracted for class {class_id}!")
            continue

        # Store
        for feat in features:
            all_features.append(feat)
            all_labels.append(class_id)

        # Save Normal-class features for training stats (first 10 min only per spec)
        if class_id == 0:
            # Spec: "compute mean and std from 10 minutes of clean Normal-class output"
            # 10 minutes at step=10s gives ~58 windows (600s - 30s window) / 10s step + 1
            ten_min_windows = int((600 - 30) / 10) + 1  # 58 windows from 10 min
            normal_features.extend(features[:ten_min_windows])

    # Convert to arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n{'=' * 60}")
    print(f"Dataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X.shape}")
    for config in DATASET_CONFIG:
        count = np.sum(y == config["class_id"])
        print(f"  Class {config['class_id']} ({config['label']}): {count} samples")
    print(f"{'=' * 60}")

    # Compute and save training stats from Normal-class data
    print(f"\nComputing normalisation stats from {len(normal_features)} Normal-class windows...")
    mean, std = save_training_stats(normal_features, STATS_FILE)

    # Normalise the entire dataset using the computed stats
    std_safe = np.where(std == 0, 1.0, std)
    X_normalised = (X - mean) / std_safe

    # Save dataset
    dataset_path = os.path.join(OUTPUT_DIR, "dataset.npz")
    np.savez(dataset_path,
             X=X, y=y,
             X_normalised=X_normalised,
             mean=mean, std=std)
    print(f"\nDataset saved to {dataset_path}")

    # Print feature statistics per class
    print(f"\nFeature Statistics (unnormalised):")
    feature_names = ["temp_mean", "temp_std", "temp_roc", "vib_rms", "vib_peak", "vib_kurtosis"]
    for config in DATASET_CONFIG:
        mask = y == config["class_id"]
        class_features = X[mask]
        print(f"\n  Class {config['class_id']} ({config['label']}):")
        for j, name in enumerate(feature_names):
            vals = class_features[:, j]
            print(f"    {name:>15s}: mean={np.mean(vals):8.4f}  std={np.std(vals):8.4f}  "
                  f"min={np.min(vals):8.4f}  max={np.max(vals):8.4f}")


if __name__ == "__main__":
    main()
