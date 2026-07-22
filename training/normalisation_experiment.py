"""
Task C2 — Mandatory Normalisation Experiment
Compares model accuracy with correct training_stats vs stats shifted by 3σ.

The assignment requires:
  "run inference with correct stats, then with stats shifted by 3σ;
   report accuracy changes in Phase 2 Report"

Usage:
    python normalisation_experiment.py
"""

import os
import sys
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
STATS_FILE = os.path.join(PROJECT_DIR, "data_pipeline", "training_stats.npy")
DATASET_FILE = os.path.join(SCRIPT_DIR, "models", "dataset.npz")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "model_fp32.tflite")

# Try to import TFLite interpreter
try:
    from ai_edge_litert import interpreter as litert
    TFLiteInterpreter = litert.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
    except ImportError:
        import tflite_runtime.interpreter as tflite
        TFLiteInterpreter = tflite.Interpreter

CLASS_NAMES = ["Normal", "Warning", "Critical"]


def run_inference_with_stats(X_raw, y_true, mean, std, model_path):
    """
    Normalise raw features using given mean/std, then run inference.
    Returns accuracy and per-class recall.
    """
    std_safe = np.where(std == 0, 1.0, std)
    X_norm = ((X_raw - mean) / std_safe).astype(np.float32)

    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    preds = []
    for i in range(len(X_norm)):
        sample = X_norm[i:i+1]
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output)
        preds.append(pred)
        if pred == y_true[i]:
            correct += 1

    accuracy = correct / len(y_true)

    # Per-class recall
    recalls = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y_true == cls_idx)
        if mask.sum() == 0:
            recalls[cls_name] = float('nan')
        else:
            cls_preds = np.array(preds)[mask]
            recalls[cls_name] = (cls_preds == cls_idx).sum() / mask.sum()

    return accuracy, recalls, preds


def main():
    print("=" * 70)
    print("Task C2 — Mandatory Normalisation Experiment")
    print("Comparing correct training_stats vs 3σ-shifted stats")
    print("=" * 70)

    # Load dataset (raw unnormalised features)
    data = np.load(DATASET_FILE)
    X_raw = data["X"].astype(np.float32)
    y = data["y"]

    # Split same way as training
    from sklearn.model_selection import train_test_split
    _, X_val_raw, _, y_val = train_test_split(
        X_raw, y, test_size=0.20, random_state=0, stratify=y
    )

    # Load correct training stats
    stats = np.load(STATS_FILE, allow_pickle=True).item()
    correct_mean = stats['mean']
    correct_std = stats['std']

    print(f"\nDataset: {len(y)} total samples, {len(y_val)} validation samples")
    print(f"\nCorrect training stats:")
    print(f"  Mean: {correct_mean}")
    print(f"  Std:  {correct_std}")

    # ---- Experiment 1: Correct stats ----
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: Inference with CORRECT normalisation stats")
    print("-" * 70)

    acc_correct, recalls_correct, _ = run_inference_with_stats(
        X_val_raw, y_val, correct_mean, correct_std, MODEL_PATH
    )

    print(f"\n  Accuracy: {acc_correct * 100:.2f}%")
    for cls, recall in recalls_correct.items():
        print(f"  {cls} recall: {recall * 100:.2f}%")

    # ---- Experiment 2: Stats shifted by +3σ ----
    shifted_mean = correct_mean + 3 * correct_std
    shifted_std = correct_std  # Keep std the same, only shift mean

    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Inference with 3σ-SHIFTED normalisation stats")
    print("-" * 70)
    print(f"\n  Shifted Mean: {shifted_mean}")
    print(f"  (Original mean + 3 × std)")

    acc_shifted, recalls_shifted, _ = run_inference_with_stats(
        X_val_raw, y_val, shifted_mean, correct_std, MODEL_PATH
    )

    print(f"\n  Accuracy: {acc_shifted * 100:.2f}%")
    for cls, recall in recalls_shifted.items():
        print(f"  {cls} recall: {recall * 100:.2f}%")

    # ---- Experiment 3: Stats shifted by +10σ (severe drift) ----
    shifted_mean_10 = correct_mean + 10 * correct_std

    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Inference with 10σ-SHIFTED mean (severe drift)")
    print("-" * 70)
    print(f"\n  Shifted Mean: {shifted_mean_10}")

    acc_10s, recalls_10s, _ = run_inference_with_stats(
        X_val_raw, y_val, shifted_mean_10, correct_std, MODEL_PATH
    )

    print(f"\n  Accuracy: {acc_10s * 100:.2f}%")
    for cls, recall in recalls_10s.items():
        print(f"  {cls} recall: {recall * 100:.2f}%")

    # ---- Experiment 4: Zero-mean normalisation (no stats) ----
    zero_mean = np.zeros_like(correct_mean)
    unit_std = np.ones_like(correct_std)

    print("\n" + "-" * 70)
    print("EXPERIMENT 4: Inference WITHOUT normalisation (mean=0, std=1)")
    print("-" * 70)
    print("  This simulates what happens if training_stats.npy is missing")
    print("  and the system falls back to unnormalised features.")

    acc_nonorm, recalls_nonorm, _ = run_inference_with_stats(
        X_val_raw, y_val, zero_mean, unit_std, MODEL_PATH
    )

    print(f"\n  Accuracy: {acc_nonorm * 100:.2f}%")
    for cls, recall in recalls_nonorm.items():
        print(f"  {cls} recall: {recall * 100:.2f}%")

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print("SUMMARY — Normalisation Experiment Results")
    print("=" * 70)
    print(f"\n{'Condition':<40} {'Accuracy':>10} {'Normal':>10} {'Warning':>10} {'Critical':>10}")
    print("-" * 80)
    print(f"{'Correct stats':<40} {acc_correct*100:>9.2f}% {recalls_correct['Normal']*100:>9.2f}% "
          f"{recalls_correct['Warning']*100:>9.2f}% {recalls_correct['Critical']*100:>9.2f}%")
    print(f"{'Mean shifted by +3σ':<40} {acc_shifted*100:>9.2f}% {recalls_shifted['Normal']*100:>9.2f}% "
          f"{recalls_shifted['Warning']*100:>9.2f}% {recalls_shifted['Critical']*100:>9.2f}%")
    print(f"{'Mean shifted by +10σ (severe)':<40} {acc_10s*100:>9.2f}% {recalls_10s['Normal']*100:>9.2f}% "
          f"{recalls_10s['Warning']*100:>9.2f}% {recalls_10s['Critical']*100:>9.2f}%")
    print(f"{'No normalisation (mean=0, std=1)':<40} {acc_nonorm*100:>9.2f}% {recalls_nonorm['Normal']*100:>9.2f}% "
          f"{recalls_nonorm['Warning']*100:>9.2f}% {recalls_nonorm['Critical']*100:>9.2f}%")

    accuracy_drop_3s = (acc_correct - acc_shifted) * 100
    accuracy_drop_10s = (acc_correct - acc_10s) * 100
    accuracy_drop_no = (acc_correct - acc_nonorm) * 100
    print(f"\n  Accuracy drop (correct → 3σ-shifted):     {accuracy_drop_3s:+.2f} pp")
    print(f"  Accuracy drop (correct → 10σ-shifted):    {accuracy_drop_10s:+.2f} pp")
    print(f"  Accuracy drop (correct → no normalisation): {accuracy_drop_no:+.2f} pp")

    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    if accuracy_drop_3s > 5 or accuracy_drop_10s > 5 or accuracy_drop_no > 5:
        print("  Shifting normalisation stats causes a SIGNIFICANT accuracy drop.")
        print("  This confirms that the model is sensitive to normalisation parameters.")
    else:
        print("  The model shows robustness to moderate normalisation shifts (3σ).")
        print("  This is because the three classes (Normal/Warning/Critical) have")
        print("  well-separated feature distributions — the anomaly modes produce")
        print("  drastically different sensor readings (e.g., temp drift +0.08°C/reading,")
        print("  vibration jump from N(0.45,0.05) to N(1.2,0.15)).")
        print()
        print("  However, in a safety-critical cold-chain system, even small accuracy")
        print("  changes matter. The PSI drift monitor (Task E1) would detect the")
        print("  resulting shift in confidence score distributions and trigger a")
        print("  [LOGIBRIDGE DRIFT ALERT], prompting model revalidation.")

    print("\n  KEY TAKEAWAY: Never recompute normalisation stats from live data.")
    print("  Always use the frozen training_stats.npy from the training phase.")
    print("  If sensor characteristics change (e.g., new sensor model), retrain")
    print("  the model and regenerate training_stats.npy as part of the retrain cycle.")
    print("=" * 70)


if __name__ == "__main__":
    main()
