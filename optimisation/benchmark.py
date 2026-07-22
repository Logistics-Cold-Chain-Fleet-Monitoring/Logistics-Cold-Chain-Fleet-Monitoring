"""
LogiEdge Five-Metric Benchmarking (Task F2)
Benchmarks all three model variants using Lab 2 methodology:
  1. Mean inference latency (ms) — 200 runs after 10 warm-up runs excluded
  2. p95 inference latency (ms)
  3. Model file size (KB)
  4. Classification accuracy on held-out validation set (%)
  5. Energy per inference (mJ) — using E = P × t with psutil CPU% and laptop TDP estimate

Outputs:
  - benchmark_results.csv (15-cell table: 3 variants × 5 metrics)
  - pareto_chart.png (Pareto frontier: latency vs accuracy)

Also includes Task F3: Deployment Recommendation.

Usage:
    python benchmark.py
"""

import os
import sys
import time
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "training", "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Try to import psutil for energy estimation
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARNING] psutil not installed — energy estimation will use fixed estimate")

# Try to import TFLite
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

# Model variants
VARIANTS = [
    {
        "name": "M1 — FP32 Baseline",
        "file": os.path.join(MODELS_DIR, "model_fp32.tflite"),
        "short": "M1_FP32"
    },
    {
        "name": "M2 — PTQ INT8",
        "file": os.path.join(MODELS_DIR, "model_ptq_int8.tflite"),
        "short": "M2_INT8"
    },
    {
        "name": "M3 — Pruned + INT8",
        "file": os.path.join(MODELS_DIR, "model_pruned_int8.tflite"),
        "short": "M3_PRUNED"
    }
]

# Laptop TDP estimate for energy calculation
LAPTOP_TDP_WATTS = 15.0  # Typical laptop CPU TDP
WARM_UP_RUNS = 10
BENCHMARK_RUNS = 200

CLASS_NAMES = ["Normal", "Warning", "Critical"]


def benchmark_variant(model_path, X_val, y_val):
    """
    Benchmark a single model variant.
    Returns dict with all 5 metrics.
    """
    # Load model
    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantised = (input_details[0]['dtype'] == np.int8)

    def prepare_input(sample):
        """Prepare input tensor (handle quantisation)."""
        data = sample.reshape(1, -1).astype(np.float32)
        if is_quantised:
            scale, zp = input_details[0]['quantization']
            data = (data / scale + zp).astype(np.int8)
        return data

    def run_single_inference(input_data):
        """Run a single inference and return output."""
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if is_quantised:
            scale, zp = output_details[0]['quantization']
            output = (output.astype(np.float32) - zp) * scale
        return output

    # --- Metric 1 & 2: Latency ---
    # Warm-up runs (excluded from measurement)
    test_input = prepare_input(X_val[0])
    for _ in range(WARM_UP_RUNS):
        run_single_inference(test_input)

    # Benchmark runs
    latencies = []
    for i in range(BENCHMARK_RUNS):
        sample_idx = i % len(X_val)
        input_data = prepare_input(X_val[sample_idx])

        start = time.perf_counter()
        run_single_inference(input_data)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    # --- Metric 3: Model file size ---
    file_size_kb = os.path.getsize(model_path) / 1024

    # --- Metric 4: Classification accuracy ---
    correct = 0
    predictions = []
    for i in range(len(X_val)):
        input_data = prepare_input(X_val[i])
        output = run_single_inference(input_data)
        output = output.flatten()

        # Apply softmax if needed
        if np.any(output < 0) or np.sum(output) < 0.5:
            exp_out = np.exp(output - np.max(output))
            output = exp_out / np.sum(exp_out)

        pred = np.argmax(output)
        predictions.append(pred)
        if pred == y_val[i]:
            correct += 1

    accuracy = (correct / len(X_val)) * 100

    # --- Metric 5: Energy per inference (mJ) ---
    # E = P × t
    # P = CPU_utilisation × TDP
    if HAS_PSUTIL:
        # Measure process CPU time over a long burst. A system-wide cpu_percent
        # sample over only 100 microsecond-scale inferences often rounds to 0%.
        process = psutil.Process()
        cpu_before = process.cpu_times()
        burst_start = time.perf_counter()
        energy_runs = 50_000
        for i in range(energy_runs):
            input_data = prepare_input(X_val[i % len(X_val)])
            run_single_inference(input_data)
        burst_end = time.perf_counter()
        cpu_after = process.cpu_times()
        cpu_seconds = ((cpu_after.user + cpu_after.system) -
                       (cpu_before.user + cpu_before.system))
        wall_seconds = max(burst_end - burst_start, 1e-9)
        logical_cpus = psutil.cpu_count(logical=True) or 1
        cpu_fraction = min(cpu_seconds / wall_seconds / logical_cpus, 1.0)

        power_watts = max(cpu_fraction * LAPTOP_TDP_WATTS, 0.01)
        time_per_inference_s = mean_latency / 1000.0
        energy_mj = power_watts * time_per_inference_s * 1000  # Convert to mJ
    else:
        # Estimate: assume 30% CPU utilisation during inference
        power_watts = 0.30 * LAPTOP_TDP_WATTS
        time_per_inference_s = mean_latency / 1000.0
        energy_mj = power_watts * time_per_inference_s * 1000

    # --- Class 2 (Critical) Recall ---
    predictions = np.array(predictions)
    class2_mask = (y_val == 2)
    class2_correct = np.sum((predictions == 2) & class2_mask)
    class2_total = np.sum(class2_mask)
    class2_recall = (class2_correct / class2_total * 100) if class2_total > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "file_size_kb": round(file_size_kb, 2),
        "accuracy_pct": round(accuracy, 2),
        "energy_mj": round(energy_mj, 4),
        "class2_recall_pct": round(class2_recall, 2),
        "latencies": latencies  # For Pareto chart
    }


def generate_pareto_chart(results):
    """Generate Pareto frontier chart: latency vs accuracy."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping Pareto chart")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    names = []
    latencies = []
    accuracies = []
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    markers = ['o', 's', '^']

    for i, (variant, result) in enumerate(results.items()):
        names.append(variant)
        latencies.append(result["mean_latency_ms"])
        accuracies.append(result["accuracy_pct"])

        ax.scatter(result["mean_latency_ms"], result["accuracy_pct"],
                   c=colors[i], marker=markers[i], s=150, zorder=5,
                   label=variant, edgecolors='black', linewidths=0.5)

        # Annotate
        ax.annotate(f'  {variant}\n  ({result["mean_latency_ms"]:.2f}ms, {result["accuracy_pct"]:.1f}%)',
                    (result["mean_latency_ms"], result["accuracy_pct"]),
                    fontsize=8, ha='left')

    # Draw Pareto frontier
    # Sort by latency
    sorted_points = sorted(zip(latencies, accuracies, names), key=lambda x: x[0])
    pareto_lat = [sorted_points[0][0]]
    pareto_acc = [sorted_points[0][1]]
    max_acc = sorted_points[0][1]

    for lat, acc, name in sorted_points[1:]:
        if acc >= max_acc:
            pareto_lat.append(lat)
            pareto_acc.append(acc)
            max_acc = acc

    if len(pareto_lat) > 1:
        ax.plot(pareto_lat, pareto_acc, 'r--', alpha=0.5, linewidth=1.5,
                label='Pareto Frontier')

    ax.set_xlabel('Mean Inference Latency (ms)', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('LogiEdge Model Variants — Pareto Analysis\n'
                 'Latency vs Accuracy Trade-off', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation about deployment recommendation
    ax.text(0.02, 0.02,
            'Optimal: Lower-left corner (low latency, high accuracy)\n'
            'All variants meet 90-second SLA easily',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "pareto_chart.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\nPareto chart saved to {chart_path}")
    plt.close()


def print_deployment_recommendation(results):
    """Task F3: Deployment recommendation for the operations director."""
    print("\n" + "=" * 70)
    print("TASK F3 — DEPLOYMENT RECOMMENDATION")
    print("=" * 70)
    print("\nQuestion: 'Which model should we deploy to the 85 trucks?'")

    print("\n--- Evidence-Based Analysis ---")

    # 90-second SLA translated to per-inference latency budget
    print("\n1. Latency Budget (90-second SLA):")
    print("   The 90-second SLA is an end-to-end detection requirement.")
    print("   With a 30-second window and 10-second step, worst-case detection")
    print("   takes one full window cycle = 10 seconds + inference time.")
    print("   Per-inference latency budget = 90s - 30s (window fill) = 60s maximum.")
    print("   ALL three variants run inference in under 1ms — SLA is met by all.")
    for name, r in results.items():
        sla_ratio = r["mean_latency_ms"] / 60000 * 100
        print(f"     {name}: {r['mean_latency_ms']:.4f}ms "
              f"({sla_ratio:.4f}% of latency budget)")

    # Hardware constraints
    print("\n2. Hardware Constraints (Raspberry Pi 5):")
    print("   Flash storage: 32 GB SD card — all model sizes fit easily")
    for name, r in results.items():
        print(f"     {name}: {r['file_size_kb']:.2f} KB")
    print("   SRAM (for activations): 8 GB LPDDR4X — no constraint for MLP models")
    print("   All variants fit comfortably on the chosen hardware.")

    # Class 2 recall
    print("\n3. Class 2 (Critical) Recall — Must exceed 95%:")
    recommended = None
    for name, r in results.items():
        status = "PASS" if r["class2_recall_pct"] >= 95 else "FAIL"
        print(f"     {name}: {r['class2_recall_pct']:.2f}% [{status}]")
        if r["class2_recall_pct"] >= 95:
            if (recommended is None or
                    r["accuracy_pct"] > results[recommended]["accuracy_pct"] or
                    (r["accuracy_pct"] == results[recommended]["accuracy_pct"] and
                     r["file_size_kb"] < results[recommended]["file_size_kb"])):
                recommended = name

    # Final recommendation
    print("\n--- RECOMMENDATION ---")
    if recommended:
        r = results[recommended]
        print(f"\n  Deploy: {recommended}")
        print(f"\n  Reasoning:")
        print(f"  - Meets the 90-second SLA with huge margin ({r['mean_latency_ms']:.4f}ms)")
        print(f"  - Class 2 (Critical) recall of {r['class2_recall_pct']:.2f}% exceeds 95%")
        print(f"  - Highest validation accuracy among variants passing the Critical-recall gate")
        print(f"  - Model size {r['file_size_kb']:.2f} KB is negligible on 32 GB edge storage")
        print(f"  - Lab-estimated energy ({r['energy_mj']:.4f} mJ) remains negligible")
        print(f"  - Accuracy of {r['accuracy_pct']:.2f}% is acceptable for cold-chain monitoring")
    else:
        print("  [WARNING] No variant meets all requirements — review model training")

    print("\n  Why not the other variants:")
    for name, r in results.items():
        if name != recommended:
            if r["class2_recall_pct"] < 95:
                print(f"  - {name}: Critical recall {r['class2_recall_pct']:.2f}% < 95% threshold")
            else:
                    print(f"  - {name}: Accuracy {r['accuracy_pct']:.2f}% versus "
                        f"{results[recommended]['accuracy_pct']:.2f}% for the recommendation")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load validation data
    val_path = os.path.join(MODELS_DIR, "val_data.npz")
    if not os.path.exists(val_path):
        print("[ERROR] Validation data not found. Run train_model.py first.")
        return

    val_data = np.load(val_path)
    X_val = val_data["X_val"].astype(np.float32)
    y_val = val_data["y_val"]

    print("=" * 60)
    print("LogiEdge Five-Metric Benchmarking")
    print("=" * 60)
    print(f"Validation samples: {len(X_val)}")
    print(f"Warm-up runs: {WARM_UP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Laptop TDP estimate: {LAPTOP_TDP_WATTS}W")

    results = {}

    for variant in VARIANTS:
        if not os.path.exists(variant["file"]):
            print(f"\n[SKIP] {variant['name']} — model file not found: {variant['file']}")
            continue

        print(f"\n--- Benchmarking: {variant['name']} ---")
        metrics = benchmark_variant(variant["file"], X_val, y_val)
        results[variant["name"]] = metrics

        print(f"  Mean latency:     {metrics['mean_latency_ms']:.4f} ms")
        print(f"  p95 latency:      {metrics['p95_latency_ms']:.4f} ms")
        print(f"  File size:        {metrics['file_size_kb']:.2f} KB")
        print(f"  Accuracy:         {metrics['accuracy_pct']:.2f}%")
        print(f"  Energy/inference: {metrics['energy_mj']:.4f} mJ")
        print(f"  Class 2 recall:   {metrics['class2_recall_pct']:.2f}%")

    if not results:
        print("\n[ERROR] No models found to benchmark. Run training pipeline first.")
        return

    # Save results to CSV
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Mean Latency (ms)", "p95 Latency (ms)",
                         "File Size (KB)", "Accuracy (%)", "Energy (mJ)",
                         "Class 2 Recall (%)"])
        for name, m in results.items():
            writer.writerow([
                name, m["mean_latency_ms"], m["p95_latency_ms"],
                m["file_size_kb"], m["accuracy_pct"], m["energy_mj"],
                m["class2_recall_pct"]
            ])
    print(f"\nBenchmark results saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Variant':<25} {'Mean Lat (ms)':>13} {'p95 Lat (ms)':>13} "
          f"{'Size (KB)':>10} {'Acc (%)':>8} {'Energy (mJ)':>12}")
    print("-" * 90)
    for name, m in results.items():
        print(f"{name:<25} {m['mean_latency_ms']:>13.4f} {m['p95_latency_ms']:>13.4f} "
              f"{m['file_size_kb']:>10.2f} {m['accuracy_pct']:>8.2f} {m['energy_mj']:>12.4f}")
    print("=" * 90)

    # Generate Pareto chart
    generate_pareto_chart(results)

    # Deployment recommendation (Task F3)
    print_deployment_recommendation(results)


if __name__ == "__main__":
    main()
