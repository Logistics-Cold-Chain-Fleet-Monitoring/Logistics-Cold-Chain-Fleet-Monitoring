"""
LogiEdge PSI Drift Monitor (Task E1)
Implements Population Stability Index (PSI) monitoring on the model's
output confidence score distribution.

Steps:
  1. Build reference distribution from 300 clean Normal-class inferences
     across 4 bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]
  2. Monitor PSI on rolling window of last 100 inferences every 60 seconds
  3. Alert when PSI > 0.25
  4. Support --anomaly flag to inject drift mid-run

Usage:
    # Step 1: Build reference distribution
    python drift_monitor.py --mode build-reference --model-path ../training/models/model_fp32.tflite

    # Step 2: Run monitoring (with optional drift injection)
    python drift_monitor.py --mode monitor --model-path ../inference/model.tflite --truck-id TRUCK_01

    # Step 3: Demo drift injection (auto-injects anomaly data mid-run)
    python drift_monitor.py --mode demo
"""

import argparse
import json
import os
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

REFERENCE_DIST_FILE = os.path.join(SCRIPT_DIR, "reference_dist.json")

# PSI parameters from the assignment
PSI_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]  # 4 bins
PSI_THRESHOLD = 0.25
ROLLING_WINDOW = 100
CHECK_INTERVAL = 60  # seconds

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

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


def compute_psi(reference_dist, current_dist):
    """
    Compute Population Stability Index between reference and current distributions.
    PSI = sum((current_i - reference_i) * ln(current_i / reference_i))

    Both inputs should be probability distributions (sum to 1) over the same bins.
    """
    # Add small epsilon to avoid division by zero or log(0)
    eps = 1e-6
    ref = np.array(reference_dist) + eps
    cur = np.array(current_dist) + eps

    # Normalise to ensure they sum to 1
    ref = ref / ref.sum()
    cur = cur / cur.sum()

    psi = np.sum((cur - ref) * np.log(cur / ref))
    return float(psi)


def confidence_to_distribution(confidence_scores, bins=PSI_BINS):
    """Convert a list of confidence scores to a binned probability distribution."""
    counts, _ = np.histogram(confidence_scores, bins=bins)
    total = max(len(confidence_scores), 1)
    distribution = counts / total
    return distribution.tolist()


def build_reference_distribution(model_path, dataset_path=None):
    """
    Build reference distribution from 300 clean Normal-class inferences.
    """
    import sys
    sys.path.insert(0, PROJECT_DIR)
    from data_pipeline.simulator import generate_temperature, generate_vibration
    from data_pipeline.preprocessing import offline_extract_features, load_training_stats

    print("Building reference distribution from 300 Normal-class windows...")

    # Generate normal data
    temp_readings = []
    vib_readings = []
    duration = 3600  # 1 hour of normal data to get enough windows

    for tick in range(duration):
        temp_readings.append(generate_temperature(tick, "none"))
        if tick % 2 == 0:
            vib_readings.append(generate_vibration("none"))

    # Extract features
    features = offline_extract_features(temp_readings, vib_readings)
    print(f"  Extracted {len(features)} feature windows")

    # Load normalisation stats
    stats_file = os.path.join(PROJECT_DIR, "data_pipeline", "training_stats.npy")
    mean, std = load_training_stats(stats_file)
    if mean is None:
        print("[ERROR] training_stats.npy not found. Run generate_dataset.py first.")
        return

    # Normalise features
    std_safe = np.where(std == 0, 1.0, std)
    features_norm = [(f - mean) / std_safe for f in features]

    # Load model
    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantised = (input_details[0]['dtype'] == np.int8)

    # Run inference on 300 windows
    confidence_scores = []
    num_samples = min(300, len(features_norm))

    for i in range(num_samples):
        sample = np.array(features_norm[i], dtype=np.float32).reshape(1, -1)

        if is_quantised:
            input_scale, input_zp = input_details[0]['quantization']
            sample = (sample / input_scale + input_zp).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if is_quantised:
            output_scale, output_zp = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zp) * output_scale

        output = output.flatten()
        # Apply softmax if needed
        if np.any(output < 0) or np.sum(output) < 0.5:
            exp_out = np.exp(output - np.max(output))
            output = exp_out / np.sum(exp_out)

        # Track confidence in the expected Normal operating state. Maximum
        # confidence stays high for both confident Normal and Critical outputs,
        # so it cannot reveal this operational distribution shift.
        normal_confidence = float(output[0])
        confidence_scores.append(normal_confidence)

    print(f"  Collected {len(confidence_scores)} confidence scores")

    # Compute reference distribution
    ref_dist = confidence_to_distribution(confidence_scores)

    # Save reference distribution
    ref_data = {
        "bins": PSI_BINS,
        "distribution": ref_dist,
        "num_samples": len(confidence_scores),
        "mean_confidence": float(np.mean(confidence_scores)),
        "std_confidence": float(np.std(confidence_scores)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    with open(REFERENCE_DIST_FILE, "w") as f:
        json.dump(ref_data, f, indent=2)

    print(f"\nReference distribution saved to {REFERENCE_DIST_FILE}")
    print(f"  Bins: {PSI_BINS}")
    print(f"  Distribution: {[f'{d:.4f}' for d in ref_dist]}")
    print(f"  Mean confidence: {np.mean(confidence_scores):.4f}")
    return ref_data


class DriftMonitor:
    """Real-time PSI drift monitor subscribing to inference results via MQTT."""

    def __init__(self, truck_id="TRUCK_01"):
        self.truck_id = truck_id
        self.topic_base = f"logibridge/trucks/{truck_id}"

        # Load reference distribution
        if not os.path.exists(REFERENCE_DIST_FILE):
            raise FileNotFoundError(
                f"Reference distribution not found at {REFERENCE_DIST_FILE}. "
                "Run with --mode build-reference first."
            )

        with open(REFERENCE_DIST_FILE) as f:
            ref_data = json.load(f)
        self.reference_dist = ref_data["distribution"]
        print(f"[DRIFT] Loaded reference distribution: {self.reference_dist}")

        # Rolling window of confidence scores
        self.confidence_window = deque(maxlen=ROLLING_WINDOW)
        self.last_check_time = time.time()
        self.psi_history = []

        # MQTT
        self.client = mqtt.Client(client_id=f"drift_monitor_{truck_id}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[DRIFT] Connected to MQTT broker")
            client.subscribe(f"{self.topic_base}/inference", qos=1)
            print(f"[DRIFT] Subscribed to {self.topic_base}/inference")
        else:
            print(f"[DRIFT] Connection failed: rc={rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        confidence = payload.get("normal_confidence", payload.get("confidence", 0.0))
        self.confidence_window.append(confidence)

        # Check PSI every CHECK_INTERVAL seconds
        current_time = time.time()
        if current_time - self.last_check_time >= CHECK_INTERVAL:
            self._check_psi(client)
            self.last_check_time = current_time

    def _check_psi(self, client):
        """Compute PSI and check for drift."""
        if len(self.confidence_window) < 20:
            return

        # Compute current distribution
        current_dist = confidence_to_distribution(list(self.confidence_window))

        # Compute PSI
        psi = compute_psi(self.reference_dist, current_dist)
        self.psi_history.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "psi": psi,
            "window_size": len(self.confidence_window)
        })

        # Print current PSI value (required by assignment)
        print(f"  [PSI] Current PSI={psi:.3f} (window={len(self.confidence_window)} samples)")

        # Alert if PSI exceeds threshold
        if psi > PSI_THRESHOLD:
            print(f"  [LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")

            # Publish drift alert to MQTT
            alert_payload = json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "truck_id": self.truck_id,
                "psi": round(psi, 3),
                "threshold": PSI_THRESHOLD,
                "current_distribution": current_dist,
                "reference_distribution": self.reference_dist
            })
            client.publish(
                f"{self.topic_base}/alerts/drift",
                alert_payload, qos=1
            )

    def start(self):
        """Start the drift monitor."""
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        except ConnectionRefusedError:
            print(f"[ERROR] Cannot connect to MQTT at {MQTT_BROKER}:{MQTT_PORT}")
            return

        print(f"[DRIFT] Monitoring started for {self.truck_id}")
        print(f"[DRIFT] PSI threshold: {PSI_THRESHOLD}")
        print(f"[DRIFT] Rolling window: {ROLLING_WINDOW} inferences")
        print(f"[DRIFT] Check interval: {CHECK_INTERVAL} seconds")
        self.client.loop_forever()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


def run_demo(model_path):
    """
    Demo mode: simulates the full drift injection and recovery scenario.
    Runs entirely offline (no MQTT needed) to show PSI behavior.
    """
    import sys
    sys.path.insert(0, PROJECT_DIR)
    from data_pipeline.simulator import generate_temperature, generate_vibration
    from data_pipeline.preprocessing import offline_extract_features, load_training_stats

    print("=" * 60)
    print("PSI Drift Demo — Injection and Recovery")
    print("=" * 60)

    # Load reference distribution
    with open(REFERENCE_DIST_FILE) as f:
        ref_data = json.load(f)
    ref_dist = ref_data["distribution"]

    # Load model and stats
    stats_file = os.path.join(PROJECT_DIR, "data_pipeline", "training_stats.npy")
    mean, std = load_training_stats(stats_file)
    std_safe = np.where(std == 0, 1.0, std)

    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantised = (input_details[0]['dtype'] == np.int8)

    def run_inference_batch(anomaly_mode, duration, label):
        """Generate data, extract features, run inference, return confidence scores."""
        temp_r, vib_r = [], []
        for tick in range(duration):
            temp_r.append(generate_temperature(tick, anomaly_mode))
            if tick % 2 == 0:
                vib_r.append(generate_vibration(anomaly_mode))
        features = offline_extract_features(temp_r, vib_r)
        features_norm = [(f - mean) / std_safe for f in features]

        scores = []
        for feat in features_norm:
            sample = np.array(feat, dtype=np.float32).reshape(1, -1)
            if is_quantised:
                sc, zp = input_details[0]['quantization']
                sample = (sample / sc + zp).astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            if is_quantised:
                osc, ozp = output_details[0]['quantization']
                out = (out.astype(np.float32) - ozp) * osc
            out = out.flatten()
            if np.any(out < 0) or np.sum(out) < 0.5:
                exp_out = np.exp(out - np.max(out))
                out = exp_out / np.sum(exp_out)
            scores.append(float(out[0]))

        print(f"\n  [{label}] {len(scores)} inferences, "
              f"mean_conf={np.mean(scores):.4f}")
        return scores

    # Phase 1: Clean data (should show low PSI)
    print("\n--- Phase 1: Clean Normal Data ---")
    clean_scores = run_inference_batch("none", 1200, "CLEAN")
    clean_window = clean_scores[-100:]
    clean_dist = confidence_to_distribution(clean_window)
    psi_clean = compute_psi(ref_dist, clean_dist)
    print(f"  PSI (clean): {psi_clean:.3f} — {'OK' if psi_clean < PSI_THRESHOLD else 'DRIFT!'}")

    # Phase 2: Inject anomaly (PSI must cross 0.25 within 5 minutes)
    print("\n--- Phase 2: Drift Injection (--anomaly combined) ---")
    drift_scores = run_inference_batch("combined", 300, "DRIFT")
    drift_window = drift_scores[-100:] if len(drift_scores) >= 100 else drift_scores
    drift_dist = confidence_to_distribution(drift_window)
    psi_drift = compute_psi(ref_dist, drift_dist)
    print(f"  PSI (drift): {psi_drift:.3f} — {'OK' if psi_drift < PSI_THRESHOLD else 'DRIFT!'}")

    if psi_drift > PSI_THRESHOLD:
        print(f"  [LOGIBRIDGE DRIFT ALERT] PSI={psi_drift:.3f}")
    else:
        print("  [WARNING] PSI did not cross threshold — check model or reference distribution")

    # Phase 3: Recovery (PSI should return below 0.10)
    print("\n--- Phase 3: Recovery (back to clean data) ---")
    recovery_scores = run_inference_batch("none", 1200, "RECOVERY")
    recovery_window = recovery_scores[-100:]
    recovery_dist = confidence_to_distribution(recovery_window)
    psi_recovery = compute_psi(ref_dist, recovery_dist)
    print(f"  PSI (recovery): {psi_recovery:.3f} — {'OK' if psi_recovery < 0.10 else 'STILL DRIFTING'}")

    # Summary
    print("\n" + "=" * 60)
    print("PSI Drift Demo Summary")
    print("=" * 60)
    print(f"  Phase 1 (Clean):    PSI = {psi_clean:.3f}  {'✓' if psi_clean < 0.10 else '✗'}")
    print(f"  Phase 2 (Drift):    PSI = {psi_drift:.3f}  {'✓ Alert triggered' if psi_drift > PSI_THRESHOLD else '✗ No alert'}")
    print(f"  Phase 3 (Recovery): PSI = {psi_recovery:.3f}  {'✓ Recovered' if psi_recovery < 0.10 else '✗ Not recovered'}")
    print(f"\n  Reference dist: {[f'{d:.4f}' for d in ref_dist]}")
    print(f"  Clean dist:     {[f'{d:.4f}' for d in clean_dist]}")
    print(f"  Drift dist:     {[f'{d:.4f}' for d in drift_dist]}")
    print(f"  Recovery dist:  {[f'{d:.4f}' for d in recovery_dist]}")


def main():
    parser = argparse.ArgumentParser(description="LogiEdge PSI Drift Monitor")
    parser.add_argument("--mode", choices=["build-reference", "monitor", "demo"],
                        default="demo")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(PROJECT_DIR, "training", "models", "model_fp32.tflite"))
    parser.add_argument("--truck-id", type=str, default="TRUCK_01")
    args = parser.parse_args()

    if args.mode == "build-reference":
        build_reference_distribution(args.model_path)

    elif args.mode == "monitor":
        monitor = DriftMonitor(truck_id=args.truck_id)
        try:
            monitor.start()
        except KeyboardInterrupt:
            monitor.stop()
            print(f"\n[DRIFT] Monitor stopped. PSI history: {len(monitor.psi_history)} checks")

    elif args.mode == "demo":
        run_demo(args.model_path)


if __name__ == "__main__":
    main()
