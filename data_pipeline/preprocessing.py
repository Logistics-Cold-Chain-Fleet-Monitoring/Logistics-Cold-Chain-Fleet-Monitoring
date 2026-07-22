"""
LogiEdge Preprocessing Pipeline (Task C2)
Implements the complete preprocessing pipeline:
  1. Filtering: 5-sample moving average on temperature and vibration
  2. Feature extraction per 30-second sliding window (step = 10 seconds):
     - temperature mean, temperature std, temperature rate-of-change (°C/min)
     - vibration RMS, vibration peak, vibration kurtosis
  3. Normalisation using saved training_stats.npy

Also subscribes to MQTT sensor topics and processes in real-time.
Can also be used as a library by generate_dataset.py.

Usage:
    python preprocessing.py --truck-id TRUCK_01
"""

import argparse
import json
import os
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from scipy import stats as sp_stats

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Window parameters
WINDOW_SIZE = 30     # seconds
WINDOW_STEP = 10     # seconds
FILTER_SIZE = 5      # moving average window

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_FILE = os.path.join(SCRIPT_DIR, "training_stats.npy")


class SensorBuffer:
    """Buffers raw sensor readings and applies moving average filter."""

    def __init__(self, filter_size=FILTER_SIZE):
        self.raw_values = deque()
        self.timestamps = deque()
        self.filter_size = filter_size

    def add(self, value, timestamp):
        self.raw_values.append(value)
        self.timestamps.append(timestamp)

    def get_filtered(self):
        """Apply 5-sample moving average filter and return filtered values."""
        values = list(self.raw_values)
        if len(values) < self.filter_size:
            return values
        filtered = []
        for i in range(len(values)):
            start = max(0, i - self.filter_size + 1)
            window = values[start:i + 1]
            filtered.append(np.mean(window))
        return filtered

    def get_window(self, window_seconds):
        """Get the last window_seconds of filtered data."""
        filtered = self.get_filtered()
        if len(filtered) == 0:
            return []
        return filtered[-window_seconds:]

    def clear_old(self, keep_seconds):
        """Remove readings older than keep_seconds."""
        while len(self.raw_values) > keep_seconds * 2:
            self.raw_values.popleft()
            self.timestamps.popleft()


def extract_features(temp_window, vib_window):
    """
    Extract 6-value feature vector from a 30-second window:
      [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis]
    """
    if len(temp_window) < 2 or len(vib_window) < 2:
        return None

    temp_arr = np.array(temp_window)
    vib_arr = np.array(vib_window)

    # Temperature features
    temp_mean = np.mean(temp_arr)
    temp_std = np.std(temp_arr)
    # Rate of change: (last - first) / window_duration_in_minutes
    # Window is 30 seconds = 0.5 minutes
    temp_roc = (temp_arr[-1] - temp_arr[0]) / 0.5  # °C/min

    # Vibration features
    vib_rms = np.sqrt(np.mean(vib_arr ** 2))
    vib_peak = np.max(vib_arr)
    # Kurtosis — use Fisher definition (excess kurtosis)
    if len(vib_arr) >= 4:
        vib_kurtosis = float(sp_stats.kurtosis(vib_arr, fisher=True))
    else:
        vib_kurtosis = 0.0

    features = np.array([
        temp_mean, temp_std, temp_roc,
        vib_rms, vib_peak, vib_kurtosis
    ], dtype=np.float32)

    return features


def normalise_features(features, mean, std):
    """Z-score normalisation using saved training stats."""
    # Avoid division by zero
    std_safe = np.where(std == 0, 1.0, std)
    return (features - mean) / std_safe


def load_training_stats(stats_file=STATS_FILE):
    """Load saved normalisation statistics."""
    if not os.path.exists(stats_file):
        return None, None
    data = np.load(stats_file, allow_pickle=True).item()
    return data["mean"], data["std"]


def save_training_stats(all_features, stats_file=STATS_FILE):
    """Compute and save mean/std from clean Normal-class data."""
    features_arr = np.array(all_features)
    mean = np.mean(features_arr, axis=0)
    std = np.std(features_arr, axis=0)
    np.save(stats_file, {"mean": mean, "std": std})
    print(f"[PREPROCESS] Saved training stats to {stats_file}")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    return mean, std


class PreprocessingPipeline:
    """
    Real-time preprocessing pipeline that subscribes to MQTT sensor topics,
    extracts features from sliding windows, and publishes feature vectors.
    """

    def __init__(self, truck_id="TRUCK_01", publish_features=True):
        self.truck_id = truck_id
        self.publish_features = publish_features

        # Sensor buffers
        self.temp_buffer = SensorBuffer()
        self.vib_buffer = SensorBuffer()

        # Feature output
        self.feature_vectors = []
        self.last_window_time = 0

        # Normalisation stats
        self.norm_mean, self.norm_std = load_training_stats()
        if self.norm_mean is not None:
            print("[PREPROCESS] Loaded normalisation stats from training_stats.npy")
        else:
            print("[PREPROCESS] No training_stats.npy found — features will be unnormalised")

        # MQTT
        self.client = mqtt.Client(client_id=f"preprocess_{truck_id}")
        self.topic_base = f"logibridge/trucks/{truck_id}"

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[PREPROCESS] Connected to MQTT broker")
            client.subscribe(f"{self.topic_base}/sensors/temperature", qos=0)
            client.subscribe(f"{self.topic_base}/sensors/vibration", qos=0)
            print(f"[PREPROCESS] Subscribed to sensor topics")
        else:
            print(f"[PREPROCESS] Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        topic = msg.topic
        timestamp = payload.get("timestamp", "")
        value = payload.get("value", 0.0)

        if "temperature" in topic:
            self.temp_buffer.add(value, timestamp)
        elif "vibration" in topic:
            self.vib_buffer.add(value, timestamp)

        # Check if we should extract a new window
        self._try_extract_window()

    def _try_extract_window(self):
        """Extract features if enough time has passed since last window."""
        current_time = time.time()
        if current_time - self.last_window_time < WINDOW_STEP:
            return

        # Get filtered window data
        temp_window = self.temp_buffer.get_window(WINDOW_SIZE)
        vib_window = self.vib_buffer.get_window(WINDOW_SIZE // 2)  # vibration is at 0.5 Hz

        if len(temp_window) < 10 or len(vib_window) < 5:
            return

        features = extract_features(temp_window, vib_window)
        if features is None:
            return

        # Normalise if stats are available
        if self.norm_mean is not None and self.norm_std is not None:
            features_norm = normalise_features(features, self.norm_mean, self.norm_std)
        else:
            features_norm = features

        self.feature_vectors.append(features_norm)
        self.last_window_time = current_time

        # Publish to MQTT
        if self.publish_features:
            feat_payload = json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "features": features_norm.tolist(),
                "raw_features": features.tolist()
            })
            self.client.publish(
                f"{self.topic_base}/features/window",
                feat_payload, qos=0
            )

        # Clean up old buffer data
        self.temp_buffer.clear_old(WINDOW_SIZE * 3)
        self.vib_buffer.clear_old(WINDOW_SIZE * 3)

    def start(self):
        """Start the real-time preprocessing pipeline."""
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        except ConnectionRefusedError:
            print(f"[ERROR] Cannot connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
            return

        print(f"[PREPROCESS] Starting pipeline for {self.truck_id}...")
        self.client.loop_forever()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


def offline_extract_features(temp_readings, vib_readings,
                             window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """
    Offline feature extraction from lists of sensor readings.
    Used by generate_dataset.py for batch processing.

    Args:
        temp_readings: list of temperature values (1 Hz)
        vib_readings: list of vibration RMS values (0.5 Hz)
        window_size: window size in seconds
        step: step size in seconds

    Returns:
        list of 6-value feature vectors (unnormalised)
    """
    features_list = []

    # Apply moving average filter
    temp_filtered = []
    for i in range(len(temp_readings)):
        start = max(0, i - FILTER_SIZE + 1)
        window = temp_readings[start:i + 1]
        temp_filtered.append(np.mean(window))

    vib_filtered = []
    for i in range(len(vib_readings)):
        start = max(0, i - FILTER_SIZE + 1)
        window = vib_readings[start:i + 1]
        vib_filtered.append(np.mean(window))

    # Sliding window feature extraction
    # Temperature is at 1 Hz, so window_size samples = window_size seconds
    # Vibration is at 0.5 Hz, so window_size/2 samples = window_size seconds
    t_win = window_size
    v_win = window_size // 2
    t_step = step
    v_step = step // 2

    num_windows = (len(temp_filtered) - t_win) // t_step + 1

    for i in range(num_windows):
        t_start = i * t_step
        t_end = t_start + t_win
        v_start = i * v_step
        v_end = v_start + v_win

        if t_end > len(temp_filtered) or v_end > len(vib_filtered):
            break

        t_window = temp_filtered[t_start:t_end]
        v_window = vib_filtered[v_start:v_end]

        feat = extract_features(t_window, v_window)
        if feat is not None:
            features_list.append(feat)

    return features_list


def main():
    parser = argparse.ArgumentParser(description="LogiEdge Preprocessing Pipeline")
    parser.add_argument("--truck-id", type=str, default="TRUCK_01")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(truck_id=args.truck_id)
    try:
        pipeline.start()
    except KeyboardInterrupt:
        pipeline.stop()
        print(f"\n[PREPROCESS] Stopped. Extracted {len(pipeline.feature_vectors)} feature vectors.")


if __name__ == "__main__":
    main()
