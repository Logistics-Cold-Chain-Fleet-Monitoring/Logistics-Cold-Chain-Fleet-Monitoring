"""
LogiEdge Inference Service (Task D2)
Containerised inference pipeline:
  1. Subscribes to MQTT feature vectors
  2. Runs TFLite model inference
  3. Publishes results to logibridge/trucks/{truck_id}/inference
  4. Publishes alerts for Warning/Critical detections

Accepts MODEL_PATH environment variable to switch model variants without rebuild.

Usage (standalone):
    python inference_service.py --truck-id TRUCK_01

Usage (Docker):
    docker run -e MODEL_PATH=/app/model.tflite -e TRUCK_ID=TRUCK_01 logibridge-inference
"""

import argparse
import json
import os
import sqlite3
import time
import numpy as np
import paho.mqtt.client as mqtt

# Environment variables for Docker
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")
TRUCK_ID = os.environ.get("TRUCK_ID", "TRUCK_01")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
OPS_MQTT_BROKER = os.environ.get("OPS_MQTT_BROKER", "").strip()
OPS_MQTT_PORT = int(os.environ.get("OPS_MQTT_PORT", "1883"))
ALERT_DB_PATH = os.environ.get("ALERT_DB_PATH", "/opt/logibridge/alerts.db")

CLASS_NAMES = ["Normal", "Warning", "Critical"]

# Try to import TFLite — handle both full TF and tflite-runtime
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


class AlertStore:
    """Durable local alert queue used while the cellular link is unavailable."""

    def __init__(self, db_path):
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                synced INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.connection.commit()

    def append(self, topic, payload):
        cursor = self.connection.execute(
            "INSERT INTO alerts(topic, payload, created_at) VALUES (?, ?, ?)",
            (topic, payload, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.connection.commit()
        return cursor.lastrowid

    def pending(self):
        return self.connection.execute(
            "SELECT id, topic, payload FROM alerts WHERE synced = 0 ORDER BY id"
        ).fetchall()

    def mark_synced(self, alert_id):
        self.connection.execute(
            "UPDATE alerts SET synced = 1 WHERE id = ?", (alert_id,)
        )
        self.connection.commit()

    def close(self):
        self.connection.close()


class InferenceService:
    """Real-time inference service subscribing to MQTT feature vectors."""

    def __init__(self, model_path, truck_id, broker, port,
                 ops_broker="", ops_port=1883, alert_db_path=ALERT_DB_PATH):
        self.truck_id = truck_id
        self.topic_base = f"logibridge/trucks/{truck_id}"
        self.inference_count = 0
        self.confidence_scores = []  # For drift monitoring
        self.alert_store = AlertStore(alert_db_path)
        self.ops_connected = False

        # Load TFLite model
        print(f"[INFERENCE] Loading model from {model_path}")
        self.interpreter = TFLiteInterpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"[INFERENCE] Model loaded successfully")
        print(f"  Input type: {self.input_details[0]['dtype']}")
        print(f"  Output type: {self.output_details[0]['dtype']}")
        print(f"  Input shape: {self.input_details[0]['shape']}")

        # Check if model is quantised (INT8)
        self.is_quantised = (self.input_details[0]['dtype'] == np.int8)
        if self.is_quantised:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
            print(f"  Quantised model detected (INT8)")
        else:
            print(f"  FP32 model detected")

        # MQTT client
        self.client = mqtt.Client(client_id=f"inference_{truck_id}")
        self.broker = broker
        self.port = port

        # The optional second broker represents the cellular operations-centre
        # uplink. Alerts stay in SQLite until this client acknowledges them.
        self.ops_client = None
        if ops_broker:
            self.ops_client = mqtt.Client(client_id=f"uplink_{truck_id}")
            self.ops_client.on_connect = self.on_ops_connect
            self.ops_client.on_disconnect = self.on_ops_disconnect
            try:
                self.ops_client.connect_async(ops_broker, ops_port, keepalive=60)
                self.ops_client.loop_start()
                print(f"[UPLINK] Operations broker configured at {ops_broker}:{ops_port}")
            except OSError as error:
                print(f"[UPLINK] Offline; alerts will remain queued: {error}")

    def on_ops_connect(self, client, userdata, flags, rc):
        self.ops_connected = (rc == 0)
        if self.ops_connected:
            print("[UPLINK] Cellular connection restored; synchronising alerts")
            self.sync_pending_alerts()

    def on_ops_disconnect(self, client, userdata, rc):
        self.ops_connected = False
        print("[UPLINK] Cellular connection unavailable; alerts remain queued locally")

    def sync_pending_alerts(self):
        """Publish queued alerts and mark only acknowledged messages as synced."""
        if not self.ops_client or not self.ops_connected:
            return

        synced = 0
        for alert_id, topic, payload in self.alert_store.pending():
            message = self.ops_client.publish(topic, payload, qos=1)
            message.wait_for_publish(timeout=5.0)
            if message.is_published():
                self.alert_store.mark_synced(alert_id)
                synced += 1
            else:
                break
        if synced:
            print(f"[UPLINK] Synchronised {synced} queued alert(s)")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[INFERENCE] Connected to MQTT broker at {self.broker}:{self.port}")
            # Subscribe to feature vectors
            client.subscribe(f"{self.topic_base}/features/window", qos=0)
            print(f"[INFERENCE] Subscribed to {self.topic_base}/features/window")
        else:
            print(f"[INFERENCE] Connection failed: rc={rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        features = payload.get("features")
        if features is None:
            return

        # Run inference
        result = self.run_inference(np.array(features, dtype=np.float32))
        if result is None:
            return

        class_id, confidence, class_name, normal_confidence = result
        self.inference_count += 1
        self.confidence_scores.append(confidence)

        # Publish inference result
        result_payload = json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": round(float(confidence), 4),
            "normal_confidence": round(float(normal_confidence), 4),
            "inference_count": self.inference_count
        })
        client.publish(
            f"{self.topic_base}/inference",
            result_payload, qos=1
        )

        # Print result
        status_marker = "  " if class_id == 0 else "⚠ " if class_id == 1 else "🚨"
        print(f"  {status_marker} [{self.inference_count:>4}] "
              f"Class={class_id} ({class_name}) "
              f"Confidence={confidence:.4f}")

        # Publish alert for Warning or Critical
        if class_id >= 1:
            alert_payload = json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "truck_id": self.truck_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": round(float(confidence), 4),
                "action": "Intervention within 2 hours" if class_id == 1 else "IMMEDIATE ESCALATION"
            })
            alert_topic = f"{self.topic_base}/alerts/anomaly"
            # Persist before publishing so a process or cellular failure cannot
            # lose a safety-critical event. Local QoS 2 drives the truck alarm.
            self.alert_store.append(alert_topic, alert_payload)
            client.publish(alert_topic, alert_payload, qos=2)
            self.sync_pending_alerts()

    def run_inference(self, features):
        """Run TFLite inference on a single feature vector."""
        try:
            input_data = features.reshape(1, -1)

            if self.is_quantised:
                # Quantise input to INT8
                input_data = (input_data / self.input_scale + self.input_zero_point).astype(np.int8)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            if self.is_quantised:
                # Dequantise output
                output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale

            # Softmax output — get class and confidence
            output = output.flatten()
            # Apply softmax if output isn't already probabilities
            if np.any(output < 0) or np.sum(output) < 0.5:
                exp_output = np.exp(output - np.max(output))
                output = exp_output / np.sum(exp_output)

            class_id = np.argmax(output)
            confidence = float(output[class_id])
            normal_confidence = float(output[0])
            class_name = CLASS_NAMES[class_id]

            return class_id, confidence, class_name, normal_confidence

        except Exception as e:
            print(f"[INFERENCE ERROR] {e}")
            return None

    def start(self):
        """Start the inference service."""
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            self.client.connect(self.broker, self.port, keepalive=60)
        except ConnectionRefusedError:
            print(f"[ERROR] Cannot connect to MQTT at {self.broker}:{self.port}")
            return

        print(f"[INFERENCE] Service started for truck {self.truck_id}")
        print(f"[INFERENCE] Waiting for feature vectors...")
        self.client.loop_forever()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        if self.ops_client:
            self.ops_client.loop_stop()
            self.ops_client.disconnect()
        self.alert_store.close()


def main():
    parser = argparse.ArgumentParser(description="LogiEdge Inference Service")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--truck-id", type=str, default=TRUCK_ID)
    parser.add_argument("--broker", type=str, default=MQTT_BROKER)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--ops-broker", type=str, default=OPS_MQTT_BROKER)
    parser.add_argument("--ops-port", type=int, default=OPS_MQTT_PORT)
    parser.add_argument("--alert-db", type=str, default=ALERT_DB_PATH)
    args = parser.parse_args()

    service = InferenceService(
        model_path=args.model_path,
        truck_id=args.truck_id,
        broker=args.broker,
        port=args.port,
        ops_broker=args.ops_broker,
        ops_port=args.ops_port,
        alert_db_path=args.alert_db
    )

    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
        print(f"\n[INFERENCE] Stopped after {service.inference_count} inferences.")


if __name__ == "__main__":
    main()
