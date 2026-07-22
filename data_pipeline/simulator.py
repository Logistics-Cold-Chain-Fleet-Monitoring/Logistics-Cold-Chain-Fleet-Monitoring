"""
LogiEdge Sensor Simulator (Task C1)
Generates realistic cold-chain truck sensor data with three streams:
  - temperature (1 Hz)
  - vibration_rms (0.5 Hz)
  - door_event (discrete)

Publishes to local Mosquitto broker on localhost.

Usage:
    python simulator.py --anomaly none --duration 1200 --truck-id TRUCK_01
    python simulator.py --anomaly temp_drift --duration 900
    python simulator.py --anomaly vibration --duration 900
    python simulator.py --anomaly combined --duration 900
"""

import argparse
import json
import time
import random
import numpy as np
import paho.mqtt.client as mqtt

# --- Default parameters from the assignment spec ---
TEMP_SETPOINT = 4.0        # °C
TEMP_NORMAL_STD = 0.3      # °C
TEMP_DRIFT_RATE = 0.08     # °C per reading (for temp_drift anomaly)

VIB_NORMAL_MEAN = 0.45     # g
VIB_NORMAL_STD = 0.05      # g
VIB_ANOMALY_MEAN = 1.2     # g (bearing wear)
VIB_ANOMALY_STD = 0.15     # g

MQTT_BROKER = "localhost"
MQTT_PORT = 1883


def create_parser():
    parser = argparse.ArgumentParser(description="LogiEdge Sensor Simulator")
    parser.add_argument(
        "--anomaly",
        type=str,
        choices=["none", "temp_drift", "vibration", "combined"],
        default="none",
        help="Anomaly mode: none, temp_drift, vibration, or combined"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1200,
        help="Duration of simulation in seconds (default: 1200 = 20 min)"
    )
    parser.add_argument(
        "--truck-id",
        type=str,
        default="TRUCK_01",
        help="Truck identifier for MQTT topics"
    )
    return parser


def generate_temperature(tick, anomaly_mode, drift_start_tick=0):
    """Generate a temperature reading based on the current mode."""
    base_temp = np.random.normal(TEMP_SETPOINT, TEMP_NORMAL_STD)

    if anomaly_mode in ("temp_drift", "combined"):
        # Linear drift: +0.08°C per reading from the drift start
        drift = TEMP_DRIFT_RATE * (tick - drift_start_tick)
        base_temp += drift

    return round(base_temp, 3)


def generate_vibration(anomaly_mode):
    """Generate a vibration RMS reading based on the current mode."""
    if anomaly_mode in ("vibration", "combined"):
        # Step change to bearing wear signature
        value = np.random.normal(VIB_ANOMALY_MEAN, VIB_ANOMALY_STD)
    else:
        value = np.random.normal(VIB_NORMAL_MEAN, VIB_NORMAL_STD)

    return round(max(0.0, value), 4)


def generate_door_event():
    """Generate a random door event (low probability)."""
    if random.random() < 0.005:  # ~0.5% chance per second
        event_type = random.choice(["OPEN", "CLOSE"])
        return event_type
    return None


def main():
    parser = create_parser()
    args = parser.parse_args()

    anomaly_mode = args.anomaly
    duration = args.duration
    truck_id = args.truck_id

    # MQTT topic base
    topic_base = f"logibridge/trucks/{truck_id}/sensors"

    # Connect to MQTT broker
    client = mqtt.Client(client_id=f"simulator_{truck_id}")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except ConnectionRefusedError:
        print(f"[ERROR] Cannot connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        print("Please start Mosquitto: mosquitto -v")
        return

    client.loop_start()

    print(f"[SIMULATOR] Starting sensor simulation for {truck_id}")
    print(f"[SIMULATOR] Anomaly mode: {anomaly_mode}")
    print(f"[SIMULATOR] Duration: {duration} seconds")
    print(f"[SIMULATOR] Publishing to {topic_base}/...")

    tick = 0
    vib_tick = 0
    start_time = time.time()

    try:
        while tick < duration:
            current_time = time.time()
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(current_time))

            # --- Temperature stream (1 Hz) ---
            temp_value = generate_temperature(tick, anomaly_mode)
            temp_payload = json.dumps({
                "timestamp": timestamp,
                "value": temp_value,
                "unit": "celsius",
                "tick": tick
            })
            client.publish(f"{topic_base}/temperature", temp_payload, qos=0)

            # --- Vibration stream (0.5 Hz — every 2 seconds) ---
            if tick % 2 == 0:
                vib_value = generate_vibration(anomaly_mode)
                vib_payload = json.dumps({
                    "timestamp": timestamp,
                    "value": vib_value,
                    "unit": "g",
                    "tick": vib_tick
                })
                client.publish(f"{topic_base}/vibration", vib_payload, qos=0)
                vib_tick += 1

            # --- Door events (discrete, random) ---
            door_event = generate_door_event()
            if door_event:
                door_payload = json.dumps({
                    "timestamp": timestamp,
                    "event": door_event,
                    "tick": tick
                })
                client.publish(f"{topic_base}/door", door_payload, qos=1)
                print(f"  [DOOR] {door_event} at tick {tick}")

            # Progress logging
            if tick % 60 == 0:
                print(f"  [TICK {tick:>5}] temp={temp_value:.2f}°C, "
                      f"vib={'--' if tick % 2 != 0 else f'{vib_value:.3f}g'}, "
                      f"mode={anomaly_mode}")

            tick += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[SIMULATOR] Stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        elapsed = time.time() - start_time
        print(f"[SIMULATOR] Done. {tick} ticks in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
