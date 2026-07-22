# MQTT Architecture & Data Fusion Justification

## MQTT Topic Tree

```
logibridge/
├── trucks/
│   └── {truck_id}/
│       ├── sensors/
│       │   ├── temperature        # Raw temperature readings (1 Hz)
│       │   ├── vibration          # Raw vibration RMS readings (0.5 Hz)
│       │   └── door               # Door open/close events (discrete)
│       ├── features/
│       │   └── window             # Preprocessed 6-value feature vectors
│       ├── inference              # Model inference results (class + confidence)
│       ├── alerts/
│       │   ├── anomaly            # Warning/Critical class detections
│       │   └── drift              # PSI drift alerts
│       └── status/
│           ├── heartbeat          # Periodic device health check
│           └── sync               # Buffered alerts synced after reconnection
└── fleet/
    └── ops_centre/
        ├── dashboard              # Aggregated fleet status
        └── commands               # OTA update commands from ops centre
```

## QoS Justification Per Topic

| Topic | QoS Level | Justification |
|-------|-----------|---------------|
| `sensors/temperature` | QoS 0 (At most once) | High-frequency data that's processed locally. Losing a single reading out of 86,400/day doesn't affect the 30-second sliding window. No need to waste bandwidth on acknowledgements. |
| `sensors/vibration` | QoS 0 (At most once) | Same reasoning as temperature. At 0.5 Hz, a missed sample barely affects the window's RMS/kurtosis calculations. |
| `sensors/door` | QoS 1 (At least once) | Door events are discrete and infrequent. Each one matters for the chain-of-custody documentation that pharma clients need. Can't afford to miss one. |
| `features/window` | QoS 0 (At most once) | Internal message between preprocessing and inference stages on the same device. Local broker, no network involved, so delivery is essentially guaranteed anyway. |
| `inference` | QoS 1 (At least once) | This is the main output of the system. Every inference result needs to reach the operations centre eventually. QoS 1 with the local broker's persistence ensures store-and-forward during connectivity gaps. |
| `alerts/anomaly` | QoS 2 (Exactly once) | Safety-critical. A missed alert could mean vaccine spoilage. A duplicate could cause unnecessary panic. QoS 2 is justified because alert volume is low (maybe a few per day) so the overhead is acceptable. |
| `alerts/drift` | QoS 1 (At least once) | Important for MLOps but not safety-critical. If the ops team gets a duplicate drift alert, it's just a minor inconvenience. |
| `status/heartbeat` | QoS 0 (At most once) | Regular heartbeats every few minutes. If one is lost, the next one comes soon enough. |
| `fleet/ops_centre/commands` | QoS 1 (At least once) | OTA update commands must reach the truck. But sending the same command twice is fine — the Ansible playbook is idempotent. |

## Data Fusion Justification

### Why Feature-Level Fusion?

We chose **feature-level fusion** for LogiEdge: each sensor stream (temperature, vibration, door) gets its features extracted independently, and then those features are concatenated into a single 6-value joint feature vector before being passed to the MLP model.

The 6 features are:
1. Temperature mean (from temperature stream)
2. Temperature standard deviation (from temperature stream)
3. Temperature rate-of-change in °C/min (from temperature stream)
4. Vibration RMS (from vibration stream)
5. Vibration peak (from vibration stream)
6. Vibration kurtosis (from vibration stream)

### Why Not Data-Level Fusion?

Data-level fusion means combining the raw sensor readings before doing any feature extraction — basically concatenating raw temperature samples, raw vibration samples, and door events into one big input.

The problem with this for our system:
- **Different sampling rates**: Temperature is at 1 Hz, vibration at 0.5 Hz, door events are discrete. You'd need to resample everything to a common rate, which either upsamples the slow sensors (adding fake data) or downsamples the fast ones (losing information).
- **Different units and scales**: Temperature is in °C (range ~2–10), vibration is in g (range ~0.3–1.5), door events are binary. Throwing these raw values together doesn't make physical sense.
- **Huge input vector**: Over a 30-second window, you'd have 30 temperature samples + 15 vibration samples + a few door events = variable-length input. The MLP needs a fixed-size input.
- **More compute**: The model would need to learn the feature extraction itself, requiring a larger network — which defeats the purpose of running on an edge device with compute constraints.

### Why Not Decision-Level Fusion?

Decision-level fusion means training separate models for each sensor stream and then combining their predictions (e.g., majority voting or confidence-weighted averaging).

The problem with this for our system:
- **Loses cross-sensor correlations**: The whole point of our classification is to catch patterns like "temperature is drifting AND vibration is high" (the combined/Critical class). With separate models, each model only sees its own sensor. The temperature model might say "Warning" (drift detected) and the vibration model might say "Warning" (high RMS), but neither knows about the other. A voting scheme might still say "Warning" when the correct answer is "Critical".
- **More models to maintain**: Three separate models means 3× the OTA update burden, 3× the drift monitoring, 3× the benchmarking. On a fleet of 85 trucks with M2M SIMs at ₹0.10/MB, this multiplies our deployment complexity and bandwidth costs.
- **Higher total compute**: Running three models sequentially takes more time than running one model on a concatenated feature vector. Each model has its own overhead (TFLite interpreter initialisation, memory allocation).

### Why Feature-Level Fusion Works Best Here

Feature-level fusion hits the sweet spot:
- **Handles different sampling rates naturally**: We extract features per-stream over the same 30-second window, so the output is always 6 values regardless of input rates.
- **Preserves cross-sensor information**: The model sees temperature AND vibration features together, so it can learn the interaction patterns (e.g., rising temperature + high vibration kurtosis = compressor failure).
- **Compact input**: Just 6 values per inference. Our small MLP (32-16 units) handles this efficiently.
- **Domain-meaningful features**: Temperature rate-of-change, vibration RMS, and kurtosis are standard engineering features for cold-chain and rotating machinery monitoring. They're physically interpretable, which helps in debugging and explaining the system to FreightBridge's operations team.
