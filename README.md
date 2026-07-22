# LogiEdge: Intelligent Edge AI Platform for Cold-Chain Logistics

> **New to this project?** Start with [GETTING_STARTED.md](GETTING_STARTED.md). It explains the system in plain language and provides separate offline, live MQTT, Docker, training, and deployment instructions.

## Project Overview

LogiEdge is an Edge AI pipeline built for **FreightBridge Logistics Pvt. Ltd.** to do real-time cold-chain monitoring on their 85 refrigerated trucks. The system watches cargo temperature, refrigeration unit vibration, and door open/close events — and classifies the compartment state into three classes (Normal, Warning, Critical) right on the truck's edge device. It works fully offline when there's no cell signal and syncs up with the operations centre once connectivity comes back.

## Why We Need This

FreightBridge has had some serious problems recently:
- A ₹28 lakh vaccine spoilage because a refrigeration unit failed and nobody caught it
- A pharma shipment got rejected at a hospital because temperature records were incomplete
- Two trucks broke down mid-route from engine issues that sensor data could've flagged 3 days earlier

The whole point of LogiEdge is to catch these issues in real-time, right on the truck, without needing to be connected to the cloud 24/7.

## Repository Structure

```
logibridge/
├── README.md
├── scenario_architecture/
│   ├── constraint_analysis.md        # Task A1: Four Edge AI constraints analysis
│   └── system_architecture.png       # Task A2: System architecture diagram
├── hardware/
│   └── hardware_justification.md     # Task B1 & B2: Constraint Triangle + Roofline
├── data_pipeline/
│   ├── simulator.py                  # Task C1: Sensor simulator with MQTT
│   ├── preprocessing.py              # Task C2: Filtering, feature extraction, normalization
│   ├── training_stats.npy            # Generated normalization stats
│   └── mqtt_architecture.md          # Task C3: Data fusion & MQTT design
├── training/
│   ├── generate_dataset.py           # Task D1: Generate labelled training data
│   ├── train_model.py                # Task D1: Train MLP classifier
│   ├── convert_ptq.py                # Task F1: Post-Training Quantisation (M2)
│   ├── prune_quantise.py             # Task F1: Structured Pruning + PTQ (M3)
│   ├── normalisation_experiment.py   # Task C2: Mandatory normalisation experiment
│   ├── pipeline_mapping.md           # Task D3: 10-stage pipeline mapping
│   └── models/                       # Saved model files
├── inference/
│   ├── Dockerfile                    # Task D2: Docker containerisation
│   ├── inference_service.py          # Task D2: Inference + MQTT publish
│   └── model.tflite                  # Deployed model
├── monitoring/
│   ├── drift_monitor.py              # Task E1: PSI drift monitoring
│   └── reference_dist.json           # Reference confidence distribution
├── deployment/
│   ├── logibridge_deploy.yml         # Task E2: Ansible deployment playbook
│   └── ota_strategy.md               # Task E3: OTA strategy selection
├── optimisation/
│   ├── benchmark.py                  # Task F2: Five-metric benchmarking
│   └── results/
│       ├── benchmark_results.csv     # 15-cell results table
│       └── pareto_chart.png          # Pareto frontier chart
├── demo/
│   └── demo_video_link.txt           # Link to demo video (15-20 min)
└── reports/
  ├── phase1_report.md             # Phase 1 report source
  ├── phase2_report.md             # Phase 2 report source
  ├── final_report.md              # Final report source
  └── Group10_LogiEdge_Final.pdf   # LMS submission PDF
```

## Classification Targets

| Class | Label    | Condition |
|-------|----------|-----------|
| 0     | Normal   | Temperature within ±1°C of setpoint; no vibration anomaly; refrigeration unit operating correctly |
| 1     | Warning  | Temperature drifting (1–3°C outside setpoint) OR early refrigeration anomaly; intervention needed within 2 hours |
| 2     | Critical | Temperature breach (>3°C outside setpoint) OR refrigeration unit failure signature; immediate escalation required |

## How to Run

> **All commands below assume you are in the `logibridge/` project root directory.**

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | `python --version` to verify |
| pip | latest | `python -m pip install --upgrade pip` |
| Mosquitto | 2.x | MQTT broker — needed only for live inference (Step 7) |
| Docker | 24+ | Needed only for containerised inference (Step 8) |
| Ansible | 2.15+ | Needed for the seven-task deployment demonstration |

Before the Ansible demonstration, install the Docker collection with `ansible-galaxy collection install community.docker`. Run the playbook against a Linux edge node because Ansible control and the target Docker service are not provided by this Windows workspace.

### Step 0: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `tensorflow`, `tensorflow-model-optimization`, `numpy`, `scipy`, `scikit-learn`, `paho-mqtt`, `matplotlib`, `psutil`.

> **Note**: On TensorFlow 2.19+, if you see a `CreateWrapperFromFile` error during TFLite verification, install the standalone interpreter:
> ```bash
> pip install ai_edge_litert
> ```

---

### Step 1: Generate Training Dataset (Task C1 + D1)

```bash
python training/generate_dataset.py
```

**What it does:**
- Runs the sensor simulator internally in three modes (Normal 20 min, Warning 15 min, Critical 15 min)
- Applies 5-sample moving average, extracts 6 features per 30-second window (10-second step)
- Computes z-score normalisation stats from the first 10 minutes of Normal-class data
- Saves the labelled dataset

**Outputs:**
| File | Description |
|------|-------------|
| `training/models/dataset.npz` | Feature vectors (`X`, `y`, `X_normalised`) + stats |
| `data_pipeline/training_stats.npy` | Frozen mean/std for deployment normalisation |

**Expected console output:** Feature statistics per class, total sample count (~294 samples across 3 classes).

---

### Step 2: Train the MLP Classifier (Task D1)

```bash
python training/train_model.py
```

**What it does:**
- Loads `dataset.npz`, splits 80/20 train/val (stratified, seed=0)
- Trains a 2-hidden-layer MLP: Input(6) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(3, Softmax)
- Uses class-weighted loss (3× boost for Critical class) to ensure ≥95% Critical recall
- Runs up to 100 epochs with EarlyStopping (patience=15, restores best weights)
- Validates: accuracy ≥ 88%, Class 2 recall ≥ 95%

**Outputs:**
| File | Description |
|------|-------------|
| `training/models/logibridge_fp32.keras` | Keras model (M1 baseline) |
| `training/models/logibridge_fp32_saved/` | SavedModel directory (for TFLite conversion) |
| `training/models/model_fp32.tflite` | TFLite FP32 model (M1) |
| `inference/model.tflite` | Copy for inference container |
| `training/models/val_data.npz` | Validation set for benchmarking |
| `training/models/training_history.npy` | Training history (loss/accuracy curves) |

**Expected results:** ~98% accuracy, 100% Critical recall, confusion matrix printed to console.

---

### Step 3: Convert to INT8 — Post-Training Quantisation (Task F1, M2 variant)

```bash
python training/convert_ptq.py
```

**What it does:**
- Loads the FP32 SavedModel from `logibridge_fp32_saved/`
- Applies full INT8 quantisation (weights + activations) using 200+ calibration samples
- Verifies accuracy on validation set

**Outputs:**
| File | Description |
|------|-------------|
| `training/models/model_ptq_int8.tflite` | M2 — Full INT8 quantised model |

**Expected:** Model size drops from ~5.3 KB to ~4.6 KB (13% reduction).

---

### Step 4: Pruning + Quantisation (Task F1, M3 variant)

```bash
python training/prune_quantise.py
```

**What it does:**
- Rebuilds the MLP architecture, loads weights from the Keras 3 model
- Applies 35% structured filter pruning with polynomial decay over 50 fine-tuning epochs
- Strips pruning wrappers, converts to full INT8 TFLite
- Prints accuracy comparison across M1/M2/M3

**Outputs:**
| File | Description |
|------|-------------|
| `training/models/logibridge_pruned.keras` | Pruned Keras model |
| `training/models/logibridge_pruned_saved/` | Pruned SavedModel directory |
| `training/models/model_pruned_int8.tflite` | M3 — Pruned + INT8 model |

**Expected:** ~98% accuracy retained, smallest model (~4.5 KB).

---

### Step 5: Five-Metric Benchmarking + Pareto Chart (Task F2 + F3)

```bash
python optimisation/benchmark.py
```

**What it does:**
- Benchmarks all 3 variants (M1, M2, M3) on 5 metrics:
  1. Mean inference latency (200 runs, 10 warmup)
  2. p95 inference latency
  3. Model file size (KB)
  4. Validation accuracy (%)
  5. Energy per inference (mJ, estimated from latency × TDP)
- Generates a Pareto frontier chart
- Prints deployment recommendation with reasoning

**Outputs:**
| File | Description |
|------|-------------|
| `optimisation/results/benchmark_results.csv` | 3×5 results table |
| `optimisation/results/pareto_chart.png` | Pareto frontier visualisation |

**Expected recommendation:** M1 (FP32 baseline) — highest measured accuracy, 100% Critical recall, and negligible storage cost on the Raspberry Pi 5.

---

### Step 6: Normalisation Experiment (Task C2 — Mandatory)

```bash
python training/normalisation_experiment.py
```

**What it does:**
- Tests model robustness to normalisation parameter corruption:
  1. **Correct stats** — baseline (expect ~98% accuracy)
  2. **+3σ shifted mean** — mild drift (expect ~96% accuracy)
  3. **+10σ shifted mean** — severe drift (expect ~96% accuracy)
  4. **No normalisation** — mean=0, std=1 (expect ~64% accuracy, 0% Critical recall)
- Prints comparison table with per-class recall

**Key takeaway:** Without correct normalisation, Critical recall drops to 0% — catastrophic for cold-chain safety.

---

### Step 7: Drift Monitoring with PSI (Task E1)

#### 7a. Build Reference Distribution

```bash
python monitoring/drift_monitor.py --mode build-reference
```

**What it does:** Generates 1 hour of clean Normal-class data, runs inference on 300 windows, saves the reference confidence distribution.

**Output:** `monitoring/reference_dist.json`

#### 7b. Run Drift Demo (Injection + Recovery)

```bash
python monitoring/drift_monitor.py --mode demo
```

**What it does:**
1. **Phase 1 (Clean):** 118 Normal inferences → PSI ≈ 0.087 (no drift)
2. **Phase 2 (Drift):** Injects 5 minutes of combined anomaly → PSI ≈ 26.815 > 0.25 → **DRIFT ALERT**
3. **Phase 3 (Recovery):** Returns to clean data → PSI ≈ 0.046 (recovered)

#### 7c. Live Monitor Mode (requires MQTT broker)

```bash
# Terminal 1: Start Mosquitto broker
mosquitto -v

# Terminal 2: Start preprocessing
python data_pipeline/preprocessing.py --truck-id TRUCK_01

# Terminal 3: Start inference
python inference/inference_service.py --model-path training/models/model_fp32.tflite --truck-id TRUCK_01 --alert-db runtime/alerts.db

# Terminal 4: Start drift monitor
python monitoring/drift_monitor.py --mode monitor --truck-id TRUCK_01

# Terminal 5: Start sensor simulator
python data_pipeline/simulator.py --anomaly none --duration 600 --truck-id TRUCK_01
```

---

### Step 8: Containerised Inference (Task D2)

```bash
# Terminal 1: Start Mosquitto broker
docker run --rm --name logibridge-mqtt -p 1883:1883 eclipse-mosquitto:2

# Terminal 2: Build and run inference container
docker build -t logibridge-inference ./inference/
docker run --rm --name logibridge-inference -e MODEL_PATH=/app/model.tflite -e TRUCK_ID=TRUCK_01 -e MQTT_BROKER=host.docker.internal logibridge-inference

# Terminal 3: Start preprocessing on the host
python data_pipeline/preprocessing.py --truck-id TRUCK_01

# Terminal 4: Start sensor simulator to feed data
python data_pipeline/simulator.py --anomaly none --duration 300 --truck-id TRUCK_01
```

**Environment variables for the container:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `model.tflite` | Path to TFLite model inside container |
| `TRUCK_ID` | `TRUCK_01` | Truck identifier for MQTT topics |
| `MQTT_BROKER` | `localhost` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `OPS_MQTT_BROKER` | empty | Optional operations-centre broker; queued alerts sync here after reconnection |
| `OPS_MQTT_PORT` | `1883` | Operations-centre MQTT port |
| `ALERT_DB_PATH` | `/opt/logibridge/alerts.db` | Persistent SQLite store-and-forward queue |

**MQTT output topics:**
- `logibridge/trucks/{truck_id}/inference` (QoS 1) — every inference result
- `logibridge/trucks/{truck_id}/alerts/anomaly` (QoS 2) — Warning/Critical alerts only

---

### Quick Run — Full Pipeline (Steps 1–6 without MQTT)

For a quick end-to-end run without needing Mosquitto:

```bash
python training/generate_dataset.py
python training/train_model.py
python training/convert_ptq.py
python training/prune_quantise.py
python optimisation/benchmark.py
python training/normalisation_experiment.py
python monitoring/drift_monitor.py --mode build-reference
python monitoring/drift_monitor.py --mode demo
```

All offline steps (Steps 1–6 and 7a–7b) run without a broker since the simulator is called internally.


## Tech Stack
- **Edge Device**: Raspberry Pi 5 (8 GB) + AI HAT+ (Hailo-8L)
- **Messaging**: MQTT via Mosquitto broker
- **ML Framework**: TensorFlow / TensorFlow Lite
- **Containerisation**: Docker (python:3.11-slim)
- **Deployment**: Ansible
- **Language**: Python 3.11
