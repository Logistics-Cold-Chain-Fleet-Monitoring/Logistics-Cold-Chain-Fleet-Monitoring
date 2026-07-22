# Task B1 — Constraint Triangle Application & Task B2 — Arithmetic Intensity and Roofline Analysis

## Task B1: Constraint Triangle Application

### The Three Hardware Options

| Option | Hardware | India Price | TDP |
|--------|----------|-------------|-----|
| 1 | Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS Hailo-8L) | ~₹15,000/truck | 7.5W |
| 2 | Jetson Orin Nano Super Developer Kit (67 TOPS) | ~₹45,000/truck | 15W moderate load |
| 3 | STM32H7-based custom MCU with sensor ICs | ~₹3,500/truck | 0.4W |

### Constraint Triangle Vertices for Cold-Chain Deployment

The three vertices are **Compute Performance**, **Power Budget**, and **Unit Cost**. Let me apply each to our FreightBridge scenario:

**Compute Performance**: Our model is a small 2-hidden-layer MLP (32 and 16 units) doing 45 MFLOPs per inference. This is a tiny workload — we're not running YOLOv8 or a transformer here. Even a Raspberry Pi 5 CPU can handle this in under 15 ms. So compute performance is NOT the dominant constraint. All three options can meet the 90-second SLA easily.

**Power Budget**: The trucks run on a 12V supply with a 10W AI power budget (via DC-DC converter). Option 3 (STM32H7 at 0.4W) easily fits. Option 1 (RPi 5 + HAT at 7.5W) fits within the 10W budget. Option 2 (Jetson Orin Nano at 15W) **exceeds the 10W power budget** — this is a hard fail. You'd need a beefier DC-DC converter and the thermal management in a truck cabin (which can hit 50°C in Indian summers) becomes a real issue.

**Unit Cost**: For the pilot, we need 85 trucks. At full scale, 265 vehicles.
- Option 1: 85 × ₹15,000 = ₹12,75,000 (pilot) | 265 × ₹15,000 = ₹39,75,000 (full scale)
- Option 2: 85 × ₹45,000 = ₹38,25,000 (pilot) | 265 × ₹45,000 = ₹1,19,25,000 (full scale)
- Option 3: 85 × ₹3,500 = ₹2,97,500 (pilot) | 265 × ₹3,500 = ₹9,27,500 (full scale)

### Dominant Constraint Vertex

The **dominant constraint** for this deployment is **Unit Cost**, closely followed by Power Budget. Here's why: FreightBridge is a mid-sized logistics company doing a pilot with explicit plans to scale to 265 vehicles. The difference between Option 1 and Option 2 at full scale is ₹80 lakh — that's a massive cost gap for what is essentially a 6-feature MLP inference task.

### My Recommendation: Option 1 — Raspberry Pi 5 + AI HAT+

**Why Option 1 wins:**
- **Meets the 90-second latency SLA**: Our MLP model runs in <15 ms on the RPi 5 CPU alone. The Hailo-8L NPU is there as headroom for future model upgrades (e.g., if we move to CNN-based vibration analysis at scale).
- **Fits the 10W power budget**: At 7.5W total system draw, it stays under the 10W AI budget with margin.
- **Reasonable fleet cost**: ₹39.75 lakh for 265 trucks is a justifiable investment given the ₹28 lakh that one single spoilage event cost.
- **Runs full Linux**: This means we can use Docker containers, Python ML stack, MQTT broker, SQLite — the complete software ecosystem needed for the inference pipeline and store-and-forward architecture.
- **OTA updates via Docker**: We can push model updates over cellular without physical truck visits.

**Why Option 2 (Jetson Orin Nano) loses:**
- **Exceeds the 10W power budget**: At 15W under moderate load, it violates the stated constraint.
- **Massively over-specced**: 67 TOPS for a 45 MFLOP model? That's like using a bulldozer to plant a flower pot. We're using less than 0.0001% of its compute capacity.
- **3× the cost of Option 1**: ₹45,000 vs ₹15,000 per truck. At 265 trucks, we're spending an extra ₹80 lakh for compute we'll never use.
- **Thermal concerns**: 15W in an Indian truck cabin in summer with limited ventilation is asking for thermal throttling.

**Why Option 3 (STM32H7 MCU) loses:**
- **No Linux, no Docker**: The STM32H7 runs bare-metal or RTOS. We can't run Mosquitto, can't containerise the inference service, can't do OTA updates via Docker layer caching. The entire MLOps pipeline (drift monitoring, Ansible deployment, Docker OTA) becomes impossible.
- **Extremely limited RAM**: The STM32H7 has about 1 MB SRAM. Our preprocessing pipeline with sliding windows, feature buffers, and model weights needs more working memory than this MCU can offer comfortably.
- **No store-and-forward capability**: Without a proper filesystem and MQTT broker, handling the 35–90 minute connectivity gaps with proper message queuing becomes a huge software engineering effort on bare metal.
- **Development cost**: Building and debugging a TFLite Micro pipeline on bare metal takes significantly more engineering time than deploying on Linux with standard Python tools. The ₹11,500/truck savings is eaten up by increased development time.

---

## Task B2: Arithmetic Intensity and Roofline Analysis

### Given Data
- **Model computation**: ~45 MFLOPs per inference
- **Data accessed per inference**: ~18 MB (weights + activations)
- **Raspberry Pi 5 CPU**: ~16 GFLOP/s (NEON SIMD)
- **Memory bandwidth**: 12 GB/s (LPDDR4X)

### Step 1: Calculate Arithmetic Intensity

$$AI = \frac{\text{FLOPs}}{\text{Bytes Accessed}} = \frac{45 \times 10^6}{18 \times 10^6} = 2.5 \text{ FLOP/Byte}$$

### Step 2: Find the Ridge Point

The ridge point is where the compute ceiling meets the memory bandwidth ceiling:

$$\text{Ridge Point} = \frac{\text{Peak Compute (FLOP/s)}}{\text{Peak Bandwidth (B/s)}} = \frac{16 \times 10^9}{12 \times 10^9} = 1.33 \text{ FLOP/Byte}$$

### Step 3: Classify the Model

Our model's Arithmetic Intensity (2.5 FLOP/Byte) is **greater than** the ridge point (1.33 FLOP/Byte).

This means our model is **compute-bound**, not memory-bandwidth-bound.

### Step 4: What Does This Mean?

Since the model sits to the **right of the ridge point** on the Roofline chart, the performance ceiling is determined by the CPU's peak compute capacity, not by memory bandwidth. The achievable throughput is:

$$\text{Max Throughput} = \text{Peak Compute} = 16 \text{ GFLOP/s}$$

$$\text{Inference Time} = \frac{45 \times 10^6 \text{ FLOPs}}{16 \times 10^9 \text{ FLOP/s}} = 2.81 \text{ ms}$$

This is way under the 90-second SLA. Even at realistic sustained throughput (say 50–70% of peak, so ~8–11 GFLOP/s), we'd get 4–5.6 ms per inference. Still very comfortable.

### Step 5: Optimisation Recommendation

Since the model is **compute-bound**, the optimisation that would improve latency is **reducing the number of FLOPs** through:

1. **INT8 Quantisation**: Converting from FP32 to INT8 reduces the effective compute required because NEON SIMD can process 4× more INT8 operations per cycle compared to FP32. This could bring us from 45 MFLOPs (FP32) down to effectively ~11.25 MFLOPs equivalent throughput.

2. **Structured Pruning**: Removing 35% of filters reduces the FLOPs by approximately 35%, bringing the model down to ~29 MFLOPs.

3. **Combining both (M3 variant)**: Pruning + INT8 quantisation would give us the maximum FLOP reduction, bringing inference time well under 2 ms.

Note: Since we're compute-bound, optimising memory access patterns (like weight compression or activation tiling) would NOT help much here — the bottleneck is raw compute, not data movement. We need to make the computation itself smaller or use hardware that can do more FLOP/s (like offloading to the Hailo-8L NPU on the AI HAT+).
