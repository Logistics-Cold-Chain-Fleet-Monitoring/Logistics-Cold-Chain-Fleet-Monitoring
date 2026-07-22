# Task E3 — OTA Strategy Selection

## Context

FreightBridge updates truck models every **6 weeks**. Model file is **280 KB** (INT8 TFLite). Trucks use **M2M SIM at ₹0.10/MB**. Pilot fleet: **85 trucks**. Full scale: 265 vehicles. Trucks operate on routes with **35–90 minute connectivity gaps** at seven documented locations.

## Three OTA Strategies Evaluated

### Strategy 1: Full Replacement (All 85 trucks simultaneously)

**How it works**: Push the new model to all 85 trucks at once. Each truck downloads the new `model.tflite`, restarts the inference container, and begins using the updated model immediately.

**Bandwidth cost per update cycle:**

- Model file: 280 KB = 0.2734 MB per truck
- Docker layer (model layer only, thanks to layer caching): ~0.3 MB per truck
- Total per truck: ~0.3 MB
- **Total for 85 trucks: 85 × 0.3 = 25.5 MB**
- **Cost: 25.5 × ₹0.10 = ₹2.55 per update cycle**

**Pros:**
- Simple to implement — one Ansible playbook run targets all trucks
- All trucks run the same model version — no version mismatch issues
- Lowest total bandwidth (single push)

**Cons:**
- **Highest risk**: If the new model has a regression (e.g., Class 2 recall drops below 95%), ALL 85 trucks are affected simultaneously. In a pharmaceutical cold-chain, this could mean 85 trucks running with a faulty anomaly detector for hours before anyone notices.
- No validation period before fleet-wide deployment
- Connectivity gaps mean some trucks may get updated hours after others, creating a partial-rollout situation anyway (but without the safety controls of a deliberate canary)

---

### Strategy 2: Canary Deployment (10 trucks first, then remaining 75)

**How it works**: Deploy the new model to 10 designated "canary" trucks first. Monitor their inference accuracy, PSI drift values, and Class 2 recall for 48–72 hours. If all metrics are healthy, roll out to the remaining 75 trucks. If any issues are detected, roll back the canary trucks and investigate.

**Bandwidth cost per update cycle:**

- Phase 1 (canary): 10 trucks × 0.3 MB = 3.0 MB → ₹0.30
- Phase 2 (full rollout): 75 trucks × 0.3 MB = 22.5 MB → ₹2.25
- **Total: 25.5 MB → ₹2.55** (same as full replacement)
- **If rollback needed (worst case)**: 10 × 0.3 MB extra = ₹0.30 for rolling back canary
- **Total worst-case: 28.5 MB → ₹2.85**

**Pros:**
- **Limits blast radius**: If the new model is faulty, only 10 trucks are affected, not 85. With ₹28 lakh at stake per spoilage event, limiting exposure from 85 to 10 trucks reduces maximum financial risk by 88%.
- Provides a real-world validation period with actual cold-chain conditions
- The 48-hour monitoring window gives enough time to verify Class 2 recall on real Critical events (which are rare but safety-critical)
- The canary trucks can include a mix of routes (Nashik–Aurangabad for connectivity gap testing, Pune urban for high-frequency data)

**Cons:**
- Slightly more complex deployment process (two Ansible runs instead of one)
- 48–72 hour delay before full fleet gets the update
- Temporary version mismatch between canary and non-canary trucks (but inference results include model version in metadata, so the ops centre can distinguish)

---

### Strategy 3: Shadow Mode (Run both models, compare outputs)

**How it works**: Deploy the new model alongside the existing model on all 85 trucks. Both models run inference on every feature vector. The existing model's output drives alerts and decisions. The new model's output is logged for comparison. After 1–2 weeks, if the new model's performance matches or exceeds the old one, switch over.

**Bandwidth cost per update cycle:**

- Model file deployed to all 85 trucks: 85 × 0.3 MB = 25.5 MB → ₹2.55
- **Additional shadow inference logging**: Each inference generates ~200 bytes of comparison data. At 6 inferences/minute (one per 10-second window step), that's 72 KB/hour or 1.73 MB/day per truck.
- Shadow logging for 7 days: 85 × 1.73 × 7 = **1,029 MB → ₹102.90**
- **Total: ~1,055 MB → ₹105.45 per update cycle**

**Pros:**
- Zero-risk transition — the production model keeps running throughout
- Statistically rigorous comparison with real-world data over multiple days
- Can detect subtle regression that a 48-hour canary might miss

**Cons:**
- **41× higher bandwidth cost** (₹105 vs ₹2.55) due to shadow logging over cellular M2M SIMs
- **Doubles compute load**: Running two TFLite models per inference window. While our MLP is lightweight (~5 ms), this doubles the energy draw during the shadow period.
- **Not feasible during connectivity gaps**: The 35–90 minute dead zones mean shadow comparison data can't be uploaded in real-time. Data would need to be buffered locally and synced later, requiring additional storage management on the Raspberry Pi.
- **7–14 day delay** before the new model goes live — too slow for an urgent security patch or critical model fix
- **Complexity**: Requires a shadow inference orchestrator, comparison logic, and dashboard — significant engineering effort for a 6-weekly update cycle

---

## Bandwidth Cost Summary

| Strategy | Bandwidth per Update | Cost per Update | Annual Cost (8.7 cycles) |
|----------|---------------------|-----------------|--------------------------|
| Full Replacement | 25.5 MB | ₹2.55 | ₹22.19 |
| Canary (10 → 75) | 25.5–28.5 MB | ₹2.55–₹2.85 | ₹22.19–₹24.80 |
| Shadow Mode | ~1,055 MB | ₹105.45 | ₹917.42 |

## Recommendation: Canary Deployment (Strategy 2)

**We recommend Canary Deployment** for the following reasons:

### 1. Cold-Chain Safety-Criticality
FreightBridge transports pharmaceutical products — vaccines and biologics — where a missed Critical alert can cause ₹28 lakh in spoilage. Full replacement exposes all 85 trucks to an untested model. Canary limits exposure to 10 trucks (12% of the fleet) while providing real-world validation. This is the standard practice in safety-critical systems: test on a small population before fleet-wide rollout.

### 2. Cost-Effectiveness
Canary costs the same as full replacement (₹2.55/cycle) in the normal case. Even with a rollback, the worst case is ₹2.85 — essentially free on M2M SIMs. Shadow mode at ₹105/cycle is 41× more expensive, mostly due to logging bandwidth on cellular connections.

### 3. Rural Connectivity Compatibility
The Nashik–Aurangabad route has 35–90 minute connectivity gaps. Canary deployment only needs to push a 280 KB model file — this can complete in a few seconds of connectivity. Shadow mode needs to continuously upload comparison logs over the same intermittent connections, making it impractical on rural routes.

### Argument Against Full Replacement
Full replacement violates the principle of graduated rollout for safety-critical systems. If a model update introduces a regression in Class 2 (Critical) recall — which our benchmarking showed can happen (M3 variant dropped to 82.35%) — all 85 trucks would be running a compromised detector. The ₹0.30 savings per cycle doesn't justify the risk of fleet-wide failure.

### Argument Against Shadow Mode
Shadow mode is engineering overkill for this deployment. The model update frequency (every 6 weeks) is slow enough that a 48–72 hour canary validation period is acceptable. The 41× bandwidth premium for shadow logging over M2M SIMs is unjustifiable when the model file itself is only 280 KB. Additionally, the Raspberry Pi 5's 8 GB RAM is more than sufficient for our single MLP, but running two models simultaneously with comparison logging adds unnecessary complexity to the edge device's resource management.
