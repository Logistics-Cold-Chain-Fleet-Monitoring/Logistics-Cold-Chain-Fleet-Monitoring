# Task A1 — Constraint Analysis: FreightBridge Cold-Chain Deployment

## 1. Latency

A refrigeration unit failure can push cargo temperature up by 1°C per minute. So if we need to detect and alert within 90 seconds of a fault showing up in sensor data, we basically have very little room for delays. Now, India's rural cellular round-trip latency is typically between 200–500 ms under good conditions, but on routes like Nashik–Aurangabad where FreightBridge operates, you're looking at much worse — often 800 ms to 2+ seconds because of congestion and weak signal in rural Maharashtra and AP hill regions. And that's when you even have signal at all.

If we do the math: with sensor data coming in at 1 Hz (temperature) and 500 Hz (vibration), and our preprocessing window being 30 seconds with a 10-second step, we need to run inference every 10 seconds. That gives us a per-inference latency budget of roughly 10 seconds, which sounds generous — but the problem is the 90-second end-to-end detection requirement. If we send data to the cloud, the round-trip time alone eats into that budget, and during connectivity blackouts (which happen regularly, more on that below), cloud inference simply doesn't work at all. The cargo keeps warming up and nobody knows.

With edge inference on a Raspberry Pi 5, our model runs in under 15 ms. That means from the moment a fault signature appears in sensor data, we can detect it within one preprocessing window cycle (10 seconds worst case), well within the 90-second SLA. Cloud inference can't guarantee this, especially in rural India — so edge is the only option that meets the latency requirement reliably.

## 2. Bandwidth

Let's calculate the raw data volume per truck per day:

- **Temperature**: 1 Hz × 4 bytes (float32) = 4 B/s → 345.6 KB/day
- **Vibration (3-axis)**: 500 Hz × 3 axes × 4 bytes = 6,000 B/s → 518.4 MB/day
- **Door events**: Discrete, negligible (maybe a few KB/day)

**Total raw data per truck per day ≈ 518.75 MB/day**

At ₹0.10/MB, that's **₹51.88 per truck per day**, which across 85 trucks is **₹4,409/day or ₹1,32,278/month** — just for raw data transmission. And that's assuming perfect connectivity, which we definitely don't have.

With edge processing, we only transmit alert messages and periodic status summaries. An alert message is maybe 200 bytes (JSON with timestamp, class label, confidence score, truck ID). Even if we send 100 alerts per day plus hourly status pings (24 × 500 bytes), that's about **32 KB per truck per day**. At ₹0.10/MB, that's basically **₹0.003 per truck per day** — a cost reduction of over **99.99%**.

So edge processing turns an unaffordable ₹1.3 lakh/month data bill into essentially nothing. That alone justifies the approach financially.

## 3. Connectivity

FreightBridge's Nashik–Aurangabad route has **seven documented locations** where cellular signal drops for **35–90 minutes** each. That's potentially 4–10.5 hours of no connectivity on a single route. During these gaps, a cloud-only system would be completely blind — no inference, no alerts, no monitoring. The truck is carrying temperature-sensitive vaccines and medicines, and the system just... stops working. That's exactly what caused the ₹28 lakh vaccine spoilage incident.

Our edge architecture handles this properly:
- **During connectivity gaps**: The edge device continues running inference locally. All sensor data gets processed on-device, and if any anomaly is detected, the alert gets logged locally in a SQLite database with full timestamps.
- **When connectivity returns**: The device syncs all buffered alerts and status logs to the operations centre via MQTT with QoS 1 (at-least-once delivery). This guarantees no alert is ever lost, even if the truck was offline for hours.
- **Store-and-forward pattern**: The local MQTT broker (Mosquitto) on the edge device acts as a message buffer. Messages queue up during offline periods and get forwarded automatically when the cellular link is back.

This way, the operations team always gets complete visibility — just with a slight delay during the offline periods. But the critical thing is that **detection and local alerting never stops**, even without internet.

## 4. Privacy

FreightBridge's pharmaceutical clients (hospitals, PHCs, retail pharmacies) require contractual proof that cargo condition data can't be accessed by unauthorised third parties. This is especially important because:

- **Regulatory compliance**: Under India's DPDPA 2023, shipment data tied to pharmaceutical supply chains is sensitive. Sending it to a third-party cloud server introduces data processor liability and requires explicit consent mechanisms.
- **Client contracts**: District hospitals and PHCs have strict data handling requirements. If temperature and location data for vaccine shipments were to end up on a cloud server (even an encrypted one), FreightBridge would need to prove that server meets all the clients' security requirements — business associate agreements, audit rights, etc.

With on-device inference, **raw sensor data never leaves the truck**. Only processed alerts (class labels and confidence scores) are transmitted to the operations centre, which is FreightBridge's own backend. The raw vibration waveforms, detailed temperature logs, and door event timings stay on the edge device's local storage. This makes the system **privacy-by-design** — there's simply no pathway for unauthorised access to detailed cargo condition data because it doesn't travel over any external network.

This also simplifies FreightBridge's contracts with pharma clients: they can demonstrate that their monitoring system architecturally prevents raw data exposure, rather than relying on policy-level assurances about cloud security.
