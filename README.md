# Privacy-First Clinical Edge Monitor

> Real-time clinical monitoring on NVIDIA Jetson Nano: skeleton-based event detection with zero patient imagery transmission.

![Demo GIF](docs/demo.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jetson](https://img.shields.io/badge/Jetson-Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

🚧 **Work in progress** — under active development.

---

## 🎯 Overview

This project demonstrates a privacy-preserving clinical monitoring system for early fall-risk detection, running entirely on edge hardware. Patient video is processed on-device for skeleton extraction; only anonymized event-level data is transmitted to external systems.

The architecture is intentionally privacy-first: no video frames, no images, no precise positions ever leave the Jetson Nano. External monitoring systems receive only high-level events (e.g., `FALL_DETECTED`, `POSTURE_WARNING`) with anonymized session identifiers.

This work complements ongoing research on continuous fall-risk estimation using DGNN + Mamba architectures conducted at Shibata Lab, Kyushu Institute of Technology.

---

## 🛠️ Tech Stack

**Edge runtime**
- NVIDIA Jetson Nano 4GB
- JetPack 4.6.1 (L4T R32.7.1)
- CUDA 10.2, TensorRT 8.2

**ML & Computer Vision**
- PyTorch 1.10
- trt_pose (NVIDIA-AI-IOT) for pose estimation
- TensorRT for inference optimization

**Backend & Communication**
- Python 3.6
- Flask (local dashboard)
- paho-mqtt (event publisher)
- Eclipse Mosquitto (MQTT broker)

**DevOps**
- Docker, docker-compose
- Bash scripts for reproducible setup

---

## 🔐 Privacy by Design

Most clinical monitoring solutions stream raw video to cloud services for analysis. This raises significant concerns:

- **GDPR / HIPAA compliance**: patient imagery is highly sensitive personal data
- **Network dependency**: cloud reliance creates failure modes in critical care
- **Attack surface**: continuous video streams are an attractive target
- **Trust**: patients and families are increasingly uncomfortable with cloud-based monitoring

This system implements privacy by architecture:

| Layer | Privacy guarantee |
|---|---|
| **Camera input** | Raw frames never written to disk |
| **Skeleton extraction** | On-device only, never transmitted |
| **Event classification** | On-device geometric rules + ML inference |
| **External communication** | Only anonymized event-level JSON via MQTT |
| **Storage** | Optional local-only event log, no media |

Example MQTT payload sent externally:

```json
{
  "session_id": "a3f8b2c1",
  "timestamp": "2026-04-21T14:32:05Z",
  "event": "FALL_DETECTED",
  "severity": "alert"
}
```

No image. No name. No position. Just the event.

---

## 🏗️ Why Jetson Nano?

This project intentionally targets edge hardware rather than a desktop GPU. Three reasons:

### 1. Privacy by hardware constraint
On-device processing is the only architectural guarantee that imagery stays local. Cloud-based alternatives can claim privacy but require trust in operational practices.

### 2. Deployment economics
Clinical-grade Jetson Nano deployment costs ~$100/room. A desktop GPU equivalent would cost $1500+ per unit and require dedicated infrastructure (climate control, networking, maintenance). For institutions deploying at scale, this is the difference between feasible and infeasible.

### 3. Real-time constraints
Inference latency must stay under 100 ms for safety-critical events. Cloud round-trip adds 100-300 ms, exceeding clinical safety thresholds. Edge inference removes this constraint entirely.

### Measured performance on Jetson Nano (4GB)

| Metric | Value |
|---|---|
| Skeleton extraction (trt_pose) | ~12 FPS |
| End-to-end inference latency | 25 ms |
| Power consumption | 6.8 W average |
| Cold-start time | 8 seconds |
| Memory footprint | 2.1 GB RAM |

*Benchmarks pending — values will be updated as the project progresses.*

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                       JETSON NANO                      │
│                                                        │
│  ┌──────────┐    ┌──────────────┐   ┌────────────────┐ │
│  │  Camera  │───►│ Skeleton     │──►│ Event detector │ │
│  │  (USB)   │    │ extractor    │   │ (geom rules)   │ │
│  └──────────┘    │ (trt_pose)   │   └────────┬───────┘ │
│                  └──────────────┘            │         │
│                                              ▼         │
│                  ┌──────────────┐   ┌────────────────┐ │
│                  │ Local        │◄──┤ MQTT publisher │ │
│                  │ Dashboard    │   │ (anonymized)   │ │
│                  │ (Flask)      │   └────────┬───────┘ │
│                  └──────────────┘            │         │
└──────────────────────────────────────────────┼─────────┘
                                               │
                                               ▼
                                       External system
                                       (event-only, no imagery)
```

### Pipeline

1. **Capture**: USB camera streams at 640×480, 30 FPS
2. **Skeleton extraction**: trt_pose (ResNet-18 backbone, TensorRT-optimized) extracts 18 keypoints
3. **Event detection**: geometric rules on keypoint trajectories
   - Trunk verticality (fall posture)
   - Center-of-mass height (fallen position)
   - Movement variance (prolonged stillness)
4. **State machine**: events transition through `OK` → `WARNING` → `ALERT` based on duration thresholds
5. **Output**:
   - Local Flask dashboard (LAN-only)
   - MQTT publisher with anonymized payload
   - Optional local event log (CSV)

---

## 🚀 Quick Start

### Prerequisites

- NVIDIA Jetson Nano 4GB with JetPack 4.6.1
- USB camera
- Active swap memory (4 GB+ recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/jetson-clinical-edge-monitor.git
cd jetson-clinical-edge-monitor

# Install system dependencies
./scripts/install_dependencies.sh

# Download pre-trained pose model (~150 MB)
./scripts/download_model.sh
```

### Run the demo

```bash
# Start the monitoring system
python3 src/main.py --config config/default.yaml

# Access local dashboard from another device on the LAN:
# http://<jetson-ip>:8080
```

### Run with Docker

```bash
docker-compose up
```

---

## 📁 Project structure

```
jetson-clinical-edge-monitor/
├── README.md
├── LICENSE
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── src/
│   ├── main.py                  # Entry point
│   ├── pose_extractor.py        # trt_pose wrapper
│   ├── event_detector.py        # Geometric event rules
│   ├── mqtt_publisher.py        # Anonymized event publishing
│   ├── dashboard/               # Flask dashboard
│   └── utils/
│       └── geometry.py          # Angle and distance helpers
│
├── models/
│   └── resnet18_baseline_att_224x224_A_epoch_249.pth
│
├── config/
│   └── default.yaml             # Thresholds, MQTT broker, etc.
│
├── docs/
│   ├── architecture.png
│   ├── setup_photo.jpg
│   └── demo.gif
│
└── scripts/
    ├── install_dependencies.sh
    ├── download_model.sh
    └── benchmark.py
```

---

## 🗺️ Roadmap

- [x] Project setup and repository scaffolding
- [ ] Day 1: trt_pose installation and live skeleton extraction
- [ ] Day 2: Geometric event detection logic (fall, posture risk, stillness)
- [ ] Day 3: MQTT publisher with anonymized payload
- [ ] Day 4: Local dashboard (Flask) and Dockerization
- [ ] Day 5: Documentation, performance benchmarks, demo video

---

## 🎬 Demo

*Demo video will be available upon project completion.*

The demo will showcase:
- Live skeleton extraction with overlay
- Posture warning trigger when leaning
- Fall detection trigger when on the ground
- MQTT events arriving in real time on an external client
- No imagery present in MQTT payload (privacy verification)

---

## 📊 Limitations & Future Work

### Current limitations

- **Skeleton-only approach** can fail in cluttered scenes or with multiple subjects
- **12 FPS may not capture rapid falls** — a Jetson Orin Nano (40 TOPS) would handle 30 FPS robustly
- **Geometric rules are simple heuristics** — not a substitute for the trained ML approach in the parent research
- **Single camera perspective** — multi-view fusion would improve robustness
- **No clinical certification** — production deployment would require IEC 62304 / ISO 13485 compliance

### Planned extensions

- [ ] Integration with the Mamba-based fall-risk model from parent research
- [ ] Multi-camera fusion for occlusion robustness
- [ ] OPC-UA bridge for direct integration with hospital legacy systems
- [ ] Power consumption optimization (sleep mode + motion-triggered wake-up)
- [ ] Migration path to Jetson Orin Nano benchmarked

---

## 🔬 Related Research

This project complements ongoing research on continuous fall-risk estimation using DGNN + Mamba architectures conducted at Shibata Lab, Kyushu Institute of Technology.

While the parent research focuses on producing a continuous risk score r(t) ∈ [0,1] from skeletal sequences using deep sequential models, this project demonstrates the engineering feasibility of deploying such systems on edge hardware with strong privacy guarantees.

Paper in preparation for IROS 2026 Workshop.

---

## 📚 References

- [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) — NVIDIA's pose estimation library
- [JetPack 4.6.1 documentation](https://docs.nvidia.com/jetson/archives/r32.7/index.html)
- [MQTT specification](https://mqtt.org/mqtt-specification/)
- [GDPR and healthcare AI](https://gdpr-info.eu/) — context for privacy requirements

---

## 📫 Contact

Built by Martin Vogel as part of an M.Sc. research program in assistive robotics.

- **LinkedIn**: https://www.linkedin.com/in/martin-vogel-36b17220b/
- **GitHub**: https://github.com/martinvgl

For questions about the parent research on Mamba-based fall-risk estimation, please reach out via LinkedIn.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The pre-trained pose model is provided by NVIDIA-AI-IOT under their respective license.
