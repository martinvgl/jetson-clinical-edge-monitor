# Privacy-First Clinical Edge Monitor



[![Jetson](https://img.shields.io/badge/Jetson-Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()
[![Hardware](https://img.shields.io/badge/hardware-Jetson%20Nano-green)]()
[![Python](https://img.shields.io/badge/python-3.6-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements an embedded clinical monitoring system that detects events of interest (falls, prolonged stillness, posture warnings) **without ever transmitting patient imagery**. All computer vision processing — including pose estimation — is performed locally on a Jetson Nano. Only anonymized high-level events are published over MQTT.

The system targets healthcare contexts where:

- **Patient privacy is non-negotiable** (no images, no keypoints leave the device)
- **Hardware budget is constrained** (~$100 Jetson Nano vs. $1500+ workstation GPU)
- **Real-time response matters** (~200 ms event latency)

## Why this design?

**Privacy by hardware constraint.** The system can't leak imagery because no networking code ever touches a frame buffer. The only network output is a JSON event payload (e.g., `FALL_DETECTED`).

**Edge deployment economics.** A Jetson Nano + USB camera setup costs ~$100 and runs autonomously. Equivalent cloud-based monitoring would require continuous video upload, GDPR/HIPAA-compliant cloud infrastructure, and 24/7 bandwidth.

**Bridging legacy hardware and modern AI.** The Jetson Nano (JetPack 4.6.1, Python 3.6, CUDA 10.2) is a deprecated platform in the 2026 ecosystem. Successfully deploying modern PyTorch models on it required navigating broken dependency chains — a realistic challenge for industrial edge deployments where hardware refresh cycles span 5–10 years.

## Architecture

```
┌──────────────────────────────────────────────┐
│             Jetson Nano (edge)               │
│                                              │
│  USB webcam                                  │
│       ↓                                      │
│  OpenCV capture (640x480 @ 30 FPS)           │
│       ↓                                      │
│  ResNet-18 pose estimation (trt_pose)        │
│       ↓                                      │
│  18 keypoints per detected person            │
│       ↓                                      │
│  Event detector [WIP]                        │
│   ├─ Fall (sudden CoM drop)                  │
│   ├─ Stillness (no movement > N seconds)     │
│   └─ Posture warning (abnormal trunk angle)  │
│       ↓                                      │
│  Anonymized event payload [WIP]              │
│       ↓ MQTT                                 │
└──────────────────────────────────────────────┘
                                 ↓
                        clinical/events topic
                        (no images, no keypoints,
                         only event labels)
```

## Tech stack

| Layer            | Component                              |
|------------------|----------------------------------------|
| Hardware         | Jetson Nano 4GB + USB webcam           |
| OS               | Ubuntu 18.04 (JetPack 4.6.1)           |
| Runtime          | Python 3.6 / CUDA 10.2                 |
| Capture          | OpenCV 4.1.1                           |
| Inference        | PyTorch 1.10 + trt_pose (ResNet-18)    |
| Streaming        | Flask MJPEG (debug interface)          |
| Messaging        | MQTT [WIP]                             |
| Containerization | Docker [planned]                       |

## Privacy by design

Every architectural decision serves one constraint: **no patient imagery ever leaves the device**.

- ✅ Frames are processed in-memory only, never written to disk in production mode
- ✅ Keypoints are computed locally and never published — only event labels are
- ✅ MQTT payloads contain only: `session_id` (anonymous UUID), `timestamp`, `event_type`, `severity`
- ✅ Local debug stream (Flask MJPEG) is LAN-only and disabled in production builds

Example MQTT payload:
```json
{
  "session_id": "8f3a1b2c",
  "timestamp": "2026-04-30T14:23:18Z",
  "event_type": "FALL_DETECTED",
  "severity": "alert"
}
```

## Current status

- [x] Webcam capture pipeline (30 FPS)
- [x] ResNet-18 pose estimation (5 FPS, PyTorch fallback)
- [x] Live debug stream over HTTP MJPEG
- [x] Reproducible install procedure for Jetson Nano JetPack 4.6.1
- [ ] Event detector (fall / stillness / posture)
- [ ] MQTT publisher with anonymized payloads
- [ ] Docker container for reproducible deployment
- [ ] Backend service (separate repo): event ingestion, REST API, PostgreSQL persistence

## Quick start

### Prerequisites

- NVIDIA Jetson Nano with JetPack 4.6.1 flashed
- USB webcam plugged in (visible as `/dev/video0`)
- Network connectivity

### Install

See [INSTALL.md](INSTALL.md) for the full reproducible procedure. Summary:

```bash
# 1. Clone this repo
git clone https://github.com/martinvgl/jetson-clinical-edge-monitor
cd jetson-clinical-edge-monitor

# 2. Install system + Python dependencies
# (Detailed steps in INSTALL.md — order matters on Python 3.6)

# 3. Download pre-trained pose estimation model
./scripts/download_models.sh

# 4. Run the live streaming demo
python3 src/test_trt_pose_stream.py
```

### View the live stream

Once `test_trt_pose_stream.py` is running:

```
http://<jetson-ip>:5000/
```

You should see a live feed annotated with skeleton keypoints.

## Performance notes

| Pipeline stage           | Throughput |
|--------------------------|------------|
| Webcam capture only      | 30 FPS     |
| Capture + pose (PyTorch) | ~5 FPS     |
| Capture + pose (TensorRT)| ~12 FPS expected (deferred — see Known Issues) |

Even at 5 FPS, the latency is sufficient for clinical event detection: falls last ~1 s, prolonged stillness is measured over 30+ seconds, and posture warnings tolerate slow update rates.

## Known issues

- **TensorRT conversion fails** with `count of weights mismatch` between `trt_pose 0.0.1` and `torch2trt 0.5.0`. Current workaround: PyTorch-only inference. A fix or version downgrade is on the roadmap.
- **Google Drive model link is blocked**. The official trt_pose README points to a Drive link that returns "access denied". Use the GitHub Releases mirror instead (already automated in `scripts/download_models.sh`).
- **Python 3.6 ecosystem decay**. As of 2026, many recent PyPI packages no longer support Python 3.6. `INSTALL.md` documents version pins required for matplotlib, NumPy, seaborn, Flask, and others.

## Project structure

```
jetson-clinical-edge-monitor/
├── README.md                    ← you are here
├── INSTALL.md                   ← reproducible install procedure
├── LICENSE
├── .gitignore
├── configs/
│   └── human_pose.json          ← keypoint topology (18 keypoints, COCO)
├── scripts/
│   └── download_models.sh       ← fetches the ResNet-18 weights
├── src/
│   ├── test_camera.py           ← webcam sanity check
│   ├── test_trt_pose.py         ← pose estimation, image dump mode
│   └── test_trt_pose_stream.py  ← pose estimation, live MJPEG mode
└── tests/                       ← unit tests [planned]
```

## Roadmap

**Short term**
- Event detector module: fall, stillness, posture warning
- MQTT publisher with anonymized payloads
- Configuration via YAML (`configs/default.yaml`)

**Medium term**
- Backend repo: `jetson-event-ingestion-api` (FastAPI + PostgreSQL)
- Dockerfile + docker-compose
- Integration tests with simulated MQTT broker

**Long term**
- TensorRT acceleration (~12 FPS target)
- Multi-camera support
- Anomaly detection trained on healthy baselines (research direction)

## Related work

This deployment-focused project complements ongoing research on **fall-risk prediction using Selective State-Space Models (Mamba SSM)**, targeted at IROS 2026 Workshop submission.

Where the research line optimizes predictive accuracy on academic benchmarks (URFD, Le2i datasets), this repo targets **deployment realism** on constrained edge hardware. Different problems, complementary skills.

> Research repo (private until publication): `fall-risk-mamba — link upon paper acceptance`

## Disclaimer

This is an **engineering demonstration project**, not a certified medical device. It is not validated for clinical use and should not be deployed in care settings without proper regulatory review (IEC 62304, ISO 13485, and local medical device regulations). The architecture is intended to illustrate privacy-first design patterns for embedded healthcare AI.

## Author

**Martin Vogel** — Graduate researcher in assistive robotics

- M1 student at Kyushu Institute of Technology (Shibata Lab), Japan
- 5th-year engineering student at Polytech Nancy, France
- Research focus: fall-risk prediction using Selective State-Space Models (Mamba SSM)

🔗 [GitHub](https://github.com/martinvgl)

## License

MIT — see [LICENSE](LICENSE).