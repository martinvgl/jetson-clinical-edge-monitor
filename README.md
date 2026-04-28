# Privacy-First Clinical Edge Monitor

> Real-time clinical monitoring on NVIDIA Jetson Nano: skeleton-based event detection with zero patient imagery transmission.

🚧 **Work in progress** — under active development.

## Overview

This project demonstrates a privacy-preserving clinical monitoring system running entirely on edge hardware. Patient video is processed on-device for skeleton extraction; only anonymized event-level data is transmitted via MQTT.

## Why Edge?

- **Privacy by design**: video frames never leave the device
- **Real-time inference**: low-latency event detection without cloud dependency
- **Cost-effective deployment**: ~$100 hardware vs $1500+ for desktop GPU equivalent

## Tech stack

- NVIDIA Jetson Nano 4GB (JetPack 4.6.1)
- USB camera
- trt_pose (TensorRT-optimized human pose estimation)
- PyTorch
- MQTT (Mosquitto)
- Flask (local dashboard)
- Docker

## Roadmap

- [x] Project setup
- [ ] Day 1: trt_pose installation and live skeleton extraction
- [ ] Day 2: Geometric event detection logic (fall, posture risk, stillness)
- [ ] Day 3: MQTT publisher with anonymized payload
- [ ] Day 4: Local dashboard (Flask) and Dockerization
- [ ] Day 5: Documentation, performance benchmarks, demo video

## Context

This project complements ongoing research on continuous fall-risk estimation using DGNN + Mamba architectures, conducted at Shibata Lab, Kyushu Institute of Technology.

## License

MIT
