# Installation Guide

Complete reproducible installation procedure for **Jetson Nano with JetPack 4.6.1** (Ubuntu 18.04, Python 3.6.9, CUDA 10.2).

> ⚠️ As of 2026, the Jetson Nano JetPack 4.6.1 is on a deprecated Python 3.6 stack. Many modern PyPI packages no longer support it. The order of operations below was empirically determined and matters.

## Prerequisites

- Jetson Nano flashed with JetPack 4.6.1
- USB webcam connected (`/dev/video0` visible after `ls /dev/video*`)
- Internet connection
- ~6 GB free disk space (PyTorch + torchvision sources are heavy)
- ~2 hours total install time (mostly torchvision compilation)

## Step 1 — Configure swap (mandatory)

PyTorch + torchvision compilation requires more RAM than the Nano has.

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -m  # verify ~4 GB swap
```

## Step 2 — System dependencies

```bash
sudo apt update
sudo apt install -y \
    python3-pip python3-matplotlib python3-pil \
    libopenmpi-dev libopenblas-base libomp-dev \
    libjpeg-dev zlib1g-dev libpython3-dev libpython3.6-dev \
    libfreetype6-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    git wget
```

## Step 3 — PyTorch 1.10 (NVIDIA wheel)

NVIDIA provides pre-compiled wheels for JetPack:

```bash
cd ~
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
     -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install pillow==8.4.0
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 1.10.0 True
```

## Step 4 — Torchvision 0.11.1 (compile from source)

Pre-compiled wheels for ARM64 + Python 3.6 don't exist for torchvision 0.11. Compile it (≈50 min on Nano):

```bash
cd ~
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
sudo python3 setup.py install
```

Verify:
```bash
python3 -c "import torchvision; print(torchvision.__version__)"
# Expected: 0.11.1
```

## Step 5 — Pin Python deps in correct order

The default pip resolver picks versions incompatible with Python 3.6. Pin explicitly:

```bash
sudo pip3 install --no-deps "Cython==0.29.36"
sudo pip3 install --no-deps "numpy==1.19.5"
sudo pip3 install --no-deps "matplotlib==3.3.4"
sudo pip3 install --no-deps "seaborn==0.11.2"
sudo pip3 install --no-deps "Pillow==8.4.0"
sudo pip3 install packaging tqdm pycocotools gdown
```

## Step 6 — Flask (with manual deps)

Flask on Python 3.6 won't pull its modern deps automatically. Install manually:

```bash
sudo pip3 install --no-deps werkzeug jinja2 itsdangerous click markupsafe
sudo pip3 install --no-deps flask
```

Verify:
```bash
python3 -c "import flask; print(flask.__version__)"
```

## Step 7 — torch2trt + trt_pose

```bash
cd ~
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins

cd ~
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

Verify:
```bash
python3 -c "import trt_pose, trt_pose.coco, trt_pose.models; print('OK')"
```

## Step 8 — Download pre-trained model

The official Google Drive link returns "access denied" as of 2026. Use the GitHub Releases mirror instead (automated in our script):

```bash
cd ~/jetson-clinical-edge-monitor
./scripts/download_models.sh
```

Or manually:
```bash
mkdir -p ~/jetson-models
cd ~/jetson-models
wget -O resnet18_baseline_att_224x224_A_epoch_249.pth \
    https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth
```

## Step 9 — Run the demo

```bash
cd ~/jetson-clinical-edge-monitor
python3 src/test_trt_pose_stream.py
```

Open `http://<jetson-ip>:5000/` in any browser on the LAN.

## Known issues encountered during installation

| Issue | Symptom | Fix |
|---|---|---|
| Cython 3.x incompatible with NumPy 1.19.5 | Cython compilation errors during `pip install numpy` | Install `Cython==0.29.36` first |
| matplotlib 3.3.4 fails to compile | `Failed to download FreeType` error | Install `libfreetype6-dev` via apt before retrying |
| TensorRT conversion fails | `count of weights mismatch` between trt_pose and torch2trt | Use PyTorch-only inference (no `torch2trt(...)` call) |
| YOLOv5 6.2 dependency hell | matplotlib pulls NumPy 2.4 (Python 3.10+) | Use `--no-deps` flag systematically |
| Google Drive model 403 | "Access denied" on official link | Use GitHub Releases mirror |
| Flask doesn't install with deps | `--no-deps` skipped werkzeug, jinja2 | Install Flask deps manually |

## Cleanup (optional)

After successful installation, you can free disk space:

```bash
rm -rf ~/torchvision  # source tree (~1.5 GB)
rm ~/torch-1.10.0-cp36-cp36m-linux_aarch64.whl  # 322 MB
```

The compiled libraries are installed in `/usr/local/lib/python3.6/` and persist.