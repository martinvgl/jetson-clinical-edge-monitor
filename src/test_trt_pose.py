"""
Test trt_pose live keypoint extraction on Jetson Nano.

Pipeline:
  1. Load ResNet-18 baseline model (PyTorch checkpoint)
  2. Convert to TensorRT (one-time, cached)
  3. Open webcam
  4. Extract 18 keypoints per frame
  5. Save annotated frames every second to /tmp for inspection
"""

import os
import json
import time
import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image
import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule, torch2trt
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects

# ----- Config -----
MODEL_PATH = os.path.expanduser("~/jetson-models/resnet18_baseline_att_224x224_A_epoch_249.pth")
TRT_MODEL_PATH = os.path.expanduser("~/jetson-models/resnet18_trt.pth")
POSE_JSON = os.path.expanduser("~/trt_pose/tasks/human_pose/human_pose.json")
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
SAVE_DIR = "/tmp/trt_pose_output"
SAVE_EVERY_N_FRAMES = 30  # ~1 image par seconde à 30 FPS
TEST_DURATION_SEC = 30


def get_model():
    """Load model: from cached TRT if available, else convert from PyTorch."""
    if os.path.exists(TRT_MODEL_PATH):
        print(f"Loading cached TensorRT model from {TRT_MODEL_PATH}...")
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(TRT_MODEL_PATH))
        return model_trt

    print("No cached TRT model. Converting from PyTorch (this takes 2-3 min)...")

    with open(POSE_JSON, 'r') as f:
        human_pose = json.load(f)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    model = trt_pose.models.resnet18_baseline_att(
        num_parts, 2 * num_links).cuda().eval()

    print(f"Loading PyTorch checkpoint from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH))

    print("Converting to TensorRT...")
    data = torch.zeros((1, 3, INPUT_HEIGHT, INPUT_WIDTH)).cuda()
    model_trt = torch2trt(model, [data], max_workspace_size=1 << 25)

    print(f"Saving TRT model to {TRT_MODEL_PATH}...")
    torch.save(model_trt.state_dict(), TRT_MODEL_PATH)

    return model_trt


def preprocess(image, mean, std, device):
    """BGR (OpenCV) -> RGB tensor normalized for the model."""
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    # Load topology and prepare parsers
    with open(POSE_JSON, 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)

    # Load or convert model
    model_trt = get_model()

    # Normalization values (ImageNet)
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    # Open webcam
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return 1

    print(f"Streaming for {TEST_DURATION_SEC} seconds, saving annotated frames to {SAVE_DIR}/")
    print(f"(One image saved every {SAVE_EVERY_N_FRAMES} frames)")

    frame_count = 0
    saved_count = 0
    start = time.time()

    while time.time() - start < TEST_DURATION_SEC:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        data = preprocess(frame, mean, std, device)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)

        # Draw on the original frame
        draw_objects(frame, counts, objects, peaks)

        # Save periodically
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            output_path = f"{SAVE_DIR}/frame_{saved_count:04d}.jpg"
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"  Saved {output_path} (people detected: {int(counts[0])})")

        frame_count += 1

    elapsed = time.time() - start
    fps = frame_count / elapsed
    print(f"\nDone. {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
    print(f"{saved_count} annotated images saved to {SAVE_DIR}/")

    cap.release()
    return 0


if __name__ == "__main__":
    exit(main())