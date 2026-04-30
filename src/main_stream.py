"""
Integrated demo: webcam → trt_pose → event detector → live MJPEG stream.

Open http://<jetson-ip>:5000/ in a browser.
You'll see the live feed with skeletons + an overlay showing detected events
(FALL_DETECTED, POSTURE_WARNING, PROLONGED_STILLNESS).

Events are also printed to the console for now.
The MQTT publisher will plug in here once the broker is set up.
"""

import os
import json
import time
import threading
from collections import deque

import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from flask import Flask, Response

from event_detector import EventDetector, EventType, Severity


# ----- Config -----
MODEL_PATH = os.path.expanduser(
    "~/jetson-models/resnet18_baseline_att_224x224_A_epoch_249.pth")
POSE_JSON = os.path.expanduser("~/trt_pose/tasks/human_pose/human_pose.json")
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
HTTP_PORT = 5000

# How long to keep an event banner on screen after firing
EVENT_DISPLAY_SEC = 3.0


# ----- Globals shared between threads -----
output_frame = None
frame_lock = threading.Lock()
recent_events = deque(maxlen=5)


# ----- Model loading -----
def get_model():
    print("Loading PyTorch model...")
    with open(POSE_JSON, 'r') as f:
        human_pose = json.load(f)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.resnet18_baseline_att(
        num_parts, 2 * num_links).cuda().eval()
    print(f"Loading checkpoint from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(result.missing_keys)}")
    print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    print("Model loaded.")
    return model


def preprocess(image, mean, std, device):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


# ----- Keypoint extraction -----
def extract_pose_dict(counts, objects, peaks, image_width, image_height):
    """
    Convert trt_pose outputs to {kp_index: (x_px, y_px)} for the
    largest detected person (or None if no person detected).
    """
    n_obj = int(counts[0])
    if n_obj == 0:
        return None

    obj = objects[0][0]  # shape: (num_keypoints,)
    num_keypoints = obj.shape[0]

    pose = {}
    for kp_idx in range(num_keypoints):
        peak_idx = int(obj[kp_idx])
        if peak_idx < 0:
            continue  # keypoint not detected
        # peaks are normalized [0, 1] in (y, x) order
        peak = peaks[0][kp_idx][peak_idx]
        y_norm = float(peak[0])
        x_norm = float(peak[1])
        pose[kp_idx] = (x_norm * image_width, y_norm * image_height)

    return pose


# ----- Visualization helpers -----
def draw_event_banner(frame, events):
    """Draw colored banners at the top for recent events."""
    now = time.time()
    y_offset = 30
    for evt, fired_at in list(events):
        age = now - fired_at
        if age > EVENT_DISPLAY_SEC:
            continue

        if evt.severity == Severity.ALERT:
            color = (0, 0, 255)        # red
        elif evt.severity == Severity.WARNING:
            color = (0, 165, 255)      # orange
        else:
            color = (0, 255, 255)

        text = f"[{evt.severity.value.upper()}] {evt.type.value}"
        cv2.rectangle(frame, (5, y_offset - 22),
                      (5 + 14 * len(text), y_offset + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 35


# ----- Inference loop -----
def inference_loop():
    global output_frame

    device = torch.device('cuda')
    with open(POSE_JSON, 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)

    model = get_model()
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    detector = EventDetector()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("Inference loop started.")
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        t_now = time.time()

        # 1. Pose inference
        data = preprocess(frame, mean, std, device)
        with torch.no_grad():
            cmap, paf = model(data)
        cmap, paf = cmap.cpu(), paf.cpu()
        counts, objects, peaks = parse_objects(cmap, paf)

        # 2. Draw skeleton on the frame
        draw_objects(frame, counts, objects, peaks)

        # 3. Event detection
        pose = extract_pose_dict(counts, objects, peaks, w, h)
        event = detector.update(t_now, pose, h)
        if event is not None:
            print(f"[{time.strftime('%H:%M:%S')}] EVENT: "
                  f"{event.type.value} ({event.severity.value}) "
                  f"meta={event.metadata}")
            recent_events.append((event, t_now))

        # 4. Overlays
        cv2.putText(frame, f"People: {int(counts[0])}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        draw_event_banner(frame, recent_events)

        # 5. FPS counter
        fps_counter += 1
        if fps_counter >= 10:
            now = time.time()
            current_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start = now

        with frame_lock:
            output_frame = frame.copy()


def generate_mjpeg():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ----- Flask -----
app = Flask(__name__)


@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Jetson Clinical Edge Monitor</title>
        <style>
          body { background:#1a1a1a; color:#eee; font-family:sans-serif; text-align:center; margin:0; padding:20px;}
          h1 { color:#4ade80; }
          img { border:2px solid #444; border-radius:8px; }
          .footer { margin-top:20px; color:#888; font-size:0.85em;}
        </style>
      </head>
      <body>
        <h1>Jetson Clinical Edge Monitor</h1>
        <p>Live feed with skeleton detection and event monitoring.</p>
        <img src="/video" width="640" />
        <div class="footer">
          Privacy notice: this debug stream is LAN-only and disabled in production.<br>
          Press Ctrl+C in the server terminal to stop.
        </div>
      </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print(f"Server starting on http://0.0.0.0:{HTTP_PORT}/")
    print(f"Open http://<jetson-ip>:{HTTP_PORT}/ from any browser on the LAN.")
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, debug=False)