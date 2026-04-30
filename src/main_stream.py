"""
Integrated demo: webcam -> trt_pose -> event detector -> MQTT publish + live stream.

Pipeline:
  1. Capture webcam frame
  2. Run ResNet-18 pose estimation
  3. Extract keypoints
  4. Run event detector (state machine)
  5. If event fires:
       - publish anonymized payload over MQTT
       - print to console
       - draw banner on the live stream
  6. Stream annotated frame as MJPEG over HTTP

Open http://<jetson-ip>:5000/ in any browser on the LAN to view the feed.
Subscribe with: mosquitto_sub -h localhost -t "clinical/events" -v
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
from mqtt_publisher import AnonymizedEventPublisher


# ----- Config -----
MODEL_PATH = os.path.expanduser(
    "~/jetson-models/resnet18_baseline_att_224x224_A_epoch_249.pth")
POSE_JSON = os.path.expanduser("~/trt_pose/tasks/human_pose/human_pose.json")
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
HTTP_PORT = 5000

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "clinical/events"

EVENT_DISPLAY_SEC = 3.0


# ----- Globals -----
output_frame = None
frame_lock = threading.Lock()
recent_events = deque(maxlen=5)


# ----- Model -----
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


def extract_pose_dict(counts, objects, peaks, image_width, image_height):
    n_obj = int(counts[0])
    if n_obj == 0:
        return None

    obj = objects[0][0]
    num_keypoints = obj.shape[0]
    pose = {}
    for kp_idx in range(num_keypoints):
        peak_idx = int(obj[kp_idx])
        if peak_idx < 0:
            continue
        peak = peaks[0][kp_idx][peak_idx]
        y_norm = float(peak[0])
        x_norm = float(peak[1])
        pose[kp_idx] = (x_norm * image_width, y_norm * image_height)
    return pose


def draw_event_banner(frame, events):
    now = time.time()
    y_offset = 30
    for evt, fired_at in list(events):
        age = now - fired_at
        if age > EVENT_DISPLAY_SEC:
            continue
        if evt.severity == Severity.ALERT:
            color = (0, 0, 255)
        elif evt.severity == Severity.WARNING:
            color = (0, 165, 255)
        else:
            color = (0, 255, 255)
        text = f"[{evt.severity.value.upper()}] {evt.type.value}"
        cv2.rectangle(frame, (5, y_offset - 22),
                      (5 + 14 * len(text), y_offset + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 35


# ----- Inference + publish -----
def inference_loop(publisher: AnonymizedEventPublisher):
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

        # 1. Pose
        data = preprocess(frame, mean, std, device)
        with torch.no_grad():
            cmap, paf = model(data)
        cmap, paf = cmap.cpu(), paf.cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(frame, counts, objects, peaks)

        # 2. Event detection
        pose = extract_pose_dict(counts, objects, peaks, w, h)
        event = detector.update(t_now, pose, h)
        if event is not None:
            print(f"[{time.strftime('%H:%M:%S')}] EVENT: "
                  f"{event.type.value} ({event.severity.value}) "
                  f"meta={event.metadata}")
            recent_events.append((event, t_now))

            # 3. Publish over MQTT (anonymized: metadata is NOT included)
            ok = publisher.publish_event(event)
            if not ok:
                print("  WARNING: MQTT publish failed")

        # 4. Overlays
        cv2.putText(frame, f"People: {int(counts[0])}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        draw_event_banner(frame, recent_events)

        # 5. FPS
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
          body { background:#1a1a1a; color:#eee; font-family:sans-serif;
                 text-align:center; margin:0; padding:20px; }
          h1 { color:#4ade80; }
          img { border:2px solid #444; border-radius:8px; }
          .footer { margin-top:20px; color:#888; font-size:0.85em; }
        </style>
      </head>
      <body>
        <h1>Jetson Clinical Edge Monitor</h1>
        <p>Live feed with skeleton detection, event monitoring, and MQTT publishing.</p>
        <img src="/video" width="640" />
        <div class="footer">
          Privacy notice: this debug stream is LAN-only and disabled in production.<br>
          Events published to MQTT topic <code>clinical/events</code>.<br>
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
    # Connect MQTT publisher first; refuse to start if broker unreachable
    publisher = AnonymizedEventPublisher(
        broker_host=MQTT_BROKER_HOST,
        broker_port=MQTT_BROKER_PORT,
        topic=MQTT_TOPIC,
    )
    if not publisher.connect():
        raise SystemExit(
            f"Cannot connect to MQTT broker at "
            f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}. "
            f"Is mosquitto running? `sudo systemctl status mosquitto`"
        )
    print(f"MQTT publisher connected. Session: {publisher.session_id}")
    print(f"Topic: {publisher.topic}")

    t = threading.Thread(target=inference_loop, args=(publisher,), daemon=True)
    t.start()

    print(f"Server starting on http://0.0.0.0:{HTTP_PORT}/")
    print(f"Open http://<jetson-ip>:{HTTP_PORT}/ from any browser on the LAN.")
    try:
        app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, debug=False)
    finally:
        publisher.disconnect()