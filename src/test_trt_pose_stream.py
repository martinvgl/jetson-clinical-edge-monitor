"""
trt_pose live keypoint streaming via MJPEG over HTTP.

Open http://<jetson_ip>:5000/ in a browser to see the live feed
with skeletons drawn.
"""

import os
import json
import threading
import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from flask import Flask, Response

# ----- Config -----
MODEL_PATH = os.path.expanduser("~/jetson-models/resnet18_baseline_att_224x224_A_epoch_249.pth")
POSE_JSON = os.path.expanduser("~/trt_pose/tasks/human_pose/human_pose.json")
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
HTTP_PORT = 5000

# ----- Globals (shared between threads) -----
output_frame = None
frame_lock = threading.Lock()


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


def inference_loop():
    """Background thread: capture + inference + update output_frame."""
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

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("Inference loop started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        data = preprocess(frame, mean, std, device)
        with torch.no_grad():
            cmap, paf = model(data)
        cmap, paf = cmap.cpu(), paf.cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(frame, counts, objects, peaks)

        # Annotate frame with people count
        cv2.putText(frame, f"People: {int(counts[0])}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        with frame_lock:
            output_frame = frame.copy()


def generate_mjpeg():
    """Generator that yields the latest frame as MJPEG."""
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


# ----- Flask app -----
app = Flask(__name__)


@app.route('/')
def index():
    return """
    <html>
      <head><title>Jetson Pose Stream</title></head>
      <body style="background:#222; color:#eee; font-family:sans-serif; text-align:center;">
        <h1>Jetson Clinical Edge Monitor - Live Feed</h1>
        <img src="/video" width="640" />
        <p>Press Ctrl+C in the terminal to stop the server</p>
      </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Start inference in a background thread
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    # Start Flask
    print(f"Server starting on http://0.0.0.0:{HTTP_PORT}/")
    print(f"Open http://172.17.69.114:{HTTP_PORT}/ in your PC browser")
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, debug=False)