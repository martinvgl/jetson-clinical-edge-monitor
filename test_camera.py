"""
Test webcam capture and basic stream FPS measurement.
First sanity check that the Jetson + webcam pipeline works.
"""

import cv2
import time

CAMERA_INDEX = 0
TEST_DURATION_SEC = 5


def main():
    print(f"Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        return 1

    # Capture single frame for shape verification
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        cap.release()
        return 1

    print(f"Frame shape: {frame.shape}")
    print(f"Streaming for {TEST_DURATION_SEC} seconds to measure FPS...")

    frame_count = 0
    start = time.time()

    while time.time() - start < TEST_DURATION_SEC:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame_count += 1

    elapsed = time.time() - start
    fps = frame_count / elapsed

    print(f"Captured {frame_count} frames in {elapsed:.2f}s")
    print(f"Average FPS: {fps:.1f}")

    cap.release()
    return 0


if __name__ == "__main__":
    exit(main())