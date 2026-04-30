import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera 0")
    exit(1)

ret, frame = cap.read()

if ret:
    print(f"SUCCESS: Frame captured, shape={frame.shape}")
    cv2.imwrite("/tmp/test_capture.jpg", frame)
    print("Saved to /tmp/test_capture.jpg")
else:
    print("ERROR: Camera opened but cannot read frame")

cap.release()