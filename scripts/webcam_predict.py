import cv2
from ultralytics import YOLO

model = YOLO(r"N:\ROV\scripts\runs\classify\train\weights\best.pt")

# Change this mapping if needed
class_names = {
    0: "Forward",   # like
    1: "Invalid",
    2: "Left",      # one
    3: "Reverse",   # fist
    4: "Right",     # peace
    5: "Stop"       # palm
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    probs = results[0].probs
    cls_id = int(probs.top1)
    conf = float(probs.top1conf)

    label = class_names.get(cls_id, "Unknown")

    # Confidence filter
    if conf < 0.6:
        label = "Invalid"

    cv2.putText(frame, f"{label} ({conf:.2f})",
                (20,50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

