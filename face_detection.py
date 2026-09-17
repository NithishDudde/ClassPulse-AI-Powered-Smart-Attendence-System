import cv2
import os

# -----------------------------
# Model Path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx"
)

# -----------------------------
# Create YuNet Face Detector
# -----------------------------
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

# -----------------------------
# Open Webcam
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to Quit")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    count = 0

    if faces is not None:

        count = len(faces)

        for face in faces:

            x, y, width, height = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Students : {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ClassPulse Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()