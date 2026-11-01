"""
ClassPulse - Attendance Recognition
====================================
Live face detection + recognition pipeline that marks student attendance
automatically during active class periods.

Pipeline:
    Webcam -> YuNet (detection) -> face_recognition (128-D embedding)
           -> match against registered students -> confirm 3x -> mark present
"""

import os
import csv
from datetime import datetime, time

import cv2
import numpy as np
import face_recognition


# =============================================================================
# PROJECT PATHS
# =============================================================================
# ClassPulse/
#     backend/   <- this file lives here
#     dataset/
#     models/

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
ATTENDANCE_FILE = os.path.join(BACKEND_DIR, "attendance.csv")


# =============================================================================
# SETTINGS
# =============================================================================

CAMERA_INDEX = 0

YUNET_SCORE_THRESHOLD = 0.60
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

# Lower value = stricter recognition | Higher value = more tolerant recognition
FACE_MATCH_THRESHOLD = 0.50

RECOGNITION_INTERVAL = 5       # run recognition every N frames
REQUIRED_CONFIRMATIONS = 3     # consecutive hits needed before marking present
MIN_FACE_SIZE = 40             # ignore faces smaller than this (pixels)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =============================================================================
# TIMETABLE
# =============================================================================

TIMETABLE = [
    ("Period 1",      time(9, 30),  time(10, 20)),
    ("Period 2",      time(10, 20), time(11, 10)),
    ("Toilet Break",  time(11, 10), time(11, 20)),
    ("Period 3",      time(11, 20), time(12, 0)),
    ("Period 4",      time(12, 10), time(13, 0)),
    ("Lunch Break",   time(13, 0),  time(13, 40)),
    ("Period 5",      time(13, 40), time(14, 30)),
    ("Period 6",      time(14, 30), time(15, 20)),
    ("Period 7",      time(15, 20), time(16, 10)),
]


# =============================================================================
# HOLIDAYS
# =============================================================================
# Specific dates (YYYY-MM-DD) when attendance should never be marked,
# regardless of time of day. Update this list each term/year.

HOLIDAYS = {
    "2026-10-02",   # Example: Gandhi Jayanti
    "2026-10-24",   # Example: Diwali
    # Add more holiday dates here as "YYYY-MM-DD"
}

# Weekly off day(s). Monday = 0 ... Sunday = 6
WEEKLY_OFF_DAYS = {6}   # Sunday off


def is_holiday(today: datetime) -> bool:
    """Return True if `today` is a weekly off day or a declared holiday."""

    if today.weekday() in WEEKLY_OFF_DAYS:
        return True

    if today.strftime("%Y-%m-%d") in HOLIDAYS:
        return True

    return False


# =============================================================================
# ATTENDANCE CSV
# =============================================================================

CSV_HEADER = [
    "Student ID",
    "Student Name",
    "Date",
    "Period",
    "Start Time",
    "End Time",
    "Marked Time",
    "Status",
]


def initialize_csv() -> None:
    """Create attendance.csv with headers if it doesn't already exist."""

    if os.path.exists(ATTENDANCE_FILE):
        return

    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(CSV_HEADER)


def load_existing_attendance() -> set:
    """Load (date, period, student_id) keys already marked Present."""

    marked = set()

    if not os.path.exists(ATTENDANCE_FILE):
        return marked

    try:
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                student_id = row.get("Student ID", "").strip()
                date = row.get("Date", "").strip()
                period = row.get("Period", "").strip()
                status = row.get("Status", "").strip()

                if student_id and date and period and status.lower() == "present":
                    marked.add((date, period, student_id))

    except Exception as error:
        print(f"[WARNING] Could not read attendance.csv: {error}")

    return marked


def mark_attendance(student_id, student_name, period, start, end, marked) -> bool:
    """Append a Present row for this student/period, unless already marked."""

    today = datetime.now().strftime("%Y-%m-%d")
    key = (today, period, student_id)

    if key in marked:
        return False

    now = datetime.now().strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            student_id,
            student_name,
            today,
            period,
            start.strftime("%H:%M"),
            end.strftime("%H:%M"),
            now,
            "Present",
        ])

    marked.add(key)

    print()
    print("=" * 60)
    print("ATTENDANCE MARKED")
    print("=" * 60)
    print(f"Student ID   : {student_id}")
    print(f"Student Name : {student_name}")
    print(f"Period       : {period}")
    print(f"Marked Time  : {now}")
    print("=" * 60)

    return True


# =============================================================================
# PERIOD LOOKUP
# =============================================================================

def get_current_period():
    """
    Return (period_name, start_time, end_time) for the current moment.

    period_name is one of:
        "Holiday", "Before Classes", "After Classes",
        a break name (e.g. "Lunch Break"), or "Period N".
    start_time / end_time are None outside class periods.
    """

    now = datetime.now()

    if is_holiday(now):
        return "Holiday", None, None

    current_time = now.time()

    for name, start, end in TIMETABLE:
        if start <= current_time < end:
            return name, start, end

    if current_time < TIMETABLE[0][1]:
        return "Before Classes", None, None

    return "After Classes", None, None


def is_class_period(period: str) -> bool:
    return period.startswith("Period")


# =============================================================================
# STUDENT INFO
# =============================================================================

def read_student_info(folder: str):
    """Read Student ID / Name from student_info.txt, falling back to folder name."""

    folder_name = os.path.basename(folder)
    student_id, student_name = "", ""

    info_file = os.path.join(folder, "student_info.txt")

    if os.path.exists(info_file):
        try:
            with open(info_file, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    lower = line.lower()

                    if lower.startswith("student id:"):
                        student_id = line.split(":", 1)[1].strip()
                    elif lower.startswith("student name:"):
                        student_name = line.split(":", 1)[1].strip()

        except Exception as error:
            print(f"[WARNING] Cannot read {info_file}: {error}")

    if not student_id or not student_name:
        if "_" in folder_name:
            parts = folder_name.split("_", 1)
            student_id = student_id or parts[0]
            student_name = student_name or parts[1].replace("_", " ").strip()
        else:
            student_id = student_id or folder_name
            student_name = student_name or folder_name

    return student_id, student_name


# =============================================================================
# YUNET DETECTOR
# =============================================================================

def create_detector():
    """Create and return a YuNet face detector, or None on failure."""

    print()
    print("Checking YuNet model...")

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] YuNet model not found: {MODEL_PATH}")
        return None

    try:
        detector = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
        print("[OK] YuNet model loaded.")
        return detector

    except Exception as error:
        print(f"[ERROR] YuNet initialization failed: {error}")
        return None


def detect_faces(detector, frame):
    """Run YuNet on a frame. Always returns a list (never None)."""

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    return [] if faces is None else faces


def yunet_box_to_location(face, frame_width, frame_height):
    """Convert a YuNet (x, y, w, h, ...) box to face_recognition's
    (top, right, bottom, left) format. Returns None if the box is unusable."""

    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    left = max(0, x)
    top = max(0, y)
    right = min(frame_width, x + w)
    bottom = min(frame_height, y + h)

    if right <= left or bottom <= top:
        return None

    return top, right, bottom, left


# =============================================================================
# FACE ENCODING
# =============================================================================

def create_encoding_from_image(image_path: str, detector):
    """
    Detect the largest face in a registration image with YuNet, then
    generate its 128-D embedding with face_recognition. Returns None
    if no usable face is found.
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        return None

    height, width = image_bgr.shape[:2]

    faces = detect_faces(detector, image_bgr)

    # `faces` is a NumPy array when detections exist, and NumPy disallows
    # `if not array:` once it has more than one element - check length instead.
    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda face: float(face[2]) * float(face[3]))
    location = yunet_box_to_location(largest_face, width, height)

    if location is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        # YuNet already supplied the face location, so face_recognition
        # does not need to re-detect the face with HOG.
        encodings = face_recognition.face_encodings(
            image_rgb, [location], num_jitters=1, model="small"
        )
    except Exception as error:
        print(f"   Encoding error: {error}")
        return None

    return encodings[0] if encodings else None


def load_students(detector) -> dict:
    """Load every registered student folder and build one representative
    encoding per student (the encoding closest to that student's average)."""

    print()
    print("=" * 70)
    print("LOADING REGISTERED STUDENTS")
    print("=" * 70)
    print(f"\nDataset:\n{DATASET_DIR}")

    if not os.path.isdir(DATASET_DIR):
        print("\n[ERROR] Dataset folder does not exist.")
        return {}

    students = {}

    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, folder_name)

        if not os.path.isdir(folder):
            continue

        student_id, student_name = read_student_info(folder)

        print("\n" + "-" * 60)
        print(f"Student ID   : {student_id}")
        print(f"Student Name : {student_name}")

        image_files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        print(f"Face images  : {len(image_files)}")

        encodings = []

        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            print(f"Loading: {image_file}", end=" ")

            try:
                encoding = create_encoding_from_image(image_path, detector)

                if encoding is None:
                    print("-> No usable face")
                    continue

                encodings.append(encoding)
                print("-> YuNet OK")

            except Exception as error:
                print(f"-> ERROR\n   {error}")

        if not encodings:
            print(f"\n[SKIPPED] {student_name}\nNo usable face encodings.")
            continue

        # Pick the encoding closest to the student's average -
        # a real captured encoding, not a blended "ghost" vector.
        encoding_matrix = np.asarray(encodings, dtype=np.float64)
        average_encoding = np.mean(encoding_matrix, axis=0)
        distances = np.linalg.norm(encoding_matrix - average_encoding, axis=1)
        representative = encoding_matrix[int(np.argmin(distances))]

        students[student_id] = {"name": student_name, "encoding": representative}

        print(f"\n[REGISTERED] {student_name}")
        print(f"Usable images : {len(encodings)}")
        print("Stored vectors: 1")

    print("\n" + "=" * 70)
    print(f"Registered students loaded: {len(students)}")
    print("=" * 70)

    return students


def create_database(students: dict):
    """Flatten the students dict into parallel lists + a NumPy encoding matrix."""

    student_ids, student_names, encodings = [], [], []

    for student_id, data in students.items():
        student_ids.append(student_id)
        student_names.append(data["name"])
        encodings.append(data["encoding"])

    if not encodings:
        return [], [], np.empty((0, 128))

    return student_ids, student_names, np.asarray(encodings, dtype=np.float64)


def recognize(encoding, known_encodings, student_ids, student_names):
    """Match a live encoding against known students. Returns
    (student_id, student_name, distance) or (None, None, distance)."""

    if len(known_encodings) == 0:
        return None, None, None

    distances = face_recognition.face_distance(known_encodings, encoding)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance <= FACE_MATCH_THRESHOLD:
        return student_ids[best_index], student_names[best_index], best_distance

    return None, None, best_distance


# =============================================================================
# MAIN
# =============================================================================

def main():
    initialize_csv()

    detector = create_detector()
    if detector is None:
        return

    students = load_students(detector)
    if not students:
        print(
            "\nERROR: No valid registered students found.\n\n"
            "Expected:\n"
            "dataset/\n"
            "    STUDENT_ID_NAME/\n"
            "        face_01.jpg\n"
            "        face_02.jpg\n"
            "        face_03.jpg\n"
            "        face_04.jpg\n"
            "        face_05.jpg\n"
            "        student_info.txt"
        )
        return

    student_ids, student_names, known_encodings = create_database(students)

    print("\n" + "=" * 70)
    print("RECOGNITION DATABASE")
    print("=" * 70)
    print(f"Students: {len(student_ids)}")
    print("Stored encoding per student: 1")
    print(f"Total vectors: {len(known_encodings)}")
    print("=" * 70)

    marked_attendance = load_existing_attendance()

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 70)
    print("CLASSPULSE LIVE ATTENDANCE")
    print("=" * 70)
    print("YuNet live detection  : ON")
    print(f"Recognition interval  : {RECOGNITION_INTERVAL} frames")
    print(f"Required confirmations: {REQUIRED_CONFIRMATIONS}")
    print("Unknown faces         : IGNORED")
    print("\nPress Q to quit.")
    print("=" * 70)

    frame_count = 0
    active_period = None
    confirmation_counts = {}
    last_seen = {}
    display_results = {}

    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Camera frame failed.")
            break

        frame_count += 1
        period, start_time, end_time = get_current_period()

        # ---------------------------------------------------------------
        # Period changed -> reset per-period tracking state
        # ---------------------------------------------------------------
        if period != active_period:
            active_period = period
            confirmation_counts.clear()
            last_seen.clear()
            display_results.clear()

            print("\n" + "=" * 60)
            print(f"CURRENT PERIOD: {period}")

            if is_class_period(period):
                print(f"Time: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
                print("Attendance ACTIVE")
            else:
                print("Attendance PAUSED")

            print("=" * 60)

        faces = detect_faces(detector, frame)

        # ---------------------------------------------------------------
        # Recognition every N frames
        # ---------------------------------------------------------------
        if frame_count % RECOGNITION_INTERVAL == 0:
            display_results.clear()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]

            face_locations, valid_faces = [], []

            for index, face in enumerate(faces):
                location = yunet_box_to_location(face, frame_width, frame_height)
                if location is None:
                    continue
                face_locations.append(location)
                valid_faces.append((index, face))

            if face_locations:
                try:
                    encodings = face_recognition.face_encodings(
                        rgb, face_locations, num_jitters=1, model="small"
                    )
                except Exception as error:
                    print(f"[WARNING] Encoding failed: {error}")
                    encodings = []

                for position, encoding in enumerate(encodings):
                    if position >= len(valid_faces):
                        break

                    face_index, _ = valid_faces[position]
                    student_id, student_name, _ = recognize(
                        encoding, known_encodings, student_ids, student_names
                    )

                    if student_id is None:
                        display_results[face_index] = ("UNKNOWN", (0, 0, 255))
                        continue

                    display_results[face_index] = (student_name, (0, 255, 0))

                    confirmation_counts[student_id] = (
                        confirmation_counts.get(student_id, 0) + 1
                    )
                    last_seen[student_id] = frame_count

                    if is_class_period(period):
                        if confirmation_counts[student_id] >= REQUIRED_CONFIRMATIONS:
                            mark_attendance(
                                student_id, student_name, period,
                                start_time, end_time, marked_attendance,
                            )
                            # Stop the counter from growing further.
                            confirmation_counts[student_id] = REQUIRED_CONFIRMATIONS

        # ---------------------------------------------------------------
        # Draw detection boxes
        # ---------------------------------------------------------------
        for index, face in enumerate(faces):
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

            label, color = display_results.get(index, ("DETECTING...", (0, 255, 255)))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        # ---------------------------------------------------------------
        # Drop stale confirmations (student left the frame)
        # ---------------------------------------------------------------
        expired = [
            student_id
            for student_id, last_frame in last_seen.items()
            if frame_count - last_frame > RECOGNITION_INTERVAL * 3
        ]
        for student_id in expired:
            confirmation_counts.pop(student_id, None)
            last_seen.pop(student_id, None)

        # ---------------------------------------------------------------
        # UI overlay
        # ---------------------------------------------------------------
        height, width = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (width, 105), (25, 25, 25), -1)

        cv2.putText(
            frame, "ClassPulse - Live Attendance", (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2,
        )

        if is_class_period(period):
            period_text = f"{period} | {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
            period_color = (0, 255, 0)
        else:
            period_text = period
            period_color = (0, 255, 255)

        cv2.putText(
            frame, period_text, (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, period_color, 2,
        )
        cv2.putText(
            frame, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1,
        )
        cv2.putText(
            frame, f"Faces: {len(faces)}", (width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Students: {len(student_ids)}", (width - 200, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2,
        )

        status = "ACTIVE" if is_class_period(period) else "PAUSED"
        cv2.putText(
            frame, f"Attendance: {status}", (width - 200, 86),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, period_color, 1,
        )
        cv2.putText(
            frame, "Press Q to quit", (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1,
        )

        cv2.imshow("ClassPulse - Live Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\nClassPulse attendance session closed.")
    print(f"Attendance file: {ATTENDANCE_FILE}")


if __name__ == "__main__":
    main()