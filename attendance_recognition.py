import os
import csv
from datetime import datetime, time

import cv2
import numpy as np
import face_recognition


# ============================================================
# CLASSPULSE - ATTENDANCE RECOGNITION
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

# ClassPulse/
#     backend/
#     dataset/
#     models/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx"
)

ATTENDANCE_FILE = os.path.join(
    BACKEND_DIR,
    "attendance.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

YUNET_SCORE_THRESHOLD = 0.60
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

# Lower value = stricter recognition
# Higher value = more tolerant recognition
FACE_MATCH_THRESHOLD = 0.50

# Perform recognition every N frames
RECOGNITION_INTERVAL = 5

# Number of successful recognitions required
# before attendance is marked
REQUIRED_CONFIRMATIONS = 3

# Ignore extremely small faces
MIN_FACE_SIZE = 40


# ============================================================
# 7-PERIOD TIMETABLE
# ============================================================

TIMETABLE = [

    ("Period 1", time(9, 30), time(10, 20)),

    ("Period 2", time(10, 20), time(11, 10)),

    ("Toilet Break", time(11, 10), time(11, 20)),

    ("Period 3", time(11, 20), time(12, 10)),

    ("Period 4", time(12, 10), time(13, 0)),

    ("Lunch Break", time(13, 0), time(13, 40)),

    ("Period 5", time(13, 40), time(14, 30)),

    ("Period 6", time(14, 30), time(15, 20)),

    ("Period 7", time(15, 20), time(16, 10)),
]


# ============================================================
# ATTENDANCE CSV
# ============================================================

CSV_HEADER = [
    "Student ID",
    "Student Name",
    "Date",
    "Period",
    "Start Time",
    "End Time",
    "Marked Time",
    "Status"
]


def initialize_csv():

    if not os.path.exists(ATTENDANCE_FILE):

        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                CSV_HEADER
            )


# ============================================================
# GET CURRENT PERIOD
# ============================================================

def get_current_period():

    current = datetime.now().time()

    for name, start, end in TIMETABLE:

        if start <= current < end:

            return (
                name,
                start,
                end
            )

    if current < TIMETABLE[0][1]:

        return (
            "Before Classes",
            None,
            None
        )

    return (
        "After Classes",
        None,
        None
    )


def is_class_period(period):

    return period.startswith("Period")


# ============================================================
# LOAD ALREADY MARKED ATTENDANCE
# ============================================================

def load_existing_attendance():

    marked = set()

    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        return marked

    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                student_id = row.get(
                    "Student ID",
                    ""
                ).strip()

                date = row.get(
                    "Date",
                    ""
                ).strip()

                period = row.get(
                    "Period",
                    ""
                ).strip()

                status = row.get(
                    "Status",
                    ""
                ).strip()

                if (
                    student_id
                    and date
                    and period
                    and status.lower() == "present"
                ):

                    marked.add(
                        (
                            date,
                            period,
                            student_id
                        )
                    )

    except Exception as error:

        print(
            "[WARNING] Could not read attendance.csv:"
        )

        print(error)

    return marked


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(
    student_id,
    student_name,
    period,
    start,
    end,
    marked
):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    key = (
        today,
        period,
        student_id
    )

    # Already marked
    if key in marked:

        return False

    now = datetime.now().strftime(
        "%H:%M:%S"
    )

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            student_id,
            student_name,
            today,
            period,
            start.strftime("%H:%M"),
            end.strftime("%H:%M"),
            now,
            "Present"
        ])

    marked.add(key)

    print()
    print("=" * 60)
    print("ATTENDANCE MARKED")
    print("=" * 60)

    print(
        f"Student ID   : {student_id}"
    )

    print(
        f"Student Name : {student_name}"
    )

    print(
        f"Period       : {period}"
    )

    print(
        f"Marked Time  : {now}"
    )

    print("=" * 60)

    return True


# ============================================================
# READ STUDENT INFORMATION
# ============================================================

def read_student_info(folder):

    folder_name = os.path.basename(
        folder
    )

    student_id = ""
    student_name = ""

    info_file = os.path.join(
        folder,
        "student_info.txt"
    )

    if os.path.exists(info_file):

        try:

            with open(
                info_file,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    line = line.strip()

                    lower = line.lower()

                    if lower.startswith(
                        "student id:"
                    ):

                        student_id = (
                            line.split(
                                ":",
                                1
                            )[1].strip()
                        )

                    elif lower.startswith(
                        "student name:"
                    ):

                        student_name = (
                            line.split(
                                ":",
                                1
                            )[1].strip()
                        )

        except Exception as error:

            print(
                f"[WARNING] Cannot read {info_file}"
            )

            print(error)

    # --------------------------------------------------------
    # Fallback to folder name
    # --------------------------------------------------------

    if not student_id or not student_name:

        if "_" in folder_name:

            parts = folder_name.split(
                "_",
                1
            )

            if not student_id:

                student_id = parts[0]

            if not student_name:

                student_name = (
                    parts[1]
                    .replace("_", " ")
                    .strip()
                )

        else:

            if not student_id:

                student_id = folder_name

            if not student_name:

                student_name = folder_name

    return (
        student_id,
        student_name
    )


# ============================================================
# CREATE YUNET DETECTOR
# ============================================================

def create_detector():

    print()
    print("Checking YuNet model...")

    if not os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            "[ERROR] YuNet model not found:"
        )

        print(
            MODEL_PATH
        )

        return None

    try:

        detector = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K
        )

        print(
            "[OK] YuNet model loaded."
        )

        return detector

    except Exception as error:

        print(
            "[ERROR] YuNet initialization failed."
        )

        print(error)

        return None


# ============================================================
# YUNET FACE DETECTION
# ============================================================

def detect_faces(
    detector,
    frame
):

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(
        frame
    )

    if faces is None:

        return []

    return faces


# ============================================================
# CONVERT YUNET BOX TO FACE LOCATION
#
# face_recognition format:
#
# (top, right, bottom, left)
# ============================================================

def yunet_box_to_location(
    face,
    frame_width,
    frame_height
):

    x = int(face[0])
    y = int(face[1])
    w = int(face[2])
    h = int(face[3])

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:

        return None

    left = max(
        0,
        x
    )

    top = max(
        0,
        y
    )

    right = min(
        frame_width,
        x + w
    )

    bottom = min(
        frame_height,
        y + h
    )

    if right <= left or bottom <= top:

        return None

    return (
        top,
        right,
        bottom,
        left
    )


# ============================================================
# LOAD FACE ENCODING FROM REGISTRATION IMAGE
#
# IMPORTANT:
#
# YuNet detects the face.
# face_recognition generates the 128-D embedding.
#
# No HOG detection is used.
# ============================================================

def create_encoding_from_image(
    image_path,
    detector
):

    image_bgr = cv2.imread(
        image_path
    )

    if image_bgr is None:

        return None

    height, width = image_bgr.shape[:2]

    # --------------------------------------------------------
    # YuNet detection
    # --------------------------------------------------------

    faces = detect_faces(
        detector,
        image_bgr
    )

    # ----------------------------------------------------------
    # FIX:
    # `faces` is a NumPy array when detections exist, and NumPy
    # does not allow `if not array:` once it has more than one
    # element ("truth value of an array with more than one
    # element is ambiguous"). Check length instead of truthiness.
    # ----------------------------------------------------------

    if len(faces) == 0:

        return None

    # --------------------------------------------------------
    # Select largest face
    # --------------------------------------------------------

    largest_face = max(
        faces,
        key=lambda face:
        float(face[2]) * float(face[3])
    )

    location = yunet_box_to_location(
        largest_face,
        width,
        height
    )

    if location is None:

        return None

    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Create encoding
    #
    # We provide YuNet's face location directly.
    # Therefore face_recognition does NOT need
    # to detect the face again using HOG.
    # --------------------------------------------------------

    try:

        encodings = face_recognition.face_encodings(
            image_rgb,
            [location],
            num_jitters=1,
            model="small"
        )

    except Exception as error:

        print(
            f"   Encoding error: {error}"
        )

        return None

    if not encodings:

        return None

    return encodings[0]


# ============================================================
# LOAD ALL REGISTERED STUDENTS
# ============================================================

def load_students(detector):

    print()
    print("=" * 70)
    print("LOADING REGISTERED STUDENTS")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    if not os.path.isdir(
        DATASET_DIR
    ):

        print()
        print(
            "[ERROR] Dataset folder does not exist."
        )

        return {}

    students = {}

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    folders = sorted(
        os.listdir(
            DATASET_DIR
        )
    )

    for folder_name in folders:

        folder = os.path.join(
            DATASET_DIR,
            folder_name
        )

        if not os.path.isdir(folder):

            continue

        student_id, student_name = (
            read_student_info(
                folder
            )
        )

        print()
        print("-" * 60)

        print(
            f"Student ID   : {student_id}"
        )

        print(
            f"Student Name : {student_name}"
        )

        image_files = sorted([
            f
            for f in os.listdir(folder)
            if f.lower().endswith(
                image_extensions
            )
        ])

        print(
            f"Face images  : {len(image_files)}"
        )

        encodings = []

        # ----------------------------------------------------
        # Process registration images
        # ----------------------------------------------------

        for image_file in image_files:

            image_path = os.path.join(
                folder,
                image_file
            )

            print(
                f"Loading: {image_file}",
                end=" "
            )

            try:

                encoding = (
                    create_encoding_from_image(
                        image_path,
                        detector
                    )
                )

                if encoding is None:

                    print(
                        "-> No usable face"
                    )

                    continue

                encodings.append(
                    encoding
                )

                print(
                    "-> YuNet OK"
                )

            except Exception as error:

                print(
                    "-> ERROR"
                )

                print(
                    f"   {error}"
                )

        # ----------------------------------------------------
        # No usable images
        # ----------------------------------------------------

        if not encodings:

            print()
            print(
                f"[SKIPPED] {student_name}"
            )

            print(
                "No usable face encodings."
            )

            continue

        # ----------------------------------------------------
        # 5 encodings -> average representation
        # ----------------------------------------------------

        encoding_matrix = np.asarray(
            encodings,
            dtype=np.float64
        )

        average_encoding = np.mean(
            encoding_matrix,
            axis=0
        )

        # ----------------------------------------------------
        # Find encoding closest to average
        # ----------------------------------------------------

        distances = np.linalg.norm(
            encoding_matrix
            -
            average_encoding,
            axis=1
        )

        best_index = int(
            np.argmin(
                distances
            )
        )

        representative = (
            encoding_matrix[
                best_index
            ]
        )

        students[student_id] = {
            "name": student_name,
            "encoding": representative
        }

        print()
        print(
            f"[REGISTERED] {student_name}"
        )

        print(
            f"Usable images : {len(encodings)}"
        )

        print(
            "Stored vectors: 1"
        )

    print()
    print("=" * 70)

    print(
        f"Registered students loaded: "
        f"{len(students)}"
    )

    print("=" * 70)

    return students


# ============================================================
# CREATE OPTIMIZED NUMPY DATABASE
# ============================================================

def create_database(students):

    student_ids = []
    student_names = []
    encodings = []

    for student_id, data in students.items():

        student_ids.append(
            student_id
        )

        student_names.append(
            data["name"]
        )

        encodings.append(
            data["encoding"]
        )

    if not encodings:

        return (
            [],
            [],
            np.empty(
                (0, 128)
            )
        )

    return (
        student_ids,
        student_names,
        np.asarray(
            encodings,
            dtype=np.float64
        )
    )


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize(
    encoding,
    known_encodings,
    student_ids,
    student_names
):

    if len(known_encodings) == 0:

        return (
            None,
            None,
            None
        )

    distances = (
        face_recognition.face_distance(
            known_encodings,
            encoding
        )
    )

    best_index = int(
        np.argmin(
            distances
        )
    )

    best_distance = float(
        distances[best_index]
    )

    if (
        best_distance
        <=
        FACE_MATCH_THRESHOLD
    ):

        return (
            student_ids[best_index],
            student_names[best_index],
            best_distance
        )

    return (
        None,
        None,
        best_distance
    )


# ============================================================
# MAIN
# ============================================================

def main():

    initialize_csv()

    # ========================================================
    # CREATE YUNET
    # ========================================================

    detector = create_detector()

    if detector is None:

        return

    # ========================================================
    # LOAD REGISTERED STUDENTS
    # ========================================================

    students = load_students(
        detector
    )

    if not students:

        print()
        print(
            "ERROR: No valid registered students found."
        )

        print()
        print(
            "Expected:"
        )

        print(
            "dataset/"
        )

        print(
            "    STUDENT_ID_NAME/"
        )

        print(
            "        face_01.jpg"
        )

        print(
            "        face_02.jpg"
        )

        print(
            "        face_03.jpg"
        )

        print(
            "        face_04.jpg"
        )

        print(
            "        face_05.jpg"
        )

        print(
            "        student_info.txt"
        )

        return

    # ========================================================
    # CREATE OPTIMIZED DATABASE
    # ========================================================

    (
        student_ids,
        student_names,
        known_encodings
    ) = create_database(
        students
    )

    print()
    print("=" * 70)
    print("RECOGNITION DATABASE")
    print("=" * 70)

    print(
        f"Students: {len(student_ids)}"
    )

    print(
        "Stored encoding per student: 1"
    )

    print(
        f"Total vectors: {len(known_encodings)}"
    )

    print("=" * 70)

    # ========================================================
    # LOAD EXISTING ATTENDANCE
    # ========================================================

    marked_attendance = (
        load_existing_attendance()
    )

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "[ERROR] Cannot open webcam."
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print()
    print("=" * 70)
    print("CLASSPULSE LIVE ATTENDANCE")
    print("=" * 70)

    print(
        "YuNet live detection : ON"
    )

    print(
        "Recognition interval  : "
        f"{RECOGNITION_INTERVAL} frames"
    )

    print(
        "Required confirmations: "
        f"{REQUIRED_CONFIRMATIONS}"
    )

    print(
        "Unknown faces         : IGNORED"
    )

    print()
    print(
        "Press Q to quit."
    )

    print("=" * 70)

    # ========================================================
    # RUNTIME VARIABLES
    # ========================================================

    frame_count = 0

    active_period = None

    confirmation_counts = {}

    last_seen = {}

    display_results = {}

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "[ERROR] Camera frame failed."
            )

            break

        frame_count += 1

        # ====================================================
        # CURRENT PERIOD
        # ====================================================

        (
            period,
            start_time,
            end_time
        ) = get_current_period()

        # ====================================================
        # PERIOD CHANGED
        # ====================================================

        if period != active_period:

            active_period = period

            confirmation_counts.clear()

            last_seen.clear()

            display_results.clear()

            print()
            print(
                "=" * 60
            )

            print(
                f"CURRENT PERIOD: {period}"
            )

            if is_class_period(period):

                print(
                    f"Time: "
                    f"{start_time.strftime('%H:%M')}"
                    f" - "
                    f"{end_time.strftime('%H:%M')}"
                )

                print(
                    "Attendance ACTIVE"
                )

            else:

                print(
                    "Attendance PAUSED"
                )

            print(
                "=" * 60
            )

        # ====================================================
        # YUNET DETECTION
        # ====================================================

        faces = detect_faces(
            detector,
            frame
        )

        # ====================================================
        # RECOGNITION EVERY N FRAMES
        # ====================================================

        if (
            frame_count
            %
            RECOGNITION_INTERVAL
            ==
            0
        ):

            display_results.clear()

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            face_locations = []

            valid_faces = []

            # ------------------------------------------------
            # Convert YuNet boxes
            # ------------------------------------------------

            for index, face in enumerate(
                faces
            ):

                location = (
                    yunet_box_to_location(
                        face,
                        frame_width,
                        frame_height
                    )
                )

                if location is None:

                    continue

                face_locations.append(
                    location
                )

                valid_faces.append(
                    (
                        index,
                        face
                    )
                )

            # ------------------------------------------------
            # Generate embeddings
            # ------------------------------------------------

            if face_locations:

                try:

                    encodings = (
                        face_recognition.face_encodings(
                            rgb,
                            face_locations,
                            num_jitters=1,
                            model="small"
                        )
                    )

                except Exception as error:

                    print(
                        "[WARNING] Encoding failed:"
                    )

                    print(error)

                    encodings = []

                # ------------------------------------------------
                # Match faces
                # ------------------------------------------------

                for position, encoding in enumerate(
                    encodings
                ):

                    if position >= len(
                        valid_faces
                    ):

                        break

                    face_index, face = (
                        valid_faces[position]
                    )

                    (
                        student_id,
                        student_name,
                        distance
                    ) = recognize(
                        encoding,
                        known_encodings,
                        student_ids,
                        student_names
                    )

                    # ============================================
                    # UNKNOWN FACE
                    # ============================================

                    if student_id is None:

                        # Unknown faces are ignored
                        # visually marked red only

                        display_results[
                            face_index
                        ] = (
                            "UNKNOWN",
                            (0, 0, 255)
                        )

                        continue

                    # ============================================
                    # KNOWN FACE
                    # ============================================

                    display_results[
                        face_index
                    ] = (
                        student_name,
                        (0, 255, 0)
                    )

                    # ============================================
                    # CONFIRMATION
                    # ============================================

                    confirmation_counts[
                        student_id
                    ] = (
                        confirmation_counts.get(
                            student_id,
                            0
                        )
                        +
                        1
                    )

                    last_seen[
                        student_id
                    ] = frame_count

                    # ============================================
                    # ATTENDANCE
                    # ============================================

                    if is_class_period(
                        period
                    ):

                        if (
                            confirmation_counts[
                                student_id
                            ]
                            >=
                            REQUIRED_CONFIRMATIONS
                        ):

                            mark_attendance(
                                student_id,
                                student_name,
                                period,
                                start_time,
                                end_time,
                                marked_attendance
                            )

                            # Stop counter growing
                            confirmation_counts[
                                student_id
                            ] = (
                                REQUIRED_CONFIRMATIONS
                            )

        # ====================================================
        # DRAW YUNET BOXES
        # ====================================================

        for index, face in enumerate(
            faces
        ):

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            result = display_results.get(
                index
            )

            if result is None:

                label = "DETECTING..."

                color = (
                    0,
                    255,
                    255
                )

            else:

                label, color = result

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(
                        25,
                        y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # ====================================================
        # REMOVE STALE CONFIRMATIONS
        # ====================================================

        expired = []

        for student_id, last_frame in (
            last_seen.items()
        ):

            if (
                frame_count
                -
                last_frame
                >
                RECOGNITION_INTERVAL * 3
            ):

                expired.append(
                    student_id
                )

        for student_id in expired:

            confirmation_counts.pop(
                student_id,
                None
            )

            last_seen.pop(
                student_id,
                None
            )

        # ====================================================
        # USER INTERFACE
        # ====================================================

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 105),
            (25, 25, 25),
            -1
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "ClassPulse - Live Attendance",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Period
        # ----------------------------------------------------

        if is_class_period(period):

            period_text = (
                f"{period} | "
                f"{start_time.strftime('%H:%M')}"
                f" - "
                f"{end_time.strftime('%H:%M')}"
            )

            period_color = (
                0,
                255,
                0
            )

        else:

            period_text = period

            period_color = (
                0,
                255,
                255
            )

        cv2.putText(
            frame,
            period_text,
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            period_color,
            2
        )

        # ----------------------------------------------------
        # Date and time
        # ----------------------------------------------------

        cv2.putText(
            frame,
            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )

        # ----------------------------------------------------
        # Face count
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (
                width - 200,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Registered students
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Students: {len(student_ids)}",
            (
                width - 200,
                58
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Attendance status
        # ----------------------------------------------------

        status = (
            "ACTIVE"
            if is_class_period(period)
            else "PAUSED"
        )

        cv2.putText(
            frame,
            f"Attendance: {status}",
            (
                width - 200,
                86
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            period_color,
            1
        )

        # ----------------------------------------------------
        # Bottom instruction
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                20,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "ClassPulse - Live Attendance",
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    print()
    print(
        "ClassPulse attendance session closed."
    )

    print(
        f"Attendance file: {ATTENDANCE_FILE}"
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()