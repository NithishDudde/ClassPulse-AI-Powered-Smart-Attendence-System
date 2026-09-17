"""
============================================================
                    CLASSPULSE
             STUDENT REGISTRATION MODULE
============================================================

Project structure:

ClassPulse/
│
├── backend/
│   └── student_registration.py
│
└── dataset/

Registered students are stored as:

ClassPulse/
└── dataset/
    └── STUDENTID_STUDENTNAME/
        ├── face_01.jpg
        ├── face_02.jpg
        ├── face_03.jpg
        ├── face_04.jpg
        ├── face_05.jpg
        └── student_info.txt

Registration flow:

Student Details
       ↓
Start Registration
       ↓
Camera opens
       ↓
Face detected
       ↓
Automatic countdown
       ↓
Photo captured
       ↓
OK → Save
RESET → Retake
QUIT → Cancel
       ↓
5 photos saved
       ↓
Registration Successful

============================================================
"""

import os
import re
import cv2
import time
import shutil
import tkinter as tk

from tkinter import messagebox
from PIL import Image, ImageTk


# ============================================================
# PROJECT PATH
# ============================================================

# student_registration.py:
#
# ClassPulse/
#     backend/
#         student_registration.py
#
# dataset:
#
# ClassPulse/
#     dataset/
#
# Therefore we go one level UP from backend.

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BACKEND_DIR
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset"
)


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CAMERA_UPDATE_MS = 30


# ============================================================
# REGISTRATION SETTINGS
# ============================================================

TOTAL_SAMPLES = 5

COUNTDOWN_SECONDS = 3

MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80


# ============================================================
# WINDOW SETTINGS
# ============================================================

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 820

MIN_WINDOW_WIDTH = 1100
MIN_WINDOW_HEIGHT = 720


# ============================================================
# FACE DETECTOR
# ============================================================

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# ============================================================
# CAPTURE POSITIONS
# ============================================================

POSES = [

    (
        "LOOK STRAIGHT",
        "Look directly at the camera.\n"
        "Keep your face centered."
    ),

    (
        "TURN SLIGHTLY LEFT",
        "Slowly turn your face slightly LEFT.\n"
        "Keep most of your face visible."
    ),

    (
        "TURN SLIGHTLY RIGHT",
        "Slowly turn your face slightly RIGHT.\n"
        "Keep most of your face visible."
    ),

    (
        "LOOK SLIGHTLY UP",
        "Look slightly upward.\n"
        "Keep your face inside the frame."
    ),

    (
        "LOOK SLIGHTLY DOWN",
        "Look slightly downward.\n"
        "Keep your face inside the frame."
    )

]


# ============================================================
# COLORS
# ============================================================

BG = "#F1F5F9"

HEADER = "#172554"

WHITE = "#FFFFFF"

BLACK = "#020617"

TEXT = "#0F172A"

SECONDARY = "#64748B"

BORDER = "#CBD5E1"

BLUE = "#2563EB"
BLUE_HOVER = "#1D4ED8"

GREEN = "#16A34A"
GREEN_HOVER = "#15803D"

ORANGE = "#F59E0B"
ORANGE_HOVER = "#D97706"

RED = "#DC2626"
RED_HOVER = "#B91C1C"

LIGHT_BLUE = "#DBEAFE"


# ============================================================
# STUDENT REGISTRATION CLASS
# ============================================================

class StudentRegistration:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        root
    ):

        self.root = root

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.root.title(
            "ClassPulse - Student Registration"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            MIN_WINDOW_WIDTH,
            MIN_WINDOW_HEIGHT
        )

        self.root.configure(
            bg=BG
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

        # ----------------------------------------------------
        # Student information
        # ----------------------------------------------------

        self.student_id = ""

        self.student_name = ""

        self.student_folder = None

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.camera = None

        self.camera_running = False

        self.camera_job = None

        # ----------------------------------------------------
        # Face detector
        # ----------------------------------------------------

        self.detector = cv2.CascadeClassifier(
            CASCADE_PATH
        )

        # ----------------------------------------------------
        # Registration state
        # ----------------------------------------------------

        self.sample_number = 1

        self.captured_image = None

        self.countdown_start = None

        self.capture_ready = False

        self.registration_complete = False

        # ----------------------------------------------------
        # Image reference
        # ----------------------------------------------------

        self.photo = None

        # ----------------------------------------------------
        # Make sure dataset exists
        # ----------------------------------------------------

        try:

            os.makedirs(
                DATASET_DIR,
                exist_ok=True
            )

        except OSError as error:

            messagebox.showerror(
                "Dataset Error",
                (
                    "Unable to create dataset folder.\n\n"
                    f"{error}"
                )
            )

            self.root.destroy()

            return

        # ----------------------------------------------------
        # Start with student information
        # ----------------------------------------------------

        self.show_student_form()


    # ========================================================
    # CLEAR WINDOW
    # ========================================================

    def clear_window(self):

        for widget in self.root.winfo_children():

            widget.destroy()


    # ========================================================
    # HEADER
    # ========================================================

    def create_header(
        self,
        title,
        subtitle
    ):

        header = tk.Frame(
            self.root,
            bg=HEADER,
            height=88
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(
            False
        )

        tk.Label(
            header,
            text=title,
            font=(
                "Segoe UI",
                22,
                "bold"
            ),
            bg=HEADER,
            fg=WHITE
        ).pack(
            pady=(13, 0)
        )

        tk.Label(
            header,
            text=subtitle,
            font=(
                "Segoe UI",
                10
            ),
            bg=HEADER,
            fg="#CBD5E1"
        ).pack(
            pady=(1, 0)
        )


    # ========================================================
    # BUTTON CREATOR
    # ========================================================

    def create_button(
        self,
        parent,
        text,
        command,
        color,
        hover_color
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            bg=color,
            fg=WHITE,
            activebackground=hover_color,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            cursor="hand2",
            height=2
        )

        button.bind(
            "<Enter>",
            lambda event: button.configure(
                bg=hover_color
            )
        )

        button.bind(
            "<Leave>",
            lambda event: button.configure(
                bg=color
            )
        )

        return button


    # ========================================================
    # STUDENT INFORMATION PAGE
    # ========================================================

    def show_student_form(self):

        self.stop_camera()

        self.clear_window()

        self.create_header(
            "ClassPulse",
            "Student Registration"
        )

        container = tk.Frame(
            self.root,
            bg=BG
        )

        container.pack(
            fill="both",
            expand=True
        )

        card = tk.Frame(
            container,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        card.place(
            relx=0.5,
            rely=0.5,
            anchor="center",
            width=560,
            height=430
        )

        tk.Label(
            card,
            text="Register New Student",
            font=(
                "Segoe UI",
                21,
                "bold"
            ),
            bg=WHITE,
            fg=TEXT
        ).pack(
            pady=(30, 5)
        )

        tk.Label(
            card,
            text=(
                "Enter student details before "
                "starting face registration."
            ),
            font=(
                "Segoe UI",
                10
            ),
            bg=WHITE,
            fg=SECONDARY
        ).pack(
            pady=(0, 25)
        )

        # ----------------------------------------------------
        # Student ID
        # ----------------------------------------------------

        tk.Label(
            card,
            text="Student ID",
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            bg=WHITE,
            fg=TEXT
        ).pack(
            anchor="w",
            padx=45
        )

        self.id_entry = tk.Entry(
            card,
            font=(
                "Segoe UI",
                12
            ),
            relief="solid",
            bd=1
        )

        self.id_entry.pack(
            fill="x",
            padx=45,
            ipady=8,
            pady=(5, 15)
        )

        # ----------------------------------------------------
        # Student Name
        # ----------------------------------------------------

        tk.Label(
            card,
            text="Student Name",
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            bg=WHITE,
            fg=TEXT
        ).pack(
            anchor="w",
            padx=45
        )

        self.name_entry = tk.Entry(
            card,
            font=(
                "Segoe UI",
                12
            ),
            relief="solid",
            bd=1
        )

        self.name_entry.pack(
            fill="x",
            padx=45,
            ipady=8,
            pady=(5, 22)
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        buttons = tk.Frame(
            card,
            bg=WHITE
        )

        buttons.pack()

        start_button = self.create_button(
            buttons,
            "START REGISTRATION",
            self.start_registration,
            BLUE,
            BLUE_HOVER
        )

        start_button.pack(
            side="left",
            padx=5
        )

        quit_button = self.create_button(
            buttons,
            "QUIT",
            self.close_application,
            RED,
            RED_HOVER
        )

        quit_button.pack(
            side="left",
            padx=5
        )

        self.id_entry.focus_set()


    # ========================================================
    # START REGISTRATION
    # ========================================================

    def start_registration(self):

        student_id = (
            self.id_entry
            .get()
            .strip()
        )

        student_name = (
            self.name_entry
            .get()
            .strip()
        )

        if not student_id:

            messagebox.showwarning(
                "Student ID Required",
                "Please enter Student ID."
            )

            return

        if not student_name:

            messagebox.showwarning(
                "Student Name Required",
                "Please enter Student Name."
            )

            return

        # ----------------------------------------------------
        # Clean folder names
        # ----------------------------------------------------

        safe_id = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            student_id
        )

        safe_name = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            student_name
        )

        self.student_id = safe_id

        self.student_name = student_name

        self.student_folder = os.path.join(
            DATASET_DIR,
            f"{safe_id}_{safe_name}"
        )

        # ----------------------------------------------------
        # Existing student
        # ----------------------------------------------------

        if os.path.exists(
            self.student_folder
        ):

            answer = messagebox.askyesno(
                "Student Already Registered",
                (
                    "This student already has "
                    "a registration.\n\n"
                    f"Student ID: {self.student_id}\n"
                    f"Name: {self.student_name}\n\n"
                    "Do you want to replace it?"
                )
            )

            if not answer:

                return

            try:

                shutil.rmtree(
                    self.student_folder
                )

            except OSError as error:

                messagebox.showerror(
                    "Folder Error",
                    (
                        "Could not remove the old "
                        "registration.\n\n"
                        f"{error}"
                    )
                )

                return

        # ----------------------------------------------------
        # Create student folder
        # ----------------------------------------------------

        try:

            os.makedirs(
                self.student_folder,
                exist_ok=True
            )

        except OSError as error:

            messagebox.showerror(
                "Dataset Error",
                (
                    "Could not create student "
                    "dataset folder.\n\n"
                    f"{error}"
                )
            )

            return

        # ----------------------------------------------------
        # Reset registration
        # ----------------------------------------------------

        self.sample_number = 1

        self.captured_image = None

        self.countdown_start = None

        self.capture_ready = False

        self.registration_complete = False

        # ----------------------------------------------------
        # Show registration UI
        # ----------------------------------------------------

        self.show_registration_page()

        # ----------------------------------------------------
        # Start camera
        # ----------------------------------------------------

        if not self.start_camera():

            self.cleanup_student_folder()

            return


    # ========================================================
    # REGISTRATION PAGE
    # ========================================================

    def show_registration_page(self):

        self.clear_window()

        self.create_header(
            "ClassPulse",
            "Student Face Registration"
        )

        # ----------------------------------------------------
        # Main container
        # ----------------------------------------------------

        main = tk.Frame(
            self.root,
            bg=BG
        )

        main.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=12
        )

        main.grid_columnconfigure(
            0,
            weight=1
        )

        main.grid_columnconfigure(
            1,
            weight=0,
            minsize=350
        )

        main.grid_rowconfigure(
            0,
            weight=1
        )

        # ====================================================
        # CAMERA PANEL
        # ====================================================

        camera_panel = tk.Frame(
            main,
            bg=BLACK,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        camera_panel.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        camera_panel.grid_rowconfigure(
            1,
            weight=1
        )

        camera_panel.grid_columnconfigure(
            0,
            weight=1
        )

        # ----------------------------------------------------
        # Camera title
        # ----------------------------------------------------

        camera_title = tk.Frame(
            camera_panel,
            bg=BLACK,
            height=42
        )

        camera_title.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        camera_title.grid_propagate(
            False
        )

        tk.Label(
            camera_title,
            text="●  LIVE CAMERA",
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            bg=BLACK,
            fg="#86EFAC"
        ).pack(
            side="left",
            padx=14
        )

        # ----------------------------------------------------
        # Camera display
        # ----------------------------------------------------

        self.camera_label = tk.Label(
            camera_panel,
            bg=BLACK
        )

        self.camera_label.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=8,
            pady=8
        )

        # ====================================================
        # RIGHT INFORMATION PANEL
        # ====================================================

        panel = tk.Frame(
            main,
            bg=WHITE,
            width=350,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        panel.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(12, 0)
        )

        panel.grid_propagate(
            False
        )

        # ----------------------------------------------------
        # Student profile
        # ----------------------------------------------------

        profile = tk.Frame(
            panel,
            bg=WHITE
        )

        profile.pack(
            fill="x",
            padx=18,
            pady=(15, 8)
        )

        tk.Label(
            profile,
            text="STUDENT PROFILE",
            font=(
                "Segoe UI",
                8,
                "bold"
            ),
            bg=WHITE,
            fg=SECONDARY
        ).pack(
            anchor="w"
        )

        tk.Label(
            profile,
            text=self.student_name,
            font=(
                "Segoe UI",
                17,
                "bold"
            ),
            bg=WHITE,
            fg=TEXT
        ).pack(
            anchor="w",
            pady=(2, 0)
        )

        tk.Label(
            profile,
            text=(
                "Student ID: "
                + self.student_id
            ),
            font=(
                "Segoe UI",
                9
            ),
            bg=WHITE,
            fg=SECONDARY
        ).pack(
            anchor="w"
        )

        # ----------------------------------------------------
        # Separator
        # ----------------------------------------------------

        tk.Frame(
            panel,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=18,
            pady=5
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress_frame = tk.Frame(
            panel,
            bg=WHITE
        )

        progress_frame.pack(
            fill="x",
            padx=18,
            pady=8
        )

        self.progress_label = tk.Label(
            progress_frame,
            text="PHOTO 1 OF 5",
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            bg=WHITE,
            fg=TEXT
        )

        self.progress_label.pack(
            anchor="w"
        )

        progress_bg = tk.Frame(
            progress_frame,
            bg="#E2E8F0",
            height=7
        )

        progress_bg.pack(
            fill="x",
            pady=(6, 0)
        )

        self.progress_fill = tk.Frame(
            progress_bg,
            bg=BLUE,
            height=7
        )

        self.progress_fill.place(
            x=0,
            y=0,
            relheight=1,
            relwidth=0.2
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        instruction_box = tk.Frame(
            panel,
            bg=LIGHT_BLUE,
            padx=13,
            pady=10
        )

        instruction_box.pack(
            fill="x",
            padx=18,
            pady=8
        )

        tk.Label(
            instruction_box,
            text="CAPTURE INSTRUCTION",
            font=(
                "Segoe UI",
                8,
                "bold"
            ),
            bg=LIGHT_BLUE,
            fg="#1D4ED8"
        ).pack(
            anchor="w"
        )

        self.pose_label = tk.Label(
            instruction_box,
            text=POSES[0][0],
            font=(
                "Segoe UI",
                14,
                "bold"
            ),
            bg=LIGHT_BLUE,
            fg="#1D4ED8"
        )

        self.pose_label.pack(
            anchor="w",
            pady=(2, 2)
        )

        self.instruction_label = tk.Label(
            instruction_box,
            text=POSES[0][1],
            font=(
                "Segoe UI",
                9
            ),
            bg=LIGHT_BLUE,
            fg="#334155",
            justify="left",
            wraplength=300
        )

        self.instruction_label.pack(
            anchor="w"
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status_frame = tk.Frame(
            panel,
            bg=WHITE
        )

        status_frame.pack(
            fill="x",
            padx=18,
            pady=7
        )

        self.status_dot = tk.Label(
            status_frame,
            text="●",
            font=(
                "Segoe UI",
                14
            ),
            bg=WHITE,
            fg=SECONDARY
        )

        self.status_dot.pack(
            side="left"
        )

        self.status_label = tk.Label(
            status_frame,
            text="Starting camera...",
            font=(
                "Segoe UI",
                9,
                "bold"
            ),
            bg=WHITE,
            fg=SECONDARY,
            justify="left",
            wraplength=290
        )

        self.status_label.pack(
            side="left",
            padx=(6, 0)
        )

        # ----------------------------------------------------
        # Bottom buttons
        # ----------------------------------------------------

        button_frame = tk.Frame(
            panel,
            bg=WHITE
        )

        button_frame.pack(
            side="bottom",
            fill="x",
            padx=18,
            pady=15
        )

        self.ok_button = self.create_button(
            button_frame,
            "✓  OK — SAVE PHOTO",
            self.accept_capture,
            GREEN,
            GREEN_HOVER
        )

        self.ok_button.pack(
            fill="x",
            pady=3
        )

        self.reset_button = self.create_button(
            button_frame,
            "↻  RESET — RETAKE",
            self.reset_capture,
            ORANGE,
            ORANGE_HOVER
        )

        self.reset_button.pack(
            fill="x",
            pady=3
        )

        self.quit_button = self.create_button(
            button_frame,
            "✕  QUIT REGISTRATION",
            self.cancel_registration,
            RED,
            RED_HOVER
        )

        self.quit_button.pack(
            fill="x",
            pady=3
        )

        self.set_review_buttons(
            False
        )


    # ========================================================
    # START CAMERA
    # ========================================================

    def start_camera(self):

        self.release_camera()

        try:

            self.camera = cv2.VideoCapture(
                CAMERA_INDEX
            )

            if not self.camera.isOpened():

                messagebox.showerror(
                    "Camera Error",
                    (
                        "Unable to open the webcam.\n\n"
                        "Please check camera permissions."
                    )
                )

                self.camera = None

                return False

            self.camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH
            )

            self.camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT
            )

            self.camera_running = True

            self.countdown_start = None

            self.update_camera()

            return True

        except Exception as error:

            messagebox.showerror(
                "Camera Error",
                str(error)
            )

            return False


    # ========================================================
    # UPDATE CAMERA
    # ========================================================

    def update_camera(self):

        if not self.camera_running:

            return

        if self.camera is None:

            return

        success, frame = self.camera.read()

        # ----------------------------------------------------
        # Camera read failed - retry on next tick
        # ----------------------------------------------------

        if not success:

            self.status_label.config(
                text="Unable to read camera frame.",
                fg=RED
            )

            if self.camera_running:

                self.camera_job = self.root.after(
                    CAMERA_UPDATE_MS,
                    self.update_camera
                )

            return

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Keep a clean, undrawn copy for saving BEFORE any
        # boxes / countdown overlays are drawn on the frame.
        # ----------------------------------------------------

        clean_frame = frame.copy()

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = self.detect_faces(
            frame
        )

        # ====================================================
        # NO FACE
        # ====================================================

        if len(faces) == 0:

            self.countdown_start = None

            self.status_dot.config(
                fg=RED
            )

            self.status_label.config(
                text=(
                    "No face detected.\n"
                    "Move into the camera frame."
                ),
                fg=RED
            )

            self.draw_guide(
                frame
            )

        # ====================================================
        # MULTIPLE FACES
        # ====================================================

        elif len(faces) > 1:

            self.countdown_start = None

            self.status_dot.config(
                fg=RED
            )

            self.status_label.config(
                text=(
                    "Multiple faces detected.\n"
                    "Only one student should be visible."
                ),
                fg=RED
            )

            for face in faces:

                self.draw_face_box(
                    frame,
                    face,
                    (0, 0, 255)
                )

        # ====================================================
        # ONE FACE
        # ====================================================

        else:

            face = faces[0]

            self.draw_face_box(
                frame,
                face,
                (0, 255, 0)
            )

            if not self.good_face(
                frame,
                face
            ):

                self.countdown_start = None

                self.status_dot.config(
                    fg=ORANGE
                )

                self.status_label.config(
                    text=(
                        "Face detected.\n"
                        "Move slightly closer."
                    ),
                    fg=ORANGE
                )

            else:

                self.status_dot.config(
                    fg=GREEN
                )

                # ------------------------------------------------
                # Start countdown
                # ------------------------------------------------

                if self.countdown_start is None:

                    self.countdown_start = time.time()

                elapsed = (
                    time.time()
                    - self.countdown_start
                )

                remaining = (
                    COUNTDOWN_SECONDS
                    - elapsed
                )

                if remaining > 0:

                    number = int(
                        remaining
                    ) + 1

                    self.status_label.config(
                        text=(
                            "Face detected.\n"
                            f"Hold position... {number}"
                        ),
                        fg=GREEN
                    )

                    self.draw_countdown(
                        frame,
                        number
                    )

                else:

                    # ------------------------------------------
                    # Use the clean, undrawn frame for saving
                    # ------------------------------------------

                    self.capture_photo(
                        clean_frame
                    )

                    return

        # ----------------------------------------------------
        # Display frame
        # ----------------------------------------------------

        self.display_frame(
            frame
        )

        # ----------------------------------------------------
        # Continue camera loop
        # ----------------------------------------------------

        if self.camera_running:

            self.camera_job = self.root.after(
                CAMERA_UPDATE_MS,
                self.update_camera
            )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces(
        self,
        frame
    ):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(
            gray
        )

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(
                MIN_FACE_WIDTH,
                MIN_FACE_HEIGHT
            )
        )

        return faces


    # ========================================================
    # FACE QUALITY CHECK
    # ========================================================

    def good_face(
        self,
        frame,
        face
    ):

        x, y, w, h = face

        frame_height, frame_width = (
            frame.shape[:2]
        )

        if w < MIN_FACE_WIDTH:

            return False

        if h < MIN_FACE_HEIGHT:

            return False

        if x < 5 or y < 5:

            return False

        if (
            x + w
            > frame_width - 5
        ):

            return False

        if (
            y + h
            > frame_height - 5
        ):

            return False

        return True


    # ========================================================
    # DRAW FACE BOX
    # ========================================================

    def draw_face_box(
        self,
        frame,
        face,
        color
    ):

        x, y, w, h = face

        cv2.rectangle(
            frame,
            (x, y),
            (
                x + w,
                y + h
            ),
            color,
            3
        )

        corner = 18

        # Top-left
        cv2.line(
            frame,
            (x, y),
            (
                x + corner,
                y
            ),
            color,
            5
        )

        cv2.line(
            frame,
            (x, y),
            (
                x,
                y + corner
            ),
            color,
            5
        )

        # Top-right
        cv2.line(
            frame,
            (
                x + w,
                y
            ),
            (
                x + w - corner,
                y
            ),
            color,
            5
        )

        cv2.line(
            frame,
            (
                x + w,
                y
            ),
            (
                x + w,
                y + corner
            ),
            color,
            5
        )

        # Bottom-left
        cv2.line(
            frame,
            (
                x,
                y + h
            ),
            (
                x + corner,
                y + h
            ),
            color,
            5
        )

        cv2.line(
            frame,
            (
                x,
                y + h
            ),
            (
                x,
                y + h - corner
            ),
            color,
            5
        )

        # Bottom-right
        cv2.line(
            frame,
            (
                x + w,
                y + h
            ),
            (
                x + w - corner,
                y + h
            ),
            color,
            5
        )

        cv2.line(
            frame,
            (
                x + w,
                y + h
            ),
            (
                x + w,
                y + h - corner
            ),
            color,
            5
        )


    # ========================================================
    # DRAW GUIDE
    # ========================================================

    def draw_guide(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        box_width = int(
            width * 0.32
        )

        box_height = int(
            height * 0.52
        )

        x = (
            width
            - box_width
        ) // 2

        y = (
            height
            - box_height
        ) // 2

        cv2.rectangle(
            frame,
            (x, y),
            (
                x + box_width,
                y + box_height
            ),
            (120, 120, 120),
            2
        )


    # ========================================================
    # COUNTDOWN DISPLAY
    # ========================================================

    def draw_countdown(
        self,
        frame,
        number
    ):

        height, width = frame.shape[:2]

        text = str(number)

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 4

        thickness = 8

        text_size = cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )[0]

        text_x = (
            width
            - text_size[0]
        ) // 2

        text_y = (
            height
            + text_size[1]
        ) // 2

        cv2.circle(
            frame,
            (
                width // 2,
                height // 2
            ),
            75,
            (15, 23, 42),
            -1
        )

        cv2.putText(
            frame,
            text,
            (
                text_x,
                text_y
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


    # ========================================================
    # CAPTURE PHOTO
    # ========================================================

    def capture_photo(
        self,
        frame
    ):

        # Save a copy for review.
        self.captured_image = frame.copy()

        # Stop live camera immediately.
        self.camera_running = False

        self.release_camera()

        self.countdown_start = None

        self.capture_ready = True

        self.set_review_buttons(
            True
        )

        self.status_dot.config(
            fg=GREEN
        )

        self.status_label.config(
            text=(
                "Photo captured.\n"
                "Review the image and choose "
                "OK or RESET."
            ),
            fg=TEXT
        )

        self.display_frame(
            self.captured_image
        )


    # ========================================================
    # DISPLAY CAMERA FRAME
    # ========================================================

    def display_frame(
        self,
        frame
    ):

        if not hasattr(
            self,
            "camera_label"
        ):

            return

        if not self.camera_label.winfo_exists():

            return

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        display_width = max(
            self.camera_label.winfo_width(),
            700
        )

        display_height = max(
            self.camera_label.winfo_height(),
            450
        )

        image.thumbnail(
            (
                display_width,
                display_height
            ),
            Image.Resampling.LANCZOS
        )

        self.photo = ImageTk.PhotoImage(
            image
        )

        self.camera_label.configure(
            image=self.photo
        )


    # ========================================================
    # REVIEW BUTTON STATE
    # ========================================================

    def set_review_buttons(
        self,
        enabled
    ):

        if not hasattr(
            self,
            "ok_button"
        ):

            return

        state = (
            "normal"
            if enabled
            else "disabled"
        )

        self.ok_button.config(
            state=state
        )

        self.reset_button.config(
            state=state
        )


    # ========================================================
    # OK - SAVE PHOTO
    # ========================================================

    def accept_capture(self):

        if not self.capture_ready:

            return

        if self.captured_image is None:

            return

        # ----------------------------------------------------
        # Create filename
        # ----------------------------------------------------

        filename = (
            f"face_{self.sample_number:02d}.jpg"
        )

        filepath = os.path.join(
            self.student_folder,
            filename
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        try:

            saved = cv2.imwrite(
                filepath,
                self.captured_image
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                (
                    "Could not save the photo.\n\n"
                    f"{error}"
                )
            )

            return

        # ----------------------------------------------------
        # Check cv2.imwrite()
        # ----------------------------------------------------

        if not saved:

            messagebox.showerror(
                "Save Failed",
                (
                    "The photo could not be saved.\n\n"
                    "Please try RESET."
                )
            )

            return

        # ----------------------------------------------------
        # Check actual file
        # ----------------------------------------------------

        if not os.path.isfile(
            filepath
        ):

            messagebox.showerror(
                "Save Verification Failed",
                (
                    "The image file was not found "
                    "after saving."
                )
            )

            return

        # ----------------------------------------------------
        # Check file size
        # ----------------------------------------------------

        if os.path.getsize(
            filepath
        ) <= 0:

            messagebox.showerror(
                "Invalid Image",
                (
                    "The saved image is empty.\n\n"
                    "Please use RESET."
                )
            )

            return

        # ====================================================
        # ALL 5 PHOTOS COMPLETE
        # ====================================================

        if (
            self.sample_number
            == TOTAL_SAMPLES
        ):

            self.complete_registration()

            return

        # ====================================================
        # NEXT PHOTO
        # ====================================================

        self.sample_number += 1

        self.captured_image = None

        self.capture_ready = False

        self.countdown_start = None

        self.update_pose()

        self.set_review_buttons(
            False
        )

        self.status_label.config(
            text=(
                "Photo saved successfully.\n"
                "Preparing next capture..."
            ),
            fg=GREEN
        )

        self.start_camera()


    # ========================================================
    # RESET - RETAKE
    # ========================================================

    def reset_capture(self):

        if not self.capture_ready:

            return

        self.captured_image = None

        self.capture_ready = False

        self.countdown_start = None

        self.set_review_buttons(
            False
        )

        self.status_label.config(
            text=(
                "Retaking this photo.\n"
                "Follow the instruction."
            ),
            fg=ORANGE
        )

        self.start_camera()


    # ========================================================
    # UPDATE POSE
    # ========================================================

    def update_pose(self):

        index = (
            self.sample_number - 1
        )

        title, description = POSES[
            index
        ]

        self.progress_label.config(
            text=(
                f"PHOTO {self.sample_number} "
                f"OF {TOTAL_SAMPLES}"
            )
        )

        self.pose_label.config(
            text=title
        )

        self.instruction_label.config(
            text=description
        )

        progress = (
            self.sample_number
            / TOTAL_SAMPLES
        )

        self.progress_fill.place(
            x=0,
            y=0,
            relheight=1,
            relwidth=progress
        )


    # ========================================================
    # COMPLETE REGISTRATION
    # ========================================================

    def complete_registration(self):

        self.stop_camera()

        # ----------------------------------------------------
        # Save student information
        # ----------------------------------------------------

        info_path = os.path.join(
            self.student_folder,
            "student_info.txt"
        )

        try:

            with open(
                info_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    f"Student ID: {self.student_id}\n"
                )

                file.write(
                    f"Student Name: {self.student_name}\n"
                )

                file.write(
                    f"Face Samples: {TOTAL_SAMPLES}\n"
                )

                file.write(
                    f"Dataset Folder: "
                    f"{self.student_folder}\n"
                )

        except OSError as error:

            messagebox.showerror(
                "Registration Error",
                (
                    "Face images were saved, "
                    "but student information "
                    "could not be saved.\n\n"
                    f"{error}"
                )
            )

            return

        # ----------------------------------------------------
        # Verify every image
        # ----------------------------------------------------

        saved_images = []

        for number in range(
            1,
            TOTAL_SAMPLES + 1
        ):

            filename = (
                f"face_{number:02d}.jpg"
            )

            path = os.path.join(
                self.student_folder,
                filename
            )

            if os.path.isfile(
                path
            ):

                if os.path.getsize(
                    path
                ) > 0:

                    saved_images.append(
                        path
                    )

        # ----------------------------------------------------
        # Do not report success unless all
        # 5 images actually exist.
        # ----------------------------------------------------

        if len(
            saved_images
        ) != TOTAL_SAMPLES:

            messagebox.showerror(
                "Registration Failed",
                (
                    "Registration could not "
                    "be verified.\n\n"
                    f"Expected: {TOTAL_SAMPLES}\n"
                    f"Found: {len(saved_images)}\n\n"
                    "Please check the dataset folder."
                )
            )

            return

        self.registration_complete = True

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        messagebox.showinfo(
            "Registration Successful",
            (
                "✓ Student registration completed!\n\n"
                f"Student ID: {self.student_id}\n"
                f"Student Name: {self.student_name}\n\n"
                f"Face Samples Saved: "
                f"{len(saved_images)}\n\n"
                "Saved to:\n"
                f"{self.student_folder}"
            )
        )

        # ----------------------------------------------------
        # Close window completely.
        # It does NOT return to registration page.
        # ----------------------------------------------------

        self.stop_camera()

        self.root.destroy()


    # ========================================================
    # CANCEL REGISTRATION
    # ========================================================

    def cancel_registration(self):

        answer = messagebox.askyesno(
            "Quit Registration",
            (
                "Are you sure you want to quit "
                "student registration?\n\n"
                "The incomplete registration "
                "will be deleted."
            )
        )

        if not answer:

            return

        # ----------------------------------------------------
        # Stop camera FIRST
        # ----------------------------------------------------

        self.stop_camera()

        # ----------------------------------------------------
        # Delete incomplete folder
        # ----------------------------------------------------

        self.cleanup_student_folder()

        # ----------------------------------------------------
        # Close application.
        # Do NOT show registration page again.
        # ----------------------------------------------------

        self.root.destroy()


    # ========================================================
    # CLEANUP STUDENT FOLDER
    # ========================================================

    def cleanup_student_folder(self):

        if (
            self.student_folder
            and os.path.exists(
                self.student_folder
            )
        ):

            try:

                shutil.rmtree(
                    self.student_folder
                )

            except OSError:

                pass


    # ========================================================
    # STOP CAMERA
    # ========================================================

    def stop_camera(self):

        self.camera_running = False

        # ----------------------------------------------------
        # Cancel scheduled camera update
        # ----------------------------------------------------

        if self.camera_job is not None:

            try:

                self.root.after_cancel(
                    self.camera_job
                )

            except Exception:

                pass

            self.camera_job = None

        # ----------------------------------------------------
        # Release webcam
        # ----------------------------------------------------

        self.release_camera()


    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    def release_camera(self):

        if self.camera is not None:

            try:

                self.camera.release()

            except Exception:

                pass

            self.camera = None


    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def close_application(self):

        # ----------------------------------------------------
        # Stop camera
        # ----------------------------------------------------

        self.stop_camera()

        # ----------------------------------------------------
        # If registration is incomplete,
        # ask before exiting.
        # ----------------------------------------------------

        if (
            not self.registration_complete
            and self.student_folder
        ):

            answer = messagebox.askyesno(
                "Exit Registration",
                (
                    "Do you want to exit "
                    "student registration?"
                )
            )

            if not answer:

                return

            self.cleanup_student_folder()

        # ----------------------------------------------------
        # Close application
        # ----------------------------------------------------

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    StudentRegistration(
        root
    )

    root.mainloop()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()