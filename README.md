# ClassPulse: AI-Powered Smart Attendance System

ClassPulse is an **AI-based smart attendance system** that automatically identifies registered students from a live webcam feed and records their attendance in a CSV file.

The system uses **OpenCV, YuNet face detection, and face recognition** to detect and recognize students. Instead of manually taking attendance, the system compares faces captured from the live camera with the registered student dataset and automatically marks recognized students as present.

---

## 📌 Project Overview

Traditional attendance systems require teachers to manually call student names or use separate attendance methods. This can take time and may result in errors.

ClassPulse aims to simplify this process by using **computer vision and face recognition**.

The current implementation provides:

* Live webcam-based face detection
* Student registration through webcam
* Multiple face images for each student
* Face matching with registered students
* Automatic attendance marking
* Prevention of repeated attendance marking
* Period-based attendance handling
* CSV attendance report generation
* Ignoring unregistered/unknown faces

---

## 🎯 Objectives

The main objectives of ClassPulse are:

1. To automate the student attendance process.
2. To identify students using face recognition.
3. To reduce manual attendance work for teachers.
4. To maintain attendance records automatically.
5. To store attendance information in a simple CSV format.
6. To provide a foundation for future smart classroom features.

---

## 🔄 System Workflow

```text
                    CLASS PULSE
                        │
                        ▼
                Student Registration
                        │
                        ▼
             Capture Student Face Images
                        │
                        ▼
                  Store Dataset
                        │
                        ▼
                Start Attendance
                        │
                        ▼
                 Live Webcam Feed
                        │
                        ▼
                 YuNet Face Detection
                        │
                        ▼
                  Face Recognition
                        │
                        ▼
             Compare With Dataset
                        │
              ┌─────────┴─────────┐
              │                   │
           Matched             Not Matched
              │                   │
              ▼                   ▼
       Confirm Student          Ignore
              │
              ▼
       Mark Attendance
              │
              ▼
       Generate CSV Record
```

---

## 🧠 How the System Works

### 1. Student Registration

A student is registered using the webcam.

The system captures multiple images of the student's face from different directions:

* Straight
* Left
* Right
* Up
* Down

The images are stored in the student's individual dataset folder.

Example:

```text
dataset/
│
├── 101_StudentName/
│   ├── face_01.jpg
│   ├── face_02.jpg
│   ├── face_03.jpg
│   ├── face_04.jpg
│   ├── face_05.jpg
│   └── student_info.txt
│
└── 102_StudentName/
    ├── face_01.jpg
    ├── face_02.jpg
    ├── face_03.jpg
    ├── face_04.jpg
    ├── face_05.jpg
    └── student_info.txt
```

Using multiple images helps the recognition system handle slight changes in the student's face orientation.

---

### 2. Live Camera Feed

During attendance, the webcam continuously captures live video frames.

```text
Webcam
   ↓
Live Video
   ↓
Individual Frames
```

Each frame is processed by the face detection module.

---

### 3. Face Detection

ClassPulse uses **YuNet**, a lightweight face detection model supported through OpenCV.

The detector identifies faces present in the camera frame.

```text
Live Frame
    ↓
YuNet
    ↓
Face Detected
    ↓
Face Region
```

The system can process multiple faces appearing in the classroom camera feed.

---

### 4. Face Recognition

After detecting a face, ClassPulse compares it with the registered student faces stored in the dataset.

```text
Live Face
    ↓
Face Features
    ↓
Compare with Registered Students
    ↓
Matching Score
    ↓
Student Identification
```

If a valid match is found, the corresponding student's identity is obtained.

If the face does not match any registered student, it is treated as an **unknown face** and is not marked as present.

---

### 5. Attendance Confirmation

To reduce accidental attendance marking, a recognized student must satisfy the configured recognition/confirmation conditions before attendance is recorded.

Once the student is successfully confirmed:

```text
Student Recognized
       ↓
Confirmation Successful
       ↓
Attendance Marked
```

The system also prevents the same student from being repeatedly marked during the same attendance session.

---

### 6. CSV Attendance Record

The attendance information is automatically stored in a CSV file.

Example:

```text
Student ID,Student Name,Date,Period,Start Time,End Time,Marked Time,Status
101,Nithish,13/09/2026,P1,09:30,10:20,09:35,Present
102,Rajesh,13/09/2026,P1,09:30,10:20,09:37,Present
```

This makes the attendance data easy to open and analyze using applications such as Microsoft Excel or Google Sheets.

---

# 🛠️ Technologies Used

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Main programming language                 |
| OpenCV           | Computer vision and camera processing     |
| YuNet            | Face detection                            |
| Face Recognition | Student face matching                     |
| NumPy            | Numerical and image processing operations |
| CSV              | Attendance data storage                   |
| FastAPI          | Backend/API foundation                    |
| Uvicorn          | Running the FastAPI application           |

---

# 📂 Project Structure

```text
ClassPulse/
│
├── backend/
│   ├── student_registration.py
│   ├── attendance_recognition.py
│   ├── face_detection.py
│   └── attendance.csv
│
├── dataset/
│   ├── StudentID_Name/
│   │   ├── face_01.jpg
│   │   ├── face_02.jpg
│   │   ├── face_03.jpg
│   │   ├── face_04.jpg
│   │   ├── face_05.jpg
│   │   └── student_info.txt
│   │
│   └── ...
│
├── models/
│   └── face_detection_yunet_2023mar.onnx

```

> The exact file structure may change as the project is further developed.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ClassPulse.git
```

Move into the project directory:

```bash
cd ClassPulse
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install opencv-python numpy face-recognition fastapi uvicorn
```

> Depending on the operating system, installing `face-recognition` may require additional dependencies.

---

# ▶️ Running the Project

## Step 1 — Student Registration

Run the student registration program:

```bash
python backend/student_registration.py
```

Enter the student's:

```text
Student ID
Student Name
```

Then capture the required face images using the webcam.

The images will be stored inside the `dataset` directory.

---

## Step 2 — Start Attendance Recognition

Run:

```bash
python backend/attendance_recognition.py
```

The webcam will start and the system will begin detecting and recognizing registered students.

When a registered student is successfully recognized and confirmed, their attendance will be recorded.

---

# 📊 Attendance Output

The generated attendance file is:

```text
backend/attendance.csv
```

The CSV contains information such as:

* Student ID
* Student Name
* Date
* Period
* Start Time
* End Time
* Marked Time
* Attendance Status

Example:

```text
101,Nithish,13/09/2026,P1,09:30,10:20,09:35,Present
```

---

# 👤 Unknown Student Handling

ClassPulse does not mark every detected face as attendance.

The process is:

```text
Face Detected
      ↓
Compare With Registered Dataset
      ↓
      ├── Match → Confirm → Mark Present
      │
      └── No Match → Unknown → Ignore
```

This prevents unregistered individuals from being incorrectly added to the attendance record.

---

# 🕐 Period-Based Attendance

The system can associate attendance with classroom periods.

For example:

```text
P1 → 09:30 – 10:20
P2 → 10:20 – 11:10
P3 → 11:20 – 12:10
P4 → 12:10 – 13:00
P5 → 13:40 – 14:30
P6 → 14:30 – 15:20
P7 → 15:20 – 16:10
```

The timetable can be modified according to the institution's schedule.

---

### Model File

The YuNet ONNX model should be placed inside:

```text
models/
```

Make sure the path used in the Python code matches the actual model location.

---

# 🚀 Current Features

* [x] Live webcam feed
* [x] Student registration
* [x] Multiple face-image capture
* [x] Face detection
* [x] Face recognition
* [x] Registered student matching
* [x] Unknown face handling
* [x] Attendance confirmation
* [x] Duplicate attendance prevention
* [x] Period-based attendance
* [x] CSV attendance generation

---

# 🔮 Future Development

ClassPulse is designed to be extended into a broader smart classroom monitoring platform.

Planned features include:

* IoT-based classroom monitoring
* Environmental sensor integration
* Classroom occupancy monitoring
* Web-based attendance dashboard
* Attendance analytics
* Student attendance reports
* Automated report generation
* Additional AI-based classroom monitoring features

These features are **not part of the current completed version** and will be added in future development stages.

---

# 🎓 Project Information

**Project Name:** ClassPulse: AI-Powered Smart Attendance System

**Domain:** Artificial Intelligence / Computer Vision

**Primary Technologies:** Python, OpenCV, YuNet, Face Recognition

**Application Area:** Smart Classroom / Automated Attendance

---

# 👨‍💻 Author

**Nithish Dudde**

B.Tech — Computer Science and Engineering (AI & ML)

---

# 📄 License

This project is developed for **academic and educational purposes**.
