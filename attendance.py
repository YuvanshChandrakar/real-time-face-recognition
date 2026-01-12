import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

known_face_encodings = []
known_face_names = []

# ---------- LOAD KNOWN FACES ----------
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(KNOWN_FACES_DIR, filename)

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"Cannot read {filename}")
            continue

        # Force uint8 + contiguous
        image_bgr = np.ascontiguousarray(image_bgr, dtype=np.uint8)

        # Convert to RGB for encoding
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 🔑 GRAYSCALE for detection (THIS IS THE FIX)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        face_locations = face_recognition.face_locations(
            image_gray, model="hog"
        )

        if not face_locations:
            print(f"No face in {filename}")
            continue

        encoding = face_recognition.face_encodings(
            image_rgb, face_locations
        )[0]

        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])
        print(f"Loaded face: {filename}")

if not known_face_encodings:
    raise RuntimeError("No valid face images loaded")

print("Known faces:", known_face_names)

# ---------- WEBCAM ----------
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(gray, model="hog")
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.5
        )

        name = "Unknown"
        if True in matches:
            name = known_face_names[matches.index(True)]

            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            if not os.path.exists(ATTENDANCE_FILE):
                with open(ATTENDANCE_FILE, "w") as f:
                    f.write("Name,Date,Time\n")

            with open(ATTENDANCE_FILE, "a") as f:
                f.write(f"{name},{date},{time}\n")

            print(f"Attendance marked for {name}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
