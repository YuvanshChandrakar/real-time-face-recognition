import cv2
import os
from datetime import datetime
from deepface import DeepFace

KNOWN_DIR = "known_faces"
ATT_FILE = "attendance.csv"

# Load known images
known_images = {}
for f in os.listdir(KNOWN_DIR):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(f)[0]
        known_images[name] = os.path.join(KNOWN_DIR, f)

if not known_images:
    raise RuntimeError("No images in known_faces folder")

print("Known faces:", list(known_images.keys()))

# Init webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def mark_attendance(name):
    if not os.path.exists(ATT_FILE):
        with open(ATT_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATT_FILE, "r+") as f:
        if any(line.startswith(name) and today in line for line in f.readlines()):
            return
        now = datetime.now()
        f.write(f"{name},{today},{now.strftime('%H:%M:%S')}\n")
        print(f"Attendance marked for {name}")
frame_count = 0
VERIFY_EVERY = 15  # verify once every 15 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save current frame temporarily
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame)

    label = "Unknown"
    try:
        for name, img_path in known_images.items():
            result = DeepFace.verify(
                img1_path=img_path,
                img2_path=temp_path,
                enforce_detection=False,
                detector_backend="opencv",
                model_name="Facenet"
            )
            if result["verified"]:
                label = name
                mark_attendance(name)
                break
    except Exception:
        pass

    # UI
    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("DeepFace Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("DeepFace Attendance", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

if os.path.exists("temp.jpg"):
    os.remove("temp.jpg")
