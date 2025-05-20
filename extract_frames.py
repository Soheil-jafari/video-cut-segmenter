
import cv2
import os

def extract_frames(video_path, output_folder, max_frames=60):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open video:", video_path)
        return
    else:
        print("✅ Opened video successfully")

    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to {output_folder}")
