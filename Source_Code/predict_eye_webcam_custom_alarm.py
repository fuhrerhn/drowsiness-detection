"""Deteksi mata tertutup lewat webcam dengan CNN dan alarm audio yang berulang."""

from pathlib import Path
import time

import cv2
import mediapipe as mp
import pygame
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


# Arsitektur harus sama dengan model pada notebook training.
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


# Konfigurasi
LEFT_EYE = [362, 263, 386, 374, 385, 380]
RIGHT_EYE = [33, 133, 159, 145, 160, 144]

ALARM_DELAY = 2.0
SMOOTHING_FACTOR = 0.25
CLOSED_THRESHOLD = 0.60
OPEN_THRESHOLD = 0.40
SHOW_EYE_PREVIEW = True

project_folder = Path(__file__).resolve().parent
model_path = project_folder / "eye_cnn_best.pth"
alarm_path = project_folder / "alarm.mp3"  # Bisa diganti dengan "alarm.wav"


def crop_eye(frame, landmarks, eye_indices):
    """Ambil gambar mata dengan ukuran crop berdasarkan lebar mata."""
    frame_height, frame_width = frame.shape[:2]
    x_positions = [landmarks[index].x * frame_width for index in eye_indices]
    y_positions = [landmarks[index].y * frame_height for index in eye_indices]

    eye_width = max(x_positions) - min(x_positions)
    if eye_width < 5:
        return None, None

    center_x = (min(x_positions) + max(x_positions)) / 2
    center_y = (min(y_positions) + max(y_positions)) / 2
    crop_width = eye_width * 1.55
    crop_height = eye_width * 0.90

    left = max(0, int(center_x - crop_width / 2))
    top = max(0, int(center_y - crop_height / 2))
    right = min(frame_width, int(center_x + crop_width / 2))
    bottom = min(frame_height, int(center_y + crop_height / 2))

    if right <= left or bottom <= top:
        return None, None

    return frame[top:bottom, left:right], (left, top, right, bottom)


# Memuat model CNN hasil training.
if not model_path.exists():
    raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
if not alarm_path.exists():
    raise FileNotFoundError(f"File alarm tidak ditemukan: {alarm_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
class_names = checkpoint["classes"]
image_size = checkpoint["image_size"]
if "sleepy" not in class_names:
    raise ValueError(f"Label sleepy tidak ada pada model: {class_names}")

model = EyeCNN().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
sleepy_class_index = class_names.index("sleepy")

# Preprocessing harus sama seperti saat training.
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

pygame.mixer.init()
pygame.mixer.music.load(str(alarm_path))

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)
if not camera.isOpened():
    pygame.mixer.quit()
    raise RuntimeError("Webcam tidak dapat dibuka")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

smoothed_sleepy_score = None
eye_status = "OPEN"
closed_start_time = None
alarm_playing = False

print("Model:", model_path.name)
print("Alarm:", alarm_path.name)
print("Device:", device)
print("Tekan T untuk tes audio, Q untuk keluar")

try:
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_result = face_mesh.process(rgb_frame)
            eye_images = []

            if face_result.multi_face_landmarks:
                landmarks = face_result.multi_face_landmarks[0].landmark

                for eye_number, eye_indices in enumerate([LEFT_EYE, RIGHT_EYE], start=1):
                    eye_crop, rectangle = crop_eye(frame, landmarks, eye_indices)
                    if eye_crop is None:
                        break

                    eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
                    eye_tensor = image_transform(Image.fromarray(eye_rgb))
                    eye_images.append(eye_tensor)
                    cv2.rectangle(frame, rectangle[:2], rectangle[2:], (255, 200, 0), 1)

                    if SHOW_EYE_PREVIEW:
                        preview = ((eye_tensor.squeeze().numpy() * 0.5 + 0.5) * 255).astype("uint8")
                        cv2.imshow(f"Eye {eye_number} - CNN input", cv2.resize(preview, (192, 192)))

            if len(eye_images) == 2:
                with torch.no_grad():
                    predictions = model(torch.stack(eye_images).to(device))
                    probabilities = torch.softmax(predictions, dim=1)
                    sleepy_scores = probabilities[:, sleepy_class_index].cpu().tolist()

                current_sleepy_score = sum(sleepy_scores) / 2
                if smoothed_sleepy_score is None:
                    smoothed_sleepy_score = current_sleepy_score
                else:
                    smoothed_sleepy_score = (
                        SMOOTHING_FACTOR * current_sleepy_score
                        + (1 - SMOOTHING_FACTOR) * smoothed_sleepy_score
                    )

                # Dua threshold agar status tidak berubah-ubah tiap frame.
                if eye_status == "OPEN" and smoothed_sleepy_score >= CLOSED_THRESHOLD:
                    eye_status = "CLOSED"
                    closed_start_time = time.monotonic()
                elif eye_status == "CLOSED" and smoothed_sleepy_score <= OPEN_THRESHOLD:
                    eye_status = "OPEN"
                    closed_start_time = None

                closed_duration = (
                    time.monotonic() - closed_start_time
                    if closed_start_time is not None else 0.0
                )

                cv2.putText(frame, f"Eye 1 sleepy: {sleepy_scores[0]:.2f} | Eye 2: {sleepy_scores[1]:.2f}",
                            (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Smoothed score: {smoothed_sleepy_score:.2f}",
                            (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                displayed_status = eye_status
            else:
                # Jangan lanjut menghitung durasi jika mata tidak terdeteksi.
                eye_status = "OPEN"
                displayed_status = "FACE NOT FOUND"
                smoothed_sleepy_score = None
                closed_start_time = None
                closed_duration = 0.0

            warning = eye_status == "CLOSED" and closed_duration >= ALARM_DELAY

            # loops=-1: setelah file selesai, otomatis memutar ulang dari awal.
            if warning and not alarm_playing:
                pygame.mixer.music.play(loops=-1)
                alarm_playing = True
            elif not warning and alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False

            status_color = (0, 0, 255) if warning else (0, 220, 0)
            cv2.putText(frame, f"Eyes: {displayed_status}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"Closed: {closed_duration:.1f} s", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            if warning:
                cv2.putText(frame, "DROWSINESS WARNING!", (15, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Eye CNN - Custom Alarm", frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord("q"):
                break
            if pressed_key == ord("t") and not warning:
                pygame.mixer.music.play()  # Tes sekali, tanpa looping.
finally:
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    camera.release()
    cv2.destroyAllWindows()
