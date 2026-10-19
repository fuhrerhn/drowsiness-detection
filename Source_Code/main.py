"""DrowseGuard — deteksi mata langsung dari webcam browser menggunakan CNN."""

from pathlib import Path
import time

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


# Lokasi file tidak bergantung pada folder tempat terminal dijalankan.
SOURCE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = SOURCE_DIRECTORY / "eye_cnn_best.pth"
ALARM_PATH = SOURCE_DIRECTORY / "alarm.mp3"
STYLE_PATH = SOURCE_DIRECTORY / "style.css"

ALARM_DELAY = 2.0
SMOOTHING_FACTOR = 0.25
CLOSED_THRESHOLD = 0.60
OPEN_THRESHOLD = 0.40

LEFT_EYE = [362, 263, 386, 374, 385, 380]
RIGHT_EYE = [33, 133, 159, 145, 160, 144]


class EyeCNN(nn.Module):
    """Arsitektur yang sama seperti model saat training."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
if not ALARM_PATH.is_file():
    raise FileNotFoundError(f"Alarm tidak ditemukan: {ALARM_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
class_names = checkpoint["classes"]
if class_names != ["awake", "sleepy"]:
    raise ValueError(f"Kelas pada model harus ['awake', 'sleepy'], bukan {class_names}")

model = EyeCNN().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

image_size = checkpoint["image_size"]
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def crop_eye(frame, landmarks, eye_indices):
    """Ambil bagian mata; ukurannya mengikuti lebar mata agar tetap stabil saat merem."""
    height, width = frame.shape[:2]
    x_values = [landmarks[index].x * width for index in eye_indices]
    y_values = [landmarks[index].y * height for index in eye_indices]
    eye_width = max(x_values) - min(x_values)
    if eye_width < 5:
        return None

    center_x = (min(x_values) + max(x_values)) / 2
    center_y = (min(y_values) + max(y_values)) / 2
    crop_width = eye_width * 1.55
    crop_height = eye_width * 0.90

    left = max(0, int(center_x - crop_width / 2))
    right = min(width, int(center_x + crop_width / 2))
    top = max(0, int(center_y - crop_height / 2))
    bottom = min(height, int(center_y + crop_height / 2))

    return frame[top:bottom, left:right] if left < right and top < bottom else None


def initial_state():
    return {"eye_status": "OPEN", "smoothed_score": None,
            "closed_since": None, "last_frame_at": None, "alarm_active": False}


def status_panel(status="READY", duration=0.0, score=None, alarm=False):
    """Tampilkan hasil CNN pada empat kartu monitoring."""
    if alarm:
        eye_title = "Drowsiness Warning"
        eye_description = "Prolonged eye closure detected."
        eye_color = "danger"
    elif status == "CLOSED":
        eye_title = "Eyes Closed"
        eye_description = "Monitoring eye closure duration."
        eye_color = "warning"
    elif status == "OPEN":
        eye_title = "Eyes Open"
        eye_description = "No prolonged eye closure detected."
        eye_color = "normal"
    elif status == "FACE NOT FOUND":
        eye_title = "Face Not Found"
        eye_description = "Please face the webcam."
        eye_color = "idle"
    else:
        eye_title = "Waiting for Camera"
        eye_description = "Allow camera access to begin."
        eye_color = "idle"

    monitoring_active = status in ("OPEN", "CLOSED")
    system_title = "Monitoring Active" if monitoring_active else "System Ready"
    system_description = (
        "Real-time eye monitoring is running."
        if monitoring_active else "Waiting for webcam input."
    )
    alarm_title = "Alarm Active" if alarm else "Alarm Standby"
    alarm_description = (
        "Please open your eyes."
        if alarm else "Will alert after 2 seconds of eye closure."
    )
    progress = min(max(duration / ALARM_DELAY * 100, 0), 100)
    score_text = "" if score is None else f"<span class='score-detail'>Closed-eye score: {score:.0%}</span>"

    pulse_icon = '<svg viewBox="0 0 24 24"><path d="M2 12h4l3-7 5 14 3-7h5"/></svg>'
    eye_icon = '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.5"/></svg>'
    clock_icon = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>'
    bell_icon = '<svg viewBox="0 0 24 24"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 8-3 8h18s-3-1-3-8M10 20h4"/></svg>'

    return f"""
    <div class="status-stack">
        <div class="metric-card system-card">
            <span class="metric-icon green-icon">{pulse_icon}</span>
            <div class="metric-content">
                <div class="metric-label">System Status</div>
                <strong class="metric-value green-text">{system_title}</strong>
                <p>{system_description}</p>
            </div>
        </div>
        <div class="metric-card eye-card is-{eye_color}">
            <span class="metric-icon blue-icon">{eye_icon}</span>
            <div class="metric-content">
                <div class="metric-label">Eye Status</div>
                <strong class="metric-value eye-text">{eye_title}</strong>
                <p>{eye_description}</p>
                {score_text}
            </div>
        </div>
        <div class="metric-card duration-card">
            <span class="metric-icon gray-icon">{clock_icon}</span>
            <div class="metric-content">
                <div class="metric-label">Eye Closure Duration</div>
                <div class="duration-value">{duration:.1f}<span> / {ALARM_DELAY:.1f} sec</span></div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:{progress:.1f}%"></div>
                </div>
            </div>
        </div>
        <div class="metric-card alarm-card {'is-danger' if alarm else ''}">
            <span class="metric-icon green-icon">{bell_icon}</span>
            <div class="metric-content">
                <div class="metric-label">Alarm Status</div>
                <strong class="metric-value">{alarm_title}</strong>
                <p>{alarm_description}</p>
            </div>
        </div>
    </div>
    """


def analyze_frame(frame, state):
    """Dipanggil terus selama kamera streaming; tidak merekam video."""
    if frame is None:
        return status_panel(), None, initial_state(), None

    state = dict(state) if state else initial_state()
    current_time = time.monotonic()
    alarm_was_active = state["alarm_active"]

    # Mulai hitung ulang jika streaming berhenti/jeda cukup lama.
    if state["last_frame_at"] is not None and current_time - state["last_frame_at"] > 2.5:
        state = initial_state()
    state["last_frame_at"] = current_time

    # Frame Gradio bertipe RGB (bukan BGR).
    frame = np.ascontiguousarray(frame)
    frame = cv2.flip(frame, 1)
    image_height, image_width = frame.shape[:2]
    if image_width > 640:
        frame = cv2.resize(frame, (640, round(image_height * 640 / image_width)))
    face_result = face_mesh.process(np.ascontiguousarray(frame))

    eye_images = []
    if face_result.multi_face_landmarks:
        landmarks = face_result.multi_face_landmarks[0].landmark
        for eye_indices in (LEFT_EYE, RIGHT_EYE):
            eye_image = crop_eye(frame, landmarks, eye_indices)
            if eye_image is None:
                break
            eye_images.append(eye_image)

    if len(eye_images) == 2:
        image_tensors = [image_transform(Image.fromarray(eye)) for eye in eye_images]
        with torch.inference_mode():
            predictions = model(torch.stack(image_tensors).to(device))
            sleepy_scores = torch.softmax(predictions, dim=1)[:, 1]
            sleepy_score = sleepy_scores.mean().item()

        old_score = state["smoothed_score"]
        smoothed_score = sleepy_score if old_score is None else (
            SMOOTHING_FACTOR * sleepy_score + (1 - SMOOTHING_FACTOR) * old_score
        )
        state["smoothed_score"] = smoothed_score

        if state["eye_status"] == "OPEN" and smoothed_score >= CLOSED_THRESHOLD:
            state["eye_status"] = "CLOSED"
            state["closed_since"] = current_time
        elif state["eye_status"] == "CLOSED" and smoothed_score <= OPEN_THRESHOLD:
            state["eye_status"] = "OPEN"
            state["closed_since"] = None

        status = state["eye_status"]
        duration = current_time - state["closed_since"] if state["closed_since"] is not None else 0.0
        preview_images = []
        for eye_image in eye_images:
            resized = cv2.resize(eye_image, (190, 105))
            preview_images.append(resized)
        eye_preview = np.concatenate(preview_images, axis=1)
    else:
        state = initial_state()
        state["last_frame_at"] = current_time
        status, duration, smoothed_score, eye_preview = "FACE NOT FOUND", 0.0, None, None

    alarm_active = status == "CLOSED" and duration >= ALARM_DELAY
    state["alarm_active"] = alarm_active

    # Audio hanya dikirim saat alarm berganti OFF -> ON atau ON -> OFF.
    if alarm_active and not alarm_was_active:
        audio_result = str(ALARM_PATH)
    elif not alarm_active and alarm_was_active:
        audio_result = None
    else:
        audio_result = gr.skip()

    return status_panel(status, duration, smoothed_score, alarm_active), eye_preview, state, audio_result


def reset_monitoring():
    return status_panel(), None, initial_state(), None


# CSS dibaca dari folder Source_Code agar tidak bergantung pada lokasi terminal.
css_text = STYLE_PATH.read_text(encoding="utf-8")

# fill_width=True penting: Gradio tidak lagi membatasi aplikasi ke kolom sempit.
with gr.Blocks(title="DrowseGuard | Eye Monitoring", fill_width=True) as demo:
    gr.HTML("""
      <header class="app-header">
        <div class="brand">
          <span class="brand-logo" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.7"/></svg>
          </span>
          <div>
            <div class="brand-name">Drowse<span>Guard</span></div>
            <div class="brand-description">Real-time eye closure monitoring</div>
          </div>
        </div>
        <div class="header-state">
          <span class="online-dot"></span>
          <div><strong>Local Monitoring</strong><small>Local inference</small></div>
        </div>
      </header>
    """)

    with gr.Row(equal_height=False, elem_id="dashboard-layout"):
        with gr.Column(scale=7, min_width=380, elem_id="left-column"):
            with gr.Column(elem_id="webcam-card"):
                gr.HTML("""
                  <div class="webcam-title">
                    <div class="section-title">Live Webcam</div>
                    <span class="webcam-badge">Webcam input</span>
                  </div>
                """)
                camera = gr.Image(
                    sources=["webcam"], streaming=True, type="numpy",
                    image_mode="RGB", height=500, show_label=False,
                    elem_id="live-webcam",
                )

            with gr.Row(elem_id="info-bar"):
                gr.HTML("""
                  <div class="info-content">
                    <span class="info-symbol">i</span>
                    <div>
                      <strong>Private, local monitoring</strong>
                      <p>Camera frames are processed by Python on this computer.</p>
                    </div>
                  </div>
                """)
                reset_button = gr.Button(
                    "↻  Reset Monitoring", variant="primary", elem_id="reset-button"
                )

        with gr.Column(scale=4, min_width=310, elem_id="right-column"):
            monitoring_status = gr.HTML(
                value=status_panel(), elem_id="monitoring-status"
            )
            with gr.Column(elem_id="eye-preview-card"):
                gr.HTML("""
                  <div class="preview-heading">
                    <strong>Eye Preview</strong>
                    <span>CNN input preview</span>
                  </div>
                """)
                eye_preview = gr.Image(
                    type="numpy", interactive=False, show_label=False,
                    height=120, elem_id="eye-crops",
                )

    # Browser bisa memblokir autoplay sampai pengguna berinteraksi dengan halaman.
    with gr.Accordion(
        "Alarm audio (buka jika autoplay diblokir)",
        open=True, elem_id="audio-settings",
    ):
        alarm_player = gr.Audio(
            label="Custom Alarm", autoplay=True, loop=True,
            interactive=False, buttons=[], elem_id="audio-player",
        )

    monitoring_state = gr.State(initial_state())
    camera.stream(
        analyze_frame,
        inputs=[camera, monitoring_state],
        outputs=[monitoring_status, eye_preview, monitoring_state, alarm_player],
        stream_every=0.2,
        time_limit=3600,
        concurrency_limit=1,
        show_progress="hidden",
    )
    camera.clear(
        reset_monitoring,
        outputs=[monitoring_status, eye_preview, monitoring_state, alarm_player],
    )
    reset_button.click(
        reset_monitoring,
        outputs=[monitoring_status, eye_preview, monitoring_state, alarm_player],
    )


if __name__ == "__main__":
    print(f"Model CNN: {MODEL_PATH}")
    print(f"Alarm: {ALARM_PATH}")
    print(f"Device: {device}")
    demo.launch(
        server_name="127.0.0.1",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css_text,
        footer_links=[],
        share=False,
        allowed_paths=[str(ALARM_PATH)],
    )
