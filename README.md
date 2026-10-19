# DrowseGuard — Real-Time Eye Closure Monitoring

DrowseGuard is a webcam-based eye closure monitoring application developed as a Computer Vision project. It uses a Convolutional Neural Network (CNN) to classify eye images as **open (`awake`)** or **closed (`sleepy`)**, then tracks how long the eyes remain closed. If the closure lasts for **2 seconds**, the application displays a warning and triggers a custom audio alarm.

The application runs locally through a **Gradio** web interface, with **OpenCV** for frame processing, **MediaPipe Face Mesh** for locating the eyes, and **PyTorch** for CNN inference.

> **Scope:** DrowseGuard detects prolonged eye closure as a possible indicator of reduced alertness. It does not diagnose drowsiness or guarantee user safety.

## Features

- Live webcam input through the browser, without recording a video first.
- Eye localization and cropping with MediaPipe Face Mesh.
- CNN-based classification of open and closed eyes (no Eye Aspect Ratio calculation).
- Prediction smoothing to reduce rapid changes between states.
- Eye closure duration tracking and a visual warning after 2 seconds.
- Custom MP3 alarm that loops while the warning is active (subject to browser autoplay permissions).
- A Gradio monitoring dashboard showing camera input, eye status, duration, alarm status, and eye previews.

## How It Works

```text
Webcam (browser)
       ↓
Frame processing (OpenCV)
       ↓
Eye landmarks and cropping (MediaPipe Face Mesh)
       ↓
Grayscale → Resize to 64 × 64 → Normalize
       ↓
CNN prediction: awake / sleepy
       ↓
Prediction smoothing + eye closure timer
       ↓
Dashboard warning + looping audio alarm
```

The CNN is trained on **eye images**, not full-face images. During live use, MediaPipe locates each eye in the webcam frame before the cropped images are passed to the CNN.

## Dataset and Model

The training workflow uses the **MRL Eye Dataset**, organized into two classes:

| Class | Meaning |
| --- | --- |
| `awake` | Eye open |
| `sleepy` | Eye closed |

The dataset is split into `train`, `val`, and `test` directories. Images are converted to grayscale, resized to **64 × 64**, converted to tensors, and normalized with mean **0.5** and standard deviation **0.5**.

The CNN consists of three convolutional blocks followed by a classifier. The trained checkpoint is saved as `Source_Code/eye_cnn_best.pth`.

**Important limitation:** MRL contains infrared eye images, while a typical laptop webcam captures visible-light color images. Performance on the dataset may differ from performance during live webcam use. Dataset splitting should also be checked for overlap between subjects or recording sessions.

## Project Structure

```text
B_Group01_OpenCV_Final_Submission/
├── Dataset_or_Sample_Input/
│   ├── train/                 # Not included in GitHub if ignored
│   │   ├── awake/
│   │   └── sleepy/
│   ├── val/
│   │   ├── awake/
│   │   └── sleepy/
│   ├── test/
│   │   ├── awake/
│   │   └── sleepy/
│   ├── get_info.py
│   ├── split_data.py
│   ├── labels.txt
│   └── readme.md
├── Source_Code/
│   ├── main.py
│   ├── style.css
│   ├── train_eye_cnn.ipynb
│   ├── eye_cnn_best.pth       # Trained checkpoint
│   ├── alarm.mp3              # Custom alert audio
│   ├── requirements.txt
│   └── predict_eye_webcam_custom_alarm.py  # Earlier desktop prototype
├── .gitignore
└── README.md
```

The full image dataset is not required to **run inference**; it is used for training and evaluation. If the images are not in this repository, obtain the dataset from its original provider and prepare the class folders shown above.

## Getting Started

**Requirements:** Python, a working webcam, a supported browser, and the dependencies listed in `Source_Code/requirements.txt`. A CUDA-enabled GPU can accelerate CNN inference, but the application can also run on CPU.

### 1. Clone the repository

```bash
git clone https://github.com/fuhrerhn/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Set up Python dependencies

If you already have the `dl-gpu` Conda environment used during development:

```bash
conda activate dl-gpu
python -m pip install -r Source_Code/requirements.txt
```

**GPU note:** When setting up a new environment, install a PyTorch build compatible with your GPU/CUDA setup using the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/). Installing or upgrading packages may otherwise replace a working GPU-enabled PyTorch installation.

### 3. Prepare the required application files

Place these files in `Source_Code/`:

- `eye_cnn_best.pth` — the trained CNN checkpoint.
- `alarm.mp3` — an audio file you have permission to use.
- `style.css` — the dashboard stylesheet.

If the model or audio is not distributed with the public repository, add your own authorized copies before running the application.

### 4. Run the application

From the repository root:

```bash
python Source_Code/main.py
```

Open the local address printed in the terminal (usually **http://127.0.0.1:7860**) and grant webcam permission. Start the webcam stream to see predictions update in real time.

## Using DrowseGuard

1. Start the webcam stream in the Gradio interface.
2. Keep your face visible and well lit.
3. Watch the eye status, closure timer, and cropped-eye preview update.
4. Close both eyes for at least 2 seconds to test the warning and alarm.
5. Reopen your eyes to clear the warning, or use **Reset Monitoring** to reset the displayed monitoring state.

If the warning appears but the alarm is silent, check browser autoplay permissions, the audio player, system volume, and whether `alarm.mp3` is present.

## Known Limitations

- Infrared training images and regular webcam frames differ in appearance.
- Results depend on lighting, camera position, occlusion, and eye-crop quality.
- Live browser streaming may introduce latency compared with direct desktop webcam capture.
- Browser autoplay restrictions can prevent the alarm from playing automatically.
- The system classifies eye closure; it does not assess every cause or sign of drowsiness.

## Technologies

**Python · PyTorch · Torchvision · OpenCV · MediaPipe Face Mesh · Gradio · NumPy · Pillow**

## References

- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Gradio](https://www.gradio.app/)
- MRL Eye Dataset — cite the exact dataset page and version used for training in the final project report.

## Project Team

Developed by **Group 01** for an academic Computer Vision project. Team-member roles and contributions should be documented in the final submission.
