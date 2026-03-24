# 🪄 MagicHands

A real-time augmented reality app that detects open hands via webcam and renders an animated **magic circle** around your palm using MediaPipe and OpenCV.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Demo

Hold your open palm toward the camera and a spinning magic circle will appear around it in real time.

- Rotating dashed outer ring
- Counter-spinning inner ring
- Animated hexagram (6-pointed star)
- Glowing dot markers
- Supports **both hands** simultaneously

---

## 🛠️ Requirements

- Python 3.10+
- Webcam

### Dependencies

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/MagicHands.git
cd MagicHands
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe numpy
```

**4. Download the hand landmark model**
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float32/latest/hand_landmarker.task', 'hand_landmarker.task')"
```

**5. Run the app**
```bash
python magic.py
```

Press **Q** to quit.

---

## 📁 Project Structure

```
MagicHands/
├── magic.py               # Main application
├── hand_landmarker.task   # MediaPipe model file (downloaded separately)
└── README.md
```

---

## ⚙️ How It Works

| Component | Description |
|---|---|
| `HandDetector` | Uses MediaPipe Hand Landmarker to detect and track hands in video frames |
| `_is_open()` | Checks if at least 3 out of 4 fingers are extended to confirm an open palm |
| `MagicCircleRenderer` | Draws layered animated circles, a hexagram, glow effects, and dot markers using OpenCV |

---

## 🎨 Customization

You can easily tweak the visuals inside the `MagicCircleRenderer` class:

```python
# Change colors (BGR format)
PRIMARY   = (255, 0, 200)   # bright magenta
SECONDARY = (180, 0, 255)   # deep purple
ACCENT    = (255, 100, 255) # light pink
GLOW      = (200, 50, 255)  # glow purple
```

You can also adjust rotation speeds in `_update_angles()`:

```python
self.outer_angle = (self.outer_angle + 60 * dt) % 360   # outer ring speed
self.inner_angle = (self.inner_angle - 90 * dt) % 360   # inner ring speed
self.star_angle  = (self.star_angle  + 30 * dt) % 360   # star speed
```

---

## 🔧 Troubleshooting

**`module 'mediapipe' has no attribute 'solutions'`**
You are using MediaPipe 0.10+. This project uses the new Tasks API — make sure you downloaded the `hand_landmarker.task` model file (Step 4 above).

**Hand not detected reliably**
- Make sure your hand is well-lit and clearly visible to the camera
- Keep your palm open and facing the camera
- The model file must be the `float32` version for best accuracy

---

## 📄 License

This project is licensed under the MIT License.
