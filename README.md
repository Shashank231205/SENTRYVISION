
# SENTRYVISION — Ultra Advanced Video Intelligence Suite (PE-VISTA 3.0 Ultra)

SENTRYVISION is an industrial-grade, multimodal **Video Intelligence System** built on top of Meta Perception Encoder (PE-Core & PE-Spatial) + Perception Language Model (PLM).  
It performs **real-time, offline, and batch video understanding**, including segmentation, detection, 2D/3D pose, gaze, depth, mesh, action recognition, timeline extraction, chaptering, highlights, advanced reasoning, and video question-answering.

---

## 🚀 Features

### 🔍 Video Understanding
- Full video summary  
- Object, action, and event detection  
- Scene understanding  
- Timeline and sequence extraction  

### 🧩 Segmentation & Detection
- Per-frame segmentation masks  
- Bounding boxes  
- Optical Flow tracking  
- Motion trajectories JSON export  

### 🏃‍♂️ Pose, 3D Motion, and Angles
- 2D pose estimation  
- 3D lifting (triangulation-free)  
- 3D joint angle extraction  

### 🏏 Sports Coaching Intelligence
Persona-based expertise:
- Cricket batting/bowling  
- Tennis forehand/backhand  
- Badminton smashes/footwork  
- Boxing stance/punch  
- Football movement/striking  

### 🧠 Gaze & Face Analysis
- Face landmarks  
- Gaze estimation  
- Eye contact scoring  

### 🎬 Video Editing / Breakdown
- Automatic highlight generation  
- Chapter segmentation  
- Keyframe extraction  
- Camera motion analysis  

### 📡 Real-Time AI (Webcam / RTSP)
- Supports CCTV feeds  
- Low-latency inference  
- Frame-by-frame segmentation, pose, detection  

### ❓ Video Question Answering (VQA)
Ask:
- “What is the person doing?”  
- “Who is the batsman?”  
- “Is the ball visible?”  
- “Did the player make a mistake?”  

### 🧱 Depth, Point Cloud, Mesh
- Depth estimation from PE-Spatial  
- Point cloud reconstruction  
- Mesh generation  

---

## 🏗️ Tech Stack

| Component | Technology |
|----------|------------|
| Vision Backbone | Meta PE-Core / PE-Spatial |
| Reasoning | PLM (Video-LLM) |
| UI | Gradio Blocks |
| Video Processing | FFmpeg, OpenCV |
| 3D | Pose lifting, depth, mesh |
| Real-Time | Webcam + RTSP engine |

---

## 🔧 Installation

### 1. Clone
```bash
git clone https://github.com/Shashank231205/SENTRYVISION
cd SENTRYVISION

2. Install dependencies
pip install -r requirements.txt

3. Install FFmpeg

Windows:

winget install ffmpeg


Linux:

sudo apt install ffmpeg

4. Run the app
python app.py


The UI will run on:

http://localhost:7860

🧠 API Endpoints
POST /analyze
POST /compare
POST /timeline
POST /trajectories
POST /pose
POST /pose3d
POST /angles
POST /depth
POST /actions
POST /gaze
POST /caption
POST /vqa
POST /summary
POST /highlights
POST /chapters
POST /keyframes
POST /camera_motion
POST /coach
POST /context
POST /stream

👤 Author

Shashank KS
AI • Computer Vision • LLMs • MLOps
IIIT Nagpur 




