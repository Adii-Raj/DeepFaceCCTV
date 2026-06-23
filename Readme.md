# Survil — CCTV Face Identification System

Real-time face detection and recognition on legacy CCTV streams using YuNet + ArcFace (InsightFace buffalo_l). Runs headless as a background service on Windows or Linux, with a Tkinter setup launcher and a Flask web dashboard for monitoring.

---

## Project Structure

```
Survil/
├── core/           Detection pipeline (detector, recogniser, tracker, gallery, pipeline)
├── builder/        Dataset builder (Tkinter UI + ChromaDB ops + embedding extraction)
├── dashboard/      Flask web dashboard (app.py + index.html)
├── launcher/       Tkinter setup launcher (launcher.py + service.py)
├── data/           Runtime data — detections.db, ChromaDB (auto-created)
├── models/         YuNet + buffalo_l ONNX model files (download separately)
├── config.json     Single config file for all settings
└── requirements.txt
```

---

## Requirements

- Python 3.10 or 3.11
- Windows 10/11 or Linux (Ubuntu 20.04+)
- CCTV camera accessible via RTSP URL on the same LAN

---

## Installation

### 1. Clone and set up

```bash
git clone https://github.com/yourname/survil.git
cd Survil
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download models

Create the `models/` folder and download:

**YuNet** (face detector):
```
models/face_detection_yunet_2023mar.onnx
```
Download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

**buffalo_l** (ArcFace recogniser):
```
models/buffalo_l/w600k_r50.onnx
```
InsightFace downloads this automatically on first run. Or manually from:
https://github.com/deepinsight/insightface/tree/master/model_zoo

### 3. GPU support (optional)

If you have an NVIDIA GPU, replace `onnxruntime` with `onnxruntime-gpu`:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

## Setup — First Time

### Step 1: Build the face gallery

Open the launcher and use the Build Dataset window to add people:

```bash
python launcher/main.py
```

Click **Build Dataset**, enter a person's name, select a folder of their photos (5–20 clear face images recommended), click **Add to Gallery**. Repeat for each person.

Or run the builder standalone:

```bash
python builder/buildDataSet.py
```

### Step 2: Configure RTSP URL

Edit `config.json` and set your camera's RTSP stream URL:

```json
{
  "rtsp_url": "rtsp://admin:password@192.168.1.10:554/stream1"
}
```

Common RTSP formats:
- Dahua: `rtsp://admin:pass@ip/cam/realmonitor?channel=1&subtype=0`
- Hikvision: `rtsp://admin:pass@ip:554/h264/ch1/main/av_stream`
- Generic: `rtsp://ip:554/stream1`

### Step 3: Start detection

From the launcher, click **Start**. The pipeline runs headless in the background. You can close the launcher — detection keeps running.

Or start directly from terminal:

```bash
python core/pipeline.py --headless
```

---

## Running the Dashboard

Start the Flask dashboard (separately from the pipeline):

```bash
python dashboard/app.py
```

Open in any browser on the same LAN:
```
http://localhost:5002         ← on the same PC
http://192.168.1.x:5002        ← from any device on LAN
```

The dashboard auto-refreshes every 5 seconds and shows:
- Known / unknown counts
- Top detected people
- Pipeline log tail
- Gallery tab with all registered people

---

## Auto-start on Windows Boot (Task Scheduler)

To have detection start automatically when the PC boots:

1. Open **Task Scheduler** → Create Basic Task
2. Trigger: **At system startup**
3. Action: **Start a program**
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `core/pipeline.py --headless`
   - Start in: `C:\path\to\Survil`
4. Settings: check **Run whether user is logged on or not**

---

## Auto-start on Linux (systemd)

```ini
# /etc/systemd/system/survil.service
[Unit]
Description=Survil Face Detection
After=network.target

[Service]
ExecStart=/path/to/venv/bin/python /path/to/Survil/core/pipeline.py --headless
WorkingDirectory=/path/to/Survil
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable survil
sudo systemctl start survil
journalctl -u survil -f    # live logs
```

---

## config.json Reference

| Key | Default | Description |
|-----|---------|-------------|
| `rtsp_url` | `rtsp://192.168.1.10/stream1` | CCTV RTSP stream URL |
| `db_path` | `data/face_db` | ChromaDB storage directory |
| `collection_name` | `face_gallery` | ChromaDB collection name |
| `model_dir` | `models/buffalo_l` | InsightFace model directory |
| `detector_model` | `models/face_detection_yunet_2023mar.onnx` | YuNet model path |
| `detections_db` | `data/detections.db` | Detection log output path |
| `headless` | `true` | Skip cv2.imshow (required on servers) |
| `gallery_refresh_sec` | `60` | How often pipeline reloads gallery from ChromaDB |
| `score_threshold` | `0.25` | Minimum cosine similarity to call a match "known" |
| `nms_threshold` | `0.3` | YuNet NMS threshold |
| `confidence_threshold` | `0.7` | YuNet face confidence threshold |
| `flask_port` | `5002` | Dashboard server port |

---

## Adding New People (Live — No Restart Needed)

1. Open launcher → Build Dataset (or run `python builder/buildDataSet.py`)
2. Enter name, select image folder, click Add
3. Pipeline picks up the new person within `gallery_refresh_sec` seconds (default 60s)

No need to stop or restart the detection pipeline.

---

## Troubleshooting

**Pipeline exits immediately**
- Check `pipeline.log` for the error
- Verify RTSP URL is reachable: `ffplay rtsp://...`
- Confirm model files exist in `models/`

**All detections show as "unknown"**
- Check ChromaDB has embeddings: open Build Dataset, verify people are listed
- Lower `score_threshold` in config.json (try 0.20)
- Make sure gallery images are clear, well-lit, and front-facing

**Dashboard not loading**
- Confirm `python dashboard/app.py` is running
- Check port 5002 is not blocked by firewall
- On Windows: allow Python through Windows Defender Firewall

**cv2.imshow crash on server / macOS**
- Set `"headless": true` in config.json or pass `--headless` flag
- This is expected — servers have no display

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Face detection | YuNet (OpenCV Zoo) |
| Face recognition | ArcFace buffalo_l (InsightFace) |
| Vector database | ChromaDB |
| Tracking | IOU-based tracker |
| Pipeline | Python threading (capture + process workers) |
| Setup UI | Tkinter |
| Dashboard | Flask + vanilla JS |
| Config | JSON |