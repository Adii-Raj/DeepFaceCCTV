# Survil — CCTV Face Identification System

Real-time face detection and recognition on legacy CCTV streams using YuNet + ArcFace (InsightFace buffalo_l). Runs headless as a background service on Windows or Linux, with a Tkinter setup launcher and a Flask web dashboard for monitoring.

**Now with a professional CLI** — manage everything from the terminal.

---

## What's New: CLI (v1.0.0)

DeepFaceCCTV now includes a full command-line interface powered by Typer:

```bash
# Quick start
deepfacecctv run --camera 0                    # Webcam
deepfacecctv run --camera rtsp://...         # RTSP stream
deepfacecctv dashboard --port 8080            # Web dashboard
deepfacecctv gallery list                     # View enrolled faces
deepfacecctv dataset build ./photos          # Build from images
```

See [CLI Reference](#cli-reference) below for all commands.

---

## Project Structure

```
Survil/
├── deepfacecctv/   NEW: Professional CLI package (Typer + Pydantic)
│   ├── cli.py              CLI commands
│   ├── config.py           Typed configuration
│   └── pipeline_adapter.py Bridge to existing core modules
├── core/           Detection pipeline (detector, recogniser, tracker, gallery, pipeline)
├── builder/        Dataset builder (Tkinter UI + ChromaDB ops + embedding extraction)
├── dashboard/      Flask web dashboard (app.py + index.html)
├── launcher/       Tkinter setup launcher (launcher.py + service.py)
├── data/           Runtime data — detections.db, ChromaDB (auto-created)
│   └── face_db/            ChromaDB embeddings
├── dataset/        NEW: Organized image dataset folder (optional)
├── models/         YuNet + buffalo_l ONNX model files (download separately)
├── config.json     Single config file for all settings
├── pyproject.toml  NEW: Package metadata & CLI entry points
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
git clone https://github.com/yourname/DeepfaceCCTV.git
cd DeepfaceCCTV
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install CLI (NEW)

```bash
pip install -e .
```

This registers the `deepfacecctv` and `cctv` commands in your terminal.

### 3. Download models

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

### 4. GPU support (optional)

If you have an NVIDIA GPU, replace `onnxruntime` with `onnxruntime-gpu`:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

## Setup — First Time

### Option A: Using the CLI (Recommended)

```bash
# 1. Check system
deepfacecctv info

# 2. View configuration
deepfacecctv config show

# 3. Build gallery from organized photos
deepfacecctv dataset build ./my_photos

# Expected folder structure:
# ./my_photos/
#   person_a/
#     photo1.jpg
#     photo2.jpg
#   person_b/
#     photo1.jpg

# 4. Verify gallery
deepfacecctv gallery list
deepfacecctv gallery info
```

### Option B: Using the Tkinter Launcher (Legacy)

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

**Using CLI (NEW):**

```bash
# With RTSP stream
deepfacecctv run --camera rtsp://admin:pass@192.168.1.10:554/stream1 --headless

# With video file
deepfacecctv run --camera testing_vedio.mp4 --headless

# With webcam
deepfacecctv run --camera 0
```

**Using legacy launcher:**

From the launcher, click **Start**. The pipeline runs headless in the background. You can close the launcher — detection keeps running.

Or start directly from terminal:

```bash
python core/pipeline.py --headless
```

---

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `deepfacecctv --version` | Show version |
| `deepfacecctv info` | System and project information |
| `deepfacecctv run` | Start detection pipeline |
| `deepfacecctv dashboard` | Start Flask web dashboard |
| `deepfacecctv status` | Check dashboard health and database status |

### Pipeline Run

```bash
deepfacecctv run [OPTIONS]

Options:
  --camera, -s TEXT     Video source: file path, camera index (0), or RTSP URL
  --config, -c PATH     Path to configuration file [default: config.json]
  --headless            Run without GUI window
```

Examples:
```bash
deepfacecctv run --camera 0                              # Webcam
deepfacecctv run --camera testing_vedio.mp4 --headless   # Video file
deepfacecctv run --camera rtsp://192.168.1.10/stream1   # RTSP stream
deepfacecctv run                                         # Use config.json source
```

### Dashboard

```bash
deepfacecctv dashboard [OPTIONS]

Options:
  --host TEXT     Server bind address [default: 0.0.0.0]
  --port INTEGER  Server port [default: 5002]
```

### Status Check

Check if the dashboard is running and verify database connectivity:

```bash
deepfacecctv status

# Check remote dashboard
deepfacecctv status --host 192.168.1.41 --port 8080


Examples:
```bash
deepfacecctv dashboard              # Default port 5002
deepfacecctv dashboard --port 8080  # Custom port
deepfacecctv dashboard --host 127.0.0.1 --port 8080  # Local only
```

### Configuration

```bash
# Show current config
deepfacecctv config show

# Config file location: config.json
```

### Gallery Management

The gallery stores face embeddings in ChromaDB for recognition.

```bash
# List all enrolled identities
deepfacecctv gallery list

# Show gallery statistics
deepfacecctv gallery info

# Enroll new person from images
deepfacecctv gallery enroll "John Doe" ./photos/john1.jpg ./photos/john2.jpg

# Delete person (with confirmation)
deepfacecctv gallery delete "John Doe"

# Delete without confirmation
deepfacecctv gallery delete "John Doe" --yes

# Backup gallery to timestamped folder
deepfacecctv gallery backup

# Restore from backup
deepfacecctv gallery restore data/face_db_backup_20260626 --yes
```

### Dataset Management

Datasets are organized folders of face images used to build the gallery.

```bash
# Build dataset from organized images and add to gallery
deepfacecctv dataset build ./my_photos --output ./data/dataset

# Update dataset with new images
deepfacecctv dataset update ./new_photos --dataset ./data/dataset

# Show dataset or gallery info
deepfacecctv dataset info

# Check specific dataset folder
deepfacecctv dataset info --dataset ./custom_dataset
```

**Note:** If no image dataset folder exists, `dataset info` automatically shows gallery info instead.

---

## Running the Dashboard

**Using CLI (NEW):**

```bash
deepfacecctv dashboard
```

**Legacy method:**

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
   - Program: `C:\path	oenv\Scripts\python.exe`
   - Arguments: `-m deepfacecctv run --camera rtsp://... --headless`
   - Start in: `C:\path	o\Survil`
4. Settings: check **Run whether user is logged on or not**

Or use the legacy method:
   - Arguments: `core/pipeline.py --headless`

---

## Auto-start on Linux (systemd)

```ini
# /etc/systemd/system/survil.service
[Unit]
Description=Survil Face Detection
After=network.target

[Service]
ExecStart=/path/to/venv/bin/python -m deepfacecctv run --camera rtsp://... --headless
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
| `source` / `rtsp_url` | `null` | Video source: file path, `0` (webcam), or RTSP URL |
| `db_path` | `data/face_db` | ChromaDB storage directory |
| `collection_name` | `face_gallery` | ChromaDB collection name |
| `model_dir` | `models/buffalo_l` | InsightFace model directory |
| `detector_model` | `models/face_detection_yunet_2023mar.onnx` | YuNet model path |
| `detections_db` / `output_db` | `data/detections.db` | Detection log output path |
| `headless` | `true` | Skip cv2.imshow (required on servers) |
| `gallery_refresh_sec` | `60` | How often pipeline reloads gallery from ChromaDB |
| `score_threshold` | `0.25` | Minimum cosine similarity to call a match "known" |
| `nms_threshold` | `0.3` | YuNet NMS threshold |
| `confidence_threshold` | `0.7` | YuNet face confidence threshold |
| `flask_port` | `5002` | Dashboard server port |
| `transport` | `tcp` | RTSP transport protocol (tcp/udp/auto) |

**Note:** The CLI supports both old and new config keys for backward compatibility. Legacy keys like `rtsp_url` and `detections_db` are automatically migrated.

---

## Adding New People (Live — No Restart Needed)

**Using CLI (NEW):**

```bash
# Enroll single images
deepfacecctv gallery enroll "New Person" ./photo1.jpg ./photo2.jpg

# Or add to dataset and rebuild
deepfacecctv dataset update ./new_photos --dataset ./data/dataset
```

**Using Tkinter Launcher (Legacy):**

1. Open launcher → Build Dataset (or run `python builder/buildDataSet.py`)
2. Enter name, select image folder, click Add
3. Pipeline picks up the new person within `gallery_refresh_sec` seconds (default 60s)

No need to stop or restart the detection pipeline.

---

## Workflow Examples

### 1. First Time Setup (CLI)

```bash
# Check system
deepfacecctv info

# View config
deepfacecctv config show

# Build gallery from photos
deepfacecctv dataset build ./my_photos

# Verify
deepfacecctv gallery list
deepfacecctv gallery info

# Run pipeline
deepfacecctv run --camera rtsp://192.168.1.10/stream1 --headless

# Monitor in another terminal
deepfacecctv dashboard
```

### 2. Daily Usage

```bash
# Run pipeline with video
deepfacecctv run --camera testing_vedio.mp4 --headless

# Or with webcam
deepfacecctv run --camera 0

# Check gallery stats
deepfacecctv gallery info

# View dashboard
deepfacecctv dashboard --port 5002
```

### 3. Adding New People

```bash
# Option 1: Enroll single images
deepfacecctv gallery enroll "New Person" ./photo1.jpg ./photo2.jpg

# Option 2: Add to dataset folder and rebuild
cp ./new_photos/* ./dataset/new_person/
deepfacecctv dataset build ./dataset

# Verify
deepfacecctv gallery list
```

### 4. Backup and Maintenance

```bash
# Backup gallery
deepfacecctv gallery backup

# Check stats
deepfacecctv gallery info
deepfacecctv dataset info

# View detections
deepfacecctv dashboard --port 5002
```

---

## Troubleshooting

**"deepfacecctv: command not found"**
- Ensure you ran `pip install -e .`
- Check: `which deepfacecctv` (Linux/Mac) or `where deepfacecctv` (Windows)
- Try: `python -m deepfacecctv --help`

**"builder modules not found"**
- Ensure you installed with `pip install -e .`
- Verify `builder/` folder exists at project root

**"No such command 'dataset'"**
- Reinstall: `pip install -e .`

**Pipeline exits immediately**
- Check `pipeline.log` for the error
- Verify RTSP URL is reachable: `ffplay rtsp://...`
- Confirm model files exist in `models/`
- Check video file exists: `ls testing_vedio.mp4`

**All detections show as "unknown"**
- Check ChromaDB has embeddings: `deepfacecctv gallery list`
- Lower `score_threshold` in config.json (try 0.20)
- Make sure gallery images are clear, well-lit, and front-facing

**Dashboard not loading**
- Confirm `deepfacecctv dashboard` is running
- Check port 5002 is not blocked by firewall
- On Windows: allow Python through Windows Defender Firewall

cv2.imshow crash on server / macOS**
- Set `"headless": true` in config.json or pass `--headless` flag
- This is expected — servers have no display

**Gallery shows empty**
- Build or enroll faces first:
```bash
deepfacecctv dataset build ./your_photos
# or
deepfacecctv gallery enroll "Name" ./photo.jpg
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black deepfacecctv/

# Lint
ruff check deepfacecctv/
```

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
| Config | JSON + Pydantic |
| CLI | Typer + Rich |

---
