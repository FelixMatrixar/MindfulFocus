# Medical Pipeline Evaluator (Local-First, Privacy-Preserved)

A local-first **medical facial analysis pipeline evaluator** built with Google's ADK-style agent pattern plus MediaPipe/OpenCV. It processes live camera feeds or image folders, aggregates metrics, and generates reports **without uploading your data to any server**. Storage uses local SQLite and flat files on your machine for maximum privacy. 【10†MindfulFocus/adk_medical_evaluation】

> **Not a medical device.** This system is for research only and does **not** provide medical advice. Always consult a qualified professional. 【10†MindfulFocus/adk_medical_evaluation】

---

## Why Local-First (No Hosting)
- **No cloud hosting required** — runs from the CLI as a local app. 【10†MindfulFocus/adk_medical_evaluation】  
- **Local persistence** via `SQLiteStorage` and `LocalFileStorage` under `evaluation_data/` and `local_database/`. Your frames, metrics, and reports stay on disk on your machine. 【10†MindfulFocus/adk_medical_evaluation】  
- **ADK sessions kept in-memory** during runtime; only results/metadata are written to local storage for traceability. 【10†MindfulFocus/adk_medical_evaluation】

---

## Features
- Live camera capture and batch processing (simulated/real hooks) with MediaPipe Face Mesh and OpenCV. 【10†MindfulFocus/adk_medical_evaluation】  
- Metrics aggregation (EAR difference, symmetry, severity), quality flags, and performance stats. 【10†MindfulFocus/adk_medical_evaluation】  
- Exports JSON/CSV/TXT reports and a consolidated **final evaluation report** to local folders. 【10†MindfulFocus/adk_medical_evaluation】  
- Interactive CLI menu for common tasks (camera eval, image folder eval, quick test, session summary, DB stats). 【10†MindfulFocus/adk_medical_evaluation】

---

## Project Structure
```
adk_medical_evaluation/
├─ main_evaluator.py                  # CLI entrypoint / orchestrator
├─ agents/
│  ├─ __init__.py
│  └─ medical_evaluator_agent.py      # Agent + local tools (simulated) 【10†MindfulFocus/adk_medical_evaluation】
├─ config/
│  └─ agent_config.py                 # Model/API/storage config (local SQLite paths) 【10†MindfulFocus/adk_medical_evaluation】
└─ tools/
   ├─ __init__.py
   └─ data_processor.py               # ADK-style ToolContext tools (camera, batch, export, aggregate) 【10†MindfulFocus/adk_medical_evaluation】
local_database/
└─ medical_evaluation.db              # SQLite database (created at runtime) 【10†MindfulFocus/adk_medical_evaluation】

evaluation_data/                      # All outputs stored locally
├─ images/ ...                        # Captured frames
├─ reports/ ...                       # Aggregated results & final reports
├─ metrics/ ...                       # Metrics artifacts
├─ sessions/ ...                      # Session state snapshots
└─ logs/ ...                          # Logs
```
> Paths are configured in `STORAGE_CONFIG` inside `config/agent_config.py`. 【10†MindfulFocus/adk_medical_evaluation】

---

## Requirements

- Python 3.10+ (3.11 recommended)
- Packages: `opencv-python`, `mediapipe`, `numpy`, and Google ADK-related packages referenced by the code. 【10†MindfulFocus/adk_medical_evaluation】

### Quick setup (pip)
```bash
# (optional) create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip

# Core deps (add others from your environment as needed)
pip install opencv-python mediapipe numpy
# If you use the ADK codepaths, also install the Google ADK/GenAI libs:
# pip install google-genai google-adk
```

> If you're using Conda, create your env and `pip install` the same packages inside it.

---

## Configuration

The evaluator needs a Gemini API key only to run the *agent-style* flow (no hosting). On first run it will prompt you if it isn’t set.

- Environment variable: `GOOGLE_API_KEY` (the CLI also prompts for it). 【10†MindfulFocus/adk_medical_evaluation】  
- Local storage paths (SQLite DB and folders) are set in `config/agent_config.py` → `STORAGE_CONFIG`. 【10†MindfulFocus/adk_medical_evaluation】

Example `STORAGE_CONFIG` values:
```python
STORAGE_CONFIG = {
    "database_path": "local_database/medical_evaluation.db",
    "images_dir": "evaluation_data/images",
    "reports_dir": "evaluation_data/reports",
    "metrics_dir": "evaluation_data/metrics",
    "sessions_dir": "evaluation_data/sessions",
    "logs_dir": "evaluation_data/logs"
}
```

---

## Running

From the repository root:

```bash
python -m adk_medical_evaluation.main_evaluator
# or
python adk_medical_evaluation/main_evaluator.py
```

On start, you’ll see the **Interactive Mode** menu with options like:
1. Camera Evaluation (5 minutes)  
2. Camera Evaluation (Custom duration)  
3. Image Directory Evaluation  
4. Quick Performance Test (1 minute)  
5. Session Summary  
6. Cleanup Session Files  
7. View Database Statistics  
8. Exit  
【10†MindfulFocus/adk_medical_evaluation】

> All outputs are written under `evaluation_data/` and `local_database/` on your machine. No hosting, no uploads. 【10†MindfulFocus/adk_medical_evaluation】

---

## Typical Workflow

1. **Initialize session** (automatically on start). Sets `GOOGLE_API_KEY`, creates a local ADK session, and opens a local DB run. 【10†MindfulFocus/adk_medical_evaluation】  
2. **Capture** frames from webcam (`process_camera_stream`) or load an image folder. 【10†MindfulFocus/adk_medical_evaluation】  
3. **Batch process** frames (`batch_process_frames`). 【10†MindfulFocus/adk_medical_evaluation】  
4. **Compute metrics** and **generate medical assessment** (EAR diff, symmetry, severity). 【10†MindfulFocus/adk_medical_evaluation】  
5. **Aggregate & export** results, then **save final report**. 【10†MindfulFocus/adk_medical_evaluation】

---

## Privacy Notes (Important)
- **Local-only storage**: frames, metrics, reports are saved to your disk; the DB is a local SQLite file. 【10†MindfulFocus/adk_medical_evaluation】  
- **No hosting**: there’s no server to deploy; the CLI runs offline except for model calls. If you want a fully offline mode, stub or mock model calls—data handling remains local. 【10†MindfulFocus/adk_medical_evaluation】  
- You control retention: use the cleanup/management helpers (`manage_session_files`, `cleanup_old_files`) to prune artifacts. 【10†MindfulFocus/adk_medical_evaluation】

---

## Troubleshooting

- **Tool context mismatches**: There are two tool layers in the codebase:
  - ADK-style tools under `tools/data_processor.py` expect a `ToolContext`.  
  - The agent shim in `agents/medical_evaluator_agent.py` includes simplified tools without an explicit `tool_context`.  
  Make sure you run via `main_evaluator.py`, which wires the agent and context for you. 【10†MindfulFocus/adk_medical_evaluation】

- **Camera not opening**: check `camera_index` in stream config and that another app isn’t using the webcam. 【10†MindfulFocus/adk_medical_evaluation】

- **Performance**: reduce `fps_target` or batch size to lower CPU load. 【10†MindfulFocus/adk_medical_evaluation】

---

## License & Disclaimer

This repository is provided for research/educational purposes. It is **not** a medical device and **not** a substitute for professional medical advice. 【10†MindfulFocus/adk_medical_evaluation】
