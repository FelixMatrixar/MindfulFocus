import os, pathlib, urllib.request

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

def ensure_face_landmarker(model_dir: str | os.PathLike) -> str:
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dst = model_dir / "face_landmarker.task"
    if not dst.exists():
        print("Downloading Face Landmarker model…")
        urllib.request.urlretrieve(MODEL_URL, dst.as_posix())
        print(f"Saved: {dst}")
    return dst.as_posix()
