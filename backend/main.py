
import uuid
import cv2
import csv
import io
import base64
import numpy as np

from pathlib import Path
from threading import Lock
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from video_processor import VideoProcessor


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

video_processor = None
tasks = {}
tasks_lock = Lock()
webcam_lock = Lock()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_processor
    video_processor = VideoProcessor()
    yield
    video_processor = None


app = FastAPI(
    title="AI 3rd Vision Tracker",
    description="Advanced AI object detection and analytics",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",`r`n        "https://frontend-7ajn-r2va1rk66-amithachar7.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "project": "AI 3rd Vision Tracker",
        "status": "running",
        "version": "4.0.0"
    }


@app.get("/api/health")
def health():
    if video_processor is None:
        return {"status": "starting", "device": "unknown"}

    return {
        "status": "healthy",
        "device": video_processor.device,
        "model": "YOLO11s",
        "tracker": "ByteTrack"
    }


@app.get("/api/system")
def system_info():
    if video_processor is None:
        raise HTTPException(503, "Detector is starting.")

    return {
        "device": video_processor.device,
        "model": "YOLO11s",
        "tracker": "ByteTrack",
        "gpu": video_processor.gpu_info()
    }


def update_progress(task_id, values):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(values)


def run_video_task(task_id, input_path, output_path):
    try:
        update_progress(task_id, {
            "status": "processing",
            "progress": 0
        })

        result = video_processor.process_video(
            input_path,
            output_path,
            confidence=0.40,
            image_size=640,
            progress_callback=lambda values: update_progress(
                task_id, values
            )
        )

        report = {
            "frames_processed": result["frames_processed"],
            "total_frames": result["total_frames"],
            "unique_tracking_ids": result["unique_tracking_ids"],
            "object_categories": result["object_categories"],
            "unique_objects_by_category": result["unique_objects_by_category"],
            "device": result["device"],
            "processing_seconds": result["processing_seconds"],
            "download_url": f"/api/download/{task_id}",
            "csv_url": f"/api/export/{task_id}"
        }

        with tasks_lock:
            tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "frames_processed": result["frames_processed"],
                "total_frames": result["total_frames"],
                "fps": round(
                    result["frames_processed"] /
                    max(result["processing_seconds"], 0.001), 2
                ),
                "result": report,
                "records": result["detection_records"]
            })

    except Exception as error:
        update_progress(task_id, {
            "status": "failed",
            "error": str(error)
        })

    finally:
        Path(input_path).unlink(missing_ok=True)


@app.post("/api/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if video_processor is None:
        raise HTTPException(503, "Video processor is not ready.")

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, "Supported formats: MP4, AVI, MOV and MKV."
        )

    task_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{task_id}{extension}"
    output_path = OUTPUT_DIR / f"{task_id}.mp4"

    try:
        with input_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise HTTPException(500, "Could not save uploaded video.")
    finally:
        await file.close()

    with tasks_lock:
        tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "frames_processed": 0,
            "total_frames": 0,
            "fps": 0,
            "result": None,
            "records": [],
            "error": None
        }

    background_tasks.add_task(
        run_video_task, task_id, input_path, output_path
    )

    return {"task_id": task_id, "status": "queued"}


@app.get("/api/process-video/{task_id}")
def get_video_progress(task_id: str):
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Processing task not found.")
        return {
            key: value for key, value in task.items()
            if key != "records"
        }


@app.get("/api/download/{task_id}")
def download_video(task_id: str):
    with tasks_lock:
        task = tasks.get(task_id)

    if task is None or task["status"] != "completed":
        raise HTTPException(404, "Processed video is not ready.")

    output_path = OUTPUT_DIR / f"{task_id}.mp4"

    if not output_path.exists():
        raise HTTPException(404, "Output video not found.")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="AI_3rd_Vision_Tracker_Result.mp4"
    )


@app.get("/api/export/{task_id}")
def export_csv(task_id: str):
    with tasks_lock:
        task = tasks.get(task_id)

        if task is None or task["status"] != "completed":
            raise HTTPException(404, "Report is not ready.")

        records = list(task["records"])

    output = io.StringIO()
    columns = [
        "frame", "class_name", "tracking_id", "confidence",
        "x1", "y1", "x2", "y2"
    ]

    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="vision_report_{task_id[:8]}.csv"'
        }
    )


@app.post("/api/webcam/reset")
def reset_webcam_tracker():
    if video_processor is None:
        raise HTTPException(503, "Detector is not ready.")

    with webcam_lock:
        with video_processor.model_lock:
            video_processor.reset_trails()
            predictor = video_processor.model.predictor

            if predictor is not None and hasattr(predictor, "trackers"):
                for tracker in predictor.trackers:
                    tracker.reset()

    return {"status": "ready"}


@app.post("/api/webcam/detect")
async def webcam_detect(file: UploadFile = File(...)):
    if video_processor is None:
        raise HTTPException(503, "Detector is not ready.")

    image_bytes = await file.read()
    await file.close()

    if not image_bytes:
        raise HTTPException(400, "Empty camera frame.")

    frame = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(400, "Invalid camera frame.")

    try:
        with webcam_lock:
            with video_processor.model_lock:
                results = video_processor.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.40,
                    imgsz=640,
                    device=video_processor.device,
                    verbose=False
                )

                result = results[0]
                annotated = video_processor.draw_detections(frame, result)

                detections = []
                ids_by_class = {}
                categories = Counter()
                tracking_ids = set()

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        name = video_processor.model.names[class_id]
                        confidence = float(box.conf[0].item())

                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0].item())
                            tracking_ids.add(track_id)
                            ids_by_class.setdefault(name, set()).add(track_id)

                        categories[name] += 1

                        detections.append({
                            "tracking_id": track_id,
                            "class_name": name,
                            "confidence": round(confidence, 3),
                            "box": [
                                round(float(value), 2)
                                for value in box.xyxy[0].tolist()
                            ]
                        })

                success, encoded = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )

                if not success:
                    raise ValueError("Could not encode camera frame.")

                image_base64 = base64.b64encode(
                    encoded.tobytes()
                ).decode("utf-8")

                trail_counts = {
                    name: len(ids)
                    for name, ids in ids_by_class.items()
                }

        return {
            "image": image_base64,
            "detections": detections,
            "object_count": len(detections),
            "unique_tracking_ids": len(tracking_ids),
            "categories": dict(categories),
            "unique_objects_by_category": trail_counts,
            "device": video_processor.device
        }

    except Exception as error:
        raise HTTPException(500, f"Webcam detection failed: {error}")
