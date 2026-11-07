import cv2
import torch
import time
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict, deque
from threading import Lock
from ultralytics import YOLO


class VideoProcessor:
    """
    YOLO object detection and ByteTrack processing for uploaded videos
    and live webcam frames.

    The same draw_detections() method is used by both modes, so both
    receive the same smoothed blue movement trails.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading YOLO11s...")
        self.model = YOLO("yolo11s.pt")
        self.model_lock = Lock()

        self.upload_dir = Path(__file__).resolve().parent / "uploads"
        self.output_dir = Path(__file__).resolve().parent / "outputs"

        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.confidence = 0.40
        self.image_size = 640

        # Default trail storage is used by the live webcam.
        self.trails = defaultdict(lambda: deque(maxlen=35))

        # Each trail store (webcam or an individual video) gets its own
        # frame counter and last-seen timestamps.
        self._trail_states = {}

        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("YOLO11s ready!")

    def reset_trails(self):
        """Reset live webcam trail history."""
        self.trails.clear()
        self._trail_states.pop(id(self.trails), None)

    def _smooth_points(self, points):
        """
        Apply a light moving-average filter to reduce small tracking jitter.
        Keep the first and last points anchored to the actual detections.
        """
        raw = np.asarray(points, dtype=np.float32)

        if len(raw) < 3:
            return np.round(raw).astype(np.int32)

        smoothed = raw.copy()

        for i in range(1, len(raw) - 1):
            smoothed[i] = (
                raw[i - 1] * 0.25
                + raw[i] * 0.50
                + raw[i + 1] * 0.25
            )

        return np.round(smoothed).astype(np.int32)

    def draw_detections(self, frame, result, trails=None):
        """
        Draw smooth blue movement trails, yellow boxes, class labels,
        confidence scores and ByteTrack IDs.

        Pass a separate trails dictionary for uploaded-video processing.
        When trails is None, the persistent webcam trail store is used.
        """
        annotated = frame.copy()
        trail_store = trails if trails is not None else self.trails
        store_key = id(trail_store)

        state = self._trail_states.setdefault(
            store_key,
            {"frame": 0, "last_seen": {}}
        )
        state["frame"] += 1
        current_frame = state["frame"]
        last_seen = state["last_seen"]

        # Add the latest center point for each tracked object.
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0].item())
                x1, y1, x2, y2 = [
                    int(value) for value in box.xyxy[0].tolist()
                ]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if track_id not in trail_store:
                    trail_store[track_id] = deque(maxlen=35)

                trail_store[track_id].append(center)
                last_seen[track_id] = current_frame

        # Remove trails that have not been seen for a short period.
        stale_ids = [
            track_id
            for track_id in list(trail_store.keys())
            if current_frame - last_seen.get(track_id, 0) > 12
        ]

        for track_id in stale_ids:
            trail_store.pop(track_id, None)
            last_seen.pop(track_id, None)

        # Draw trails on a separate layer, then blend with the frame.
        trail_layer = annotated.copy()

        for track_id, points in trail_store.items():
            age = current_frame - last_seen.get(track_id, 0)
            if age > 8 or len(points) < 2:
                continue

            smooth = self._smooth_points(list(points))
            polyline = smooth.reshape((-1, 1, 2))

            # Draw older sections first, then brighter recent sections.
            for i in range(1, len(smooth)):
                fade = i / max(len(smooth) - 1, 1)
                thickness = max(1, int(1 + 2 * fade))

                # OpenCV uses BGR: this produces a vivid blue/cyan trail.
                color = (
                    255,
                    int(90 + 130 * fade),
                    int(25 + 45 * fade)
                )

                cv2.line(
                    trail_layer,
                    tuple(smooth[i - 1]),
                    tuple(smooth[i]),
                    color,
                    thickness,
                    cv2.LINE_AA
                )

            # Rounded, bright endpoint.
            endpoint = tuple(smooth[-1])
            cv2.circle(
                trail_layer,
                endpoint,
                4,
                (255, 220, 70),
                -1,
                cv2.LINE_AA
            )

        annotated = cv2.addWeighted(
            trail_layer, 0.78,
            annotated, 0.22,
            0
        )

        # Draw detection boxes and labels above the trails.
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [
                    int(value) for value in box.xyxy[0].tolist()
                ]

                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0].item())

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                label = f"{class_name} {confidence:.2f}"
                if track_id is not None:
                    label = f"ID {track_id} | {label}"

                box_color = (0, 220, 255)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                    cv2.LINE_AA
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.55
                thickness = 2

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, scale, thickness
                )

                label_y = max(y1, text_height + 10)

                cv2.rectangle(
                    annotated,
                    (x1, label_y - text_height - 8),
                    (x1 + text_width + 8, label_y + baseline),
                    box_color,
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, label_y - 4),
                    font,
                    scale,
                    (10, 20, 30),
                    thickness,
                    cv2.LINE_AA
                )

        return annotated

    def process_video(
        self,
        input_path,
        output_path,
        confidence=None,
        image_size=None,
        progress_callback=None
    ):
        started_at = time.time()

        confidence = (
            confidence if confidence is not None else self.confidence
        )
        image_size = (
            image_size if image_size is not None else self.image_size
        )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Could not open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Invalid video dimensions.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise ValueError("Could not create output video.")

        frame_count = 0
        object_counts = Counter()
        unique_objects = set()
        detection_records = []

        # Separate trail store for this uploaded video.
        video_trails = defaultdict(lambda: deque(maxlen=35))
        self._trail_states.pop(id(video_trails), None)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                with self.model_lock:
                    results = self.model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        imgsz=image_size,
                        device=self.device,
                        verbose=False
                    )

                    result = results[0]
                    annotated = self.draw_detections(
                        frame, result, video_trails
                    )

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0].item())
                            class_name = self.model.names[class_id]
                            score = float(box.conf[0].item())

                            object_counts[class_name] += 1

                            track_id = None
                            if box.id is not None:
                                track_id = int(box.id[0].item())
                                unique_objects.add((class_name, track_id))

                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            detection_records.append({
                                "frame": frame_count + 1,
                                "class_name": class_name,
                                "tracking_id": track_id,
                                "confidence": round(score, 4),
                                "x1": round(x1, 2),
                                "y1": round(y1, 2),
                                "x2": round(x2, 2),
                                "y2": round(y2, 2)
                            })

                writer.write(annotated)
                frame_count += 1

                if progress_callback and (
                    frame_count % 5 == 0 or frame_count == total_frames
                ):
                    elapsed = max(time.time() - started_at, 0.001)
                    progress_callback({
                        "progress": min(
                            int(frame_count / total_frames * 100), 99
                        ) if total_frames else 0,
                        "frames_processed": frame_count,
                        "total_frames": total_frames,
                        "fps": round(frame_count / elapsed, 2)
                    })

        finally:
            cap.release()
            writer.release()
            self._trail_states.pop(id(video_trails), None)

        processing_seconds = round(time.time() - started_at, 2)

        if frame_count == 0:
            raise ValueError("No video frames were processed.")

        return {
            "frames_processed": frame_count,
            "total_frames": total_frames,
            "unique_tracking_ids": len(unique_objects),
            "object_categories": dict(object_counts),
            "unique_objects_by_category": dict(
                Counter(name for name, _ in unique_objects)
            ),
            "detection_records": detection_records,
            "device": self.device,
            "processing_seconds": processing_seconds,
            "output_path": str(output_path)
        }

    def gpu_info(self):
        if not torch.cuda.is_available():
            return {
                "available": False,
                "name": "CPU",
                "memory_used_mb": 0,
                "memory_reserved_mb": 0,
                "memory_total_mb": 0
            }

        properties = torch.cuda.get_device_properties(0)

        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_used_mb": round(
                torch.cuda.memory_allocated(0) / 1024**2, 1
            ),
            "memory_reserved_mb": round(
                torch.cuda.memory_reserved(0) / 1024**2, 1
            ),
            "memory_total_mb": round(
                properties.total_memory / 1024**2, 1
            )
        }
