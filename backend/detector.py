
import torch
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading YOLO11n on: {self.device}")

        self.model = YOLO(model_path)

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def detect(self, frame, confidence=0.30, image_size=640):
        results = self.model.predict(
            source=frame,
            conf=confidence,
            imgsz=image_size,
            device=self.device,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                confidence_score = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence_score, 3),
                    "box": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                })

        return annotated_frame, detections
