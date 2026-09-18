import cv2
import numpy as np

DEFAULT_TARGET_CLASSES = {6: "bus", 7: "car"}

class ObjectDetector:
    def __init__(self, prototxt_path: str, model_path: str, target_classes=None, confidence_threshold=0.4):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.target_classes = target_classes or DEFAULT_TARGET_CLASSES
        self.conf_threshold = confidence_threshold

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])
                if idx in self.target_classes:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    startX, startY, endX, endY = box.astype("int")
                    boxes.append((startX, startY, endX, endY, self.target_classes[idx]))
        return boxes