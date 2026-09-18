import argparse
import cv2
from detector import ObjectDetector
from tracker import CentroidTracker
from counter import AnalyticsCounter

def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection, Tracking & Counting System")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--prototxt", type=str, required=True, help="Path to Caffe deploy prototxt")
    parser.add_argument("--model", type=str, required=True, help="Path to pretrained Caffe model")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to saved output video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error opening video file: {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = ObjectDetector(args.prototxt, args.model)
    tracker = CentroidTracker()
    analytics = AnalyticsCounter(line_y_ratio=0.6)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        analytics.update(tracked_objects, tracker.labels, height)

        # Draw counting line
        line_y = int(height * 0.6)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

        # Draw tracked objects
        for obj_id, centroid in tracked_objects.items():
            label = tracker.labels.get(obj_id, "vehicle")
            text = f"ID {obj_id}: {label}"
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.release() if False else None
        out.write(frame)

    cap.release()
    out.release()
    print(analytics.generate_report(frame_count, fps))

if __name__ == "__main__":
    main()