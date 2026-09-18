import time
from collections import Counter

class AnalyticsCounter:
    def __init__(self, line_y_ratio=0.5):
        self.line_y_ratio = line_y_ratio
        self.previous_positions = {}
        self.crossed_ids = set()
        self.vehicle_counts = Counter()
        self.start_time = time.time()

    def update(self, current_objects, labels, frame_height):
        line_y = int(frame_height * self.line_y_ratio)

        for obj_id, centroid in current_objects.items():
            cy = centroid[1]
            if obj_id in self.previous_positions:
                prev_y = self.previous_positions[obj_id]
                # Detect crossing from top to bottom
                if prev_y < line_y <= cy and obj_id not in self.crossed_ids:
                    self.crossed_ids.add(obj_id)
                    label = labels.get(obj_id, "vehicle")
                    self.vehicle_counts[label] += 1

            self.previous_positions[obj_id] = cy

    def generate_report(self, total_frames, fps):
        elapsed_time = time.time() - self.start_time
        total_vehicles = sum(self.vehicle_counts.values())
        
        report = (
            "\n" + "=" * 45 + "\n"
            "         VEHICLE ANALYTICS REPORT         \n"
            + "=" * 45 + "\n"
            f"Execution Duration : {elapsed_time:.2f} seconds\n"
            f"Frames Processed   : {total_frames} @ ~{fps:.1f} FPS\n"
            f"Total Vehicles     : {total_vehicles}\n"
            + "-" * 45 + "\n"
        )
        for v_type, count in self.vehicle_counts.items():
            report += f"  - {v_type.capitalize():<15}: {count}\n"
        report += "=" * 45 + "\n"
        return report