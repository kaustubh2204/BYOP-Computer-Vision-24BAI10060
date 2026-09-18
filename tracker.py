import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = {}       # object_id -> centroid (x, y)
        self.disappeared = {}   # object_id -> lost frame count
        self.labels = {}        # object_id -> class label
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, label):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.labels[self.next_object_id] = label
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.labels[object_id]

    def update(self, rects_with_labels):
        if len(rects_with_labels) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(rects_with_labels), 2), dtype="int")
        input_labels = []

        for i, (startX, startY, endX, endY, label) in enumerate(rects_with_labels):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
            input_labels.append(label)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_labels[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], input_labels[col])

        return self.objects