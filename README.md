# BYOP-Computer-Vision-24BAI10060


# Vehicle Detection, Tracking and Analytics System

This repo contains my computer vision term project. The goal was to build a complete pipeline that detects vehicles in video footage, tracks individual cars and buses across consecutive frames, and counts them as they pass a spatial trigger line.

I chose to build this using OpenCV's built in DNN module along with MobileNet SSD instead of heavy frameworks like PyTorch or Ultralytics. This kept the install small (under 30MB total) while still demonstrating standard CV primitives like frame preprocessing, centroid tracking, and temporal line crossing logic.

## System Architecture

The project is broken down into three main modules:

1. Detection (detector.py): Loads the MobileNet-SSD Caffe model via OpenCV dnn and extracts bounding boxes for target classes (cars and buses).
2. Tracking (tracker.py): Takes bounding box centers and uses Euclidean distance matching via NumPy to retain consistent object IDs across frames.
3. Counting and Reporting (counter.py): Tracks the y-coordinates of object centroids over time to log line crossing events and generates a final console report.

## Directory Structure

vehicle\_tracker/
|-- main.py
|-- detector.py
|-- tracker.py
|-- counter.py
|-- MobileNetSSD\_deploy.prototxt
|-- MobileNetSSD\_deploy.caffemodel
`-- traffic.mp4

## Setup and Installation

Requirements:

* Python 3.8 or newer
* OpenCV (version 4.x recommended, opencv-python-headless)
* NumPy

Installation steps:

1. Clone or download the repository files into a directory on your machine.
2. Set up a virtual environment (optional):
python -m venv venv
venv\\Scripts\\activate  (Windows)
source venv/bin/activate  (Linux/Mac)
3. Install the required dependencies:
pip install "opencv-python-headless<5.0" numpy

## Weights and Sample Video

If you need to fetch the model files and test video via terminal:

curl -L -o MobileNetSSD\_deploy.prototxt https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD\_deploy.prototxt
curl -L -o MobileNetSSD\_deploy.caffemodel https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD\_deploy.caffemodel
curl -L -o traffic.mp4 "https://vids.pexels.com/video-files/854671/854671-hd\_1920\_1080\_25fps.mp4"

## Running the Script

Execute the pipeline through main.py:

python main.py --input traffic.mp4 --prototxt MobileNetSSD\_deploy.prototxt --model MobileNetSSD\_deploy.caffemodel --output processed\_traffic.mp4

Arguments:

* \--input: Path to the input video.
* \--prototxt: Path to the prototxt file.
* \--model: Path to the binary caffemodel weights.
* \--output: Path for saving the output clip with drawn overlays.

## Sample Console Output

Once the video finishes processing, the counter prints a breakdown:



=============================================

VEHICLE ANALYTICS REPORT

Execution Duration : 26.73 seconds

Frames Processed   : 224 @ \~30.0 FPS

Total Vehicles     : 0

\---

=============================================



## Implementation Notes

* SciPy Dependency: The tracker uses pure NumPy (np.linalg.norm) rather than scipy.spatial.distance. This avoids DLL import failures on systems with strict application policies.
* OpenCV Versioning: OpenCV 5.0 drops native Caffe importers. Stick to 4.x releases (e.g., opencv-python-headless<5.0) to ensure compatibility with readNetFromCaffe.
* Environment Pathing: If running on Windows systems with AppLocker or WDAC active, place the project folder in a root path like C:\\CV\_Project instead of AppData or OneDrive.

## Limitations and Future Work

The current centroid tracking implementation works well for sparse traffic with fixed camera angles. However, fast moving objects or heavy occlusions can cause ID switching. Future revisions could integrate Kalman filtering to predict object trajectories during temporary overlap.
