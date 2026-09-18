statement_md = """# Problem Statement, Scope, Target Users, and High-Level Features

## 1. Problem Statement
Manual traffic monitoring and physical vehicle counting in urban intersections or highway camera feeds are labor-intensive, error-prone, and unsustainable for continuous analytics. Standard video surveillance systems record raw video footage but fail to offer automated real-time analytics, such as per-category vehicle counts or line-crossing tracking events. Furthermore, full-scale deep learning pipelines built on frameworks like PyTorch or TensorFlow are computationally heavy, require high-end GPU hardware, and struggle to run smoothly on budget edge devices or local development machines.

## 2. Scope of the Project
The scope of this project includes:
* Developing a lightweight, modular Python application using OpenCV's DNN module and MobileNet-SSD Caffe model to perform real-time detection of cars and buses.
* Implementing a framework-independent spatial centroid tracking algorithm built purely on NumPy matrix distance calculations.
* Setting up a configurable virtual trigger line across video frames to record spatial crossing events and tally vehicle counts per category.
* Generating a console-based analytical report summarizing total processing time, frame rates, and vehicle breakdowns upon video completion.
* Exporting an annotated MP4 video output rendering bounding boxes, object IDs, class labels, and trigger line overlays.

Out of Scope:
* High-density multi-lane tracking requiring advanced occlusion handling like Kalman filtering or DeepSORT.
* Speed estimation or license plate recognition (ALPR).
* Real-time streaming database integrations or web dashboards.

## 3. Target Users
* Traffic Management Authorities and Urban Planners who require automated counts for road usage and traffic density analysis.
* Civil Engineers and Researchers collecting localized empirical data on vehicle movement without investing in expensive proprietary hardware.
* Academic Instructors and Students seeking a practical, CPU-friendly reference project demonstrating core Computer Vision concepts using OpenCV and NumPy.

## 4. High-Level Features
* Multi-Class Object Detection: Real-time identification of cars and buses using lightweight Caffe MobileNet-SSD weights.
* Centroid-Based Spatial Tracking: Robust ID persistence and distance matching across frame transitions using pure NumPy matrix math.
* Spatial Line-Crossing Analytics: Automatic registration of downward vehicle movement across a customizable virtual line ratio.
* Dynamic Visual Overlays: Live rendering of bounding boxes, object IDs, vehicle labels, and trigger lines directly on output video frames.
* Analytical Summary Generator: Console report output providing execution timing, total frames, average FPS, and classified vehicle tallies.
"""