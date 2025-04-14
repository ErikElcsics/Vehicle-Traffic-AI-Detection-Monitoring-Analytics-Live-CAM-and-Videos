# Vehicle Traffic AI Detection Monitoring Analytics Live CAM and Videos

Intelligent AI Vehicle Detection, Counting, Monitoring &amp; Analysis - allows users to upload pre-recorded videos or stream directly from a live camera feed for real-time traffic analysis.

## 🛣️ About the App

Ever wondered how many vehicles passed through a specific area in a video? Whether you're analyzing traffic flow, monitoring congestion, or just exploring AI-based computer vision — this app has you covered.

Intelligent AI Vehicle Real-Time Vehicle Detection, Counting, Monitoring & Analysis - allows users to upload pre-recorded videos or stream directly from a live camera feed for real-time traffic analysis. 
Whether you're an urban planner, data analyst, traffic engineer, or just a curious developer, this app gives you instant insights into traffic flow, vehicle types, density levels, and more — all in a clean, visual, and customizable dashboard.

Vehicle Traffic Detection, Counting & Monitoring is a powerful and easy-to-use web app that lets you upload any video and automatically detect, track, and count vehicles like cars, buses, motorcycles, and trucks — all using the latest YOLOv8 object detection model.

A powerful computer vision web app for real-time vehicle detection, speed estimation, traffic density analysis, and summary reporting. Built with YOLOv8, Streamlit, and OpenCV, this app allows you to upload a video and monitor traffic flow effortlessly.

Whether you're a traffic analyst, a smart city researcher, or just playing with AI tools — this app makes complex vehicle detection and tracking simple, interactive, and surprisingly fun.


### ✨ Key Features:

#### "Live Camera or Video" selection 

#### 🎥 Use the webcam live!
- Record and save the full detection video.
- Download the video afterward.

#### 🎥 Upload Any Video
Simply drag and drop or browse your computer for a video file (`.mp4`, `.avi`, or `.mov`) into the app and let it do the rest.

#### 🎯 Choose Which Vehicle to Detect - 
You’re in control — pick which vehicle types you want to track from the sidebar: cars, buses, motorcycles, and trucks.

### 🎯 4. Real-time Detection & Tracking
- Live object detection with YOLOv8.
- Bounding boxes drawn on vehicles.
- Counts shown per frame in the top-left for Cars, Buses, Motorcycles and Trucks.
- Threshold Line (Yellow): Clearly visible line added near the bottom of the Region of Interest (ROI).
- Single Vehicle Count Logic: Vehicles are only counted when they cross the threshold.
- No duplicate counts.

#### 🟩 Select a Region of Interest (ROI)
Focus only on the part of the video that matters most. You can draw a box over the area where you want detection to happen — like a specific lane, gate, or intersection.
- Click to select a custom region on the first frame of the video.
- Focus analysis on intersections, lanes, or specific zones.

#### ➖ Add a Threshold Line
Want to know how many vehicles cross a certain point? Add a horizontal threshold line and count vehicles as they cross it. You can even slide it up or down to get the perfect spot.

#### 👁️ Preview Before You Run
Make sure everything looks good before running detection. You can preview your selected ROI and threshold line on the first frame of your video.

#### 🧠 Smart Detection + Tracking
The app uses YOLOv8 under the hood for real-time object detection, and a custom Centroid Tracker to track vehicles across frames and prevent double-counting.

#### 🔢 Live Counting & Density Tracking
As your video processes:
- You'll see a live stream with bounding boxes and class labels.
- Vehicle counts will update in real-time.
- You'll also see a density percentage showing how much of the selected area is covered by vehicles.

### 🚦  Speed Estimation
- Pixel-based motion tracking of vehicles across frames.
- Displayed near bounding boxes.

### 📊  Traffic Density Estimation
- Percentage of the ROI area covered by vehicles.
- Real-time stats.

### 🌡️  Heatmap & Timeline Overlay (Basic)
- Shows frame-wise traffic presence intensity.
- Optional for future upgrade.

#### ✅ Summary Report
Once your video is processed:
- You'll get a total count of all vehicles
- Individual counts for each vehicle type
- All of this is displayed clearly under a "Summary" section.

#### ⬇️ Download Results
- Save the processed video with all visual overlays.
- Download a CSV summary with all vehicle counts neatly organized for further analysis or reporting.


## 💻 Tech Stack

- [Streamlit](https://streamlit.io/)  
  Interactive UI framework used to build the user-friendly dashboard for uploading videos, selecting ROIs, customizing detection settings, and visualizing output in real time.

- [OpenCV](https://opencv.org/)  
  Handles video processing tasks including frame-by-frame analysis, ROI drawing, live camera integration, and video output generation.

- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)  
  State-of-the-art object detection model used to detect and classify vehicle types (cars, buses, trucks, motorcycles) with speed and precision.

- [PyTorch](https://pytorch.org/)  
  Deep learning framework powering the YOLOv8 model, enabling high-performance inference with support for both CPU and GPU execution.

- [NumPy](https://numpy.org/)  
  Used for efficient mathematical operations, including centroid calculations and distance matrices for object tracking.

- [Pandas](https://pandas.pydata.org/)  
  Organizes vehicle detection and tracking data, and enables exporting traffic summary reports to CSV format.

- Built-in Python Libraries:
  - `tempfile` & `os` – For managing uploaded video files and temporary storage.
  - `collections` – Used to manage structured vehicle count data efficiently.
  - `time` – Helps manage frame timing and performance monitoring.


## 📦 Folder Structure


vehicle-traffic-monitor/
├── app.py
├── README.md
├── requirements.txt
└── assets/
    └── (optional images or videos for samples)


## 🛠️ Requirements

text
streamlit
torch
torchvision
ultralytics
opencv-python
numpy
Pandas
PyTorch
YOLOv8


## 🚀 How to Run

### 1. Clone the Repo

git clone https://github.com/your-username/vehicle-traffic-monitor.git
cd vehicle-traffic-monitor


### 2. Install Requirements

#### For CPU:

pip install ultralytics opencv-python-headless torch streamlit numpy

#### For GPU (optional, for faster inference):

GPU - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python-headless streamlit numpy

### 3. Run the App
streamlit run app.py


## 🧪 Usage Guide

1. Upload a Video
   - Use the left sidebar to upload a `.mp4`, `.avi`, or `.mov` file.

2. Choose Vehicle Types
   - Select any combination of car, bus, motorcycle, and truck.

3. Select Region of Interest (ROI)
   - Click the ROI button to crop the region you want analyzed.

4. Run Detection
   - Hit the "Run Detection" button to start analysis.

5. View Results
   - Live annotated video will display:
     - Detected vehicle types
     - Speed
     - Frame-by-frame counts
     - Density bar

6. Download Output
   - Scroll down to get a summary and a download button for the processed video.


## 📚 Libraries & Technologies Used

This app is built using the following tools and libraries:

### 🔧 Technologies
- Python 3.9+
- Streamlit – for building the interactive web interface
- YOLOv8 (Ultralytics) – for real-time object detection
- OpenCV – for image and video processing
- PyTorch – deep learning framework used by YOLO
- NumPy – numerical operations for object tracking and geometry
- Pandas – for summarizing and exporting detection data
- Tempfile / OS – for temporary file handling and I/O operations


## 🚀 How to Use the App (Step-by-Step Guide)

Follow these instructions to fully use all features of the Vehicle Traffic Detection, Counting & Monitoring App:

### 1. Launch the App
Run the Streamlit app from your terminal:
streamlit run app.py

### 2. Upload a Video
- Use the "📹 Upload a video" button in the sidebar.
- Supported formats: `.mp4`, `.avi`, `.mov`.

### 3. Select Vehicle Classes
- Choose which vehicles to detect: car, bus, motorcycle, or truck (default: all selected).

### 4. Enable Threshold Line (Optional)
- Toggle the "➖ Add Threshold Line" option.
- Adjust the Y-axis line position with the slider to track objects crossing a certain line.

### 5. Select ROI (Region of Interest)
- Click "🟩 Select Region of Interest" to draw the area on the first frame where detection will occur.
- Use your mouse to draw a box and press Enter to confirm.

### 6. Preview ROI and Threshold
- Click "👁️ Preview ROI + Threshold" to confirm that the selected area and threshold line look correct.

### 7. Run Detection
- Click "🎯 Run Detection" to start the process.
- The app will:
  - Run YOLOv8 object detection within the selected ROI.
  - Count objects as they cross the threshold line.
  - Track object density in the selected region.
  - Show live video processing with visual indicators.

### 8. View Summary
- Once complete, view:
  - Total vehicles counted
  - Per-vehicle class counts
  - Visual indicators such as detection boxes, centroids, and threshold line

### 9. Download Results
- ✅ Output Video: Click "⬇️ Download Output Video" to save the processed video.
- 📄 CSV Summary: Click "📄 Export Summary to CSV" to download a CSV file containing count totals for all detected classes.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

> Built with ❤️ using Python

![image](https://github.com/user-attachments/assets/14c23a6a-b940-459e-9b61-be32c699fb87)

![image](https://github.com/user-attachments/assets/965b88ad-18f1-40c1-8a30-7eb64fddca92)

![image](https://github.com/user-attachments/assets/7c759a54-0c70-4bb0-ad20-7398febbcdb2)

![image](https://github.com/user-attachments/assets/7d159ef7-9522-4c5a-81e7-ab3a55024aca)

![image](https://github.com/user-attachments/assets/5a871f0b-7631-44ee-a642-93404c5fa1b9)

![image](https://github.com/user-attachments/assets/03ff6991-43ce-4d68-a5dd-bd47cf37241e)

![image](https://github.com/user-attachments/assets/22cbeddd-d6fc-4add-8be4-7c03fa70d6d7)

![image](https://github.com/user-attachments/assets/e37d0626-49a6-41cf-8e96-afc76806b588)







