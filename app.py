import streamlit as st
import cv2
import tempfile
import torch
import os
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict
import pandas as pd

st.sidebar.header("🚦 Detection & Analysis Setup: Control Your Traffic Tracker")
# st.sidebar.subheader("🔍 Customize Your Traffic Tracker")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**💻 Running on:** `{device.upper()}`")

# Load model
model = YOLO("yolov8n.pt")
model.to(device)

# Vehicle classes & colors
vehicle_classes = ['car', 'bus', 'motorcycle', 'truck']
class_colors = {
    'car': (0, 255, 0),
    'bus': (255, 0, 0),
    'motorcycle': (0, 0, 255),
    'truck': (255, 255, 0)
}

# UI Elements
st.title("🚗 UrbanEye Traffic Monitor")
st.subheader("Vehicle Traffic Vision AI Detection, Counting, Monitoring & Analytics")

st.sidebar.header("🔧 Options")

# Select input source
input_source = st.sidebar.radio("📷 Select Input Source", ("Video File", "Live Camera"))

# Common controls
selected_classes = st.sidebar.multiselect("🚙 Select Vehicles to Detect", vehicle_classes, default=vehicle_classes)
use_threshold = st.sidebar.checkbox("➖ Add Threshold Line")
threshold_slider = st.sidebar.slider("📏 Adjust Threshold Line (Y-axis)", 0, 1000, 300)

# Session state setup
if "roi" not in st.session_state:
    st.session_state.roi = None
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# ROI function for videos
def select_roi_on_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    roi = cv2.selectROI("Select ROI and press ENTER", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

# Centroid Tracker
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.counted_ids = set()

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > 10:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > 10:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# ================================
# 🎥 LIVE CAMERA HANDLING
# ================================
def run_live_camera():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access the camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height
    fps = 20  # You can adjust this if needed

    threshold_y = threshold_slider
    total_counts = defaultdict(int)
    ct = CentroidTracker()

    # Create output video writer
    output_filename = f"live_output_{int(time.time())}.mp4"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.info("📡 Live camera started. Press **Stop** to end session.")
    stop_button = st.button("🛑 Stop")

    while cap.isOpened():
        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Camera frame not received.")
            break

        results = model(frame, verbose=False, device=0 if device == "cuda" else "cpu")[0]

        centers = []
        center_to_class = {}
        vehicle_pixels = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name in selected_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                centers.append(center)
                center_to_class[center] = cls_name

                color = class_colors.get(cls_name, (0, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                vehicle_pixels += (x2 - x1) * (y2 - y1)

        objects = ct.update(np.array(centers))

        if use_threshold:
            cv2.line(frame, (0, threshold_y), (frame.shape[1], threshold_y), (0, 255, 255), 2)

        for object_id, centroid in objects.items():
            cx, cy = centroid
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

            if object_id not in ct.counted_ids and cy > threshold_y:
                matched_cls = None
                for center, cls_name in center_to_class.items():
                    if np.linalg.norm(np.array(center) - np.array(centroid)) < 20:
                        matched_cls = cls_name
                        break

                if matched_cls:
                    total_counts[matched_cls] += 1
                total_counts["vehicles"] += 1
                ct.counted_ids.add(object_id)

        density_ratio = vehicle_pixels / frame_area
        cv2.putText(frame, f"Density: {density_ratio:.2%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Vehicles Counted: {total_counts['vehicles']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        y_offset = 120
        for cls in vehicle_classes:
            count = total_counts.get(cls, 0)
            label = f"{cls.capitalize()}: {count}"
            cv2.putText(frame, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[cls], 2)
            y_offset += 30

        stframe.image(frame, channels="BGR", use_container_width=True)

        # Save frame to output video
        out.write(frame)

    cap.release()
    out.release()
    st.success("✅ Live camera session ended and saved.")

    # Allow user to download video
    with open(output_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Live Session Video",
            data=file,
            file_name=output_filename,
            mime="video/mp4"
        )

# ================================
# VIDEO FILE HANDLING
# ================================
if input_source == "Video File":
    uploaded_video = st.sidebar.file_uploader("📹 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.sidebar.button("🟩 Select Region of Interest"):
            roi = select_roi_on_frame(tfile.name)
            if roi != (0, 0, 0, 0):
                st.session_state.roi = roi
                st.success("ROI selected.")
            else:
                st.warning("No ROI selected.")

        if st.session_state.roi and st.sidebar.button("👁️ Preview ROI + Threshold"):
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            cap.release()
            if ret:
                x, y, w, h = st.session_state.roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                if use_threshold:
                    cv2.line(frame, (0, threshold_slider), (frame.shape[1], threshold_slider), (0, 255, 255), 2)
                st.image(frame, channels="BGR", caption="ROI Preview", use_container_width=True)

        if st.sidebar.button("🎯 Run Detection"):
            st.session_state.run_detection = True

        if st.session_state.run_detection and st.session_state.roi:
            output_path = os.path.join(tempfile.gettempdir(), "vehicle_count_result.mp4")

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            stframe = st.empty()
            progress_bar = st.progress(0)

            x, y, w, h = st.session_state.roi
            frame_area = width * height

            threshold_y = threshold_slider
            total_counts = defaultdict(int)
            frame_idx = 0

            ct = CentroidTracker()
            object_id_to_class = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

                roi_frame = frame[y:y + h, x:x + w]
                results = model(roi_frame, verbose=False, device=0 if device == "cuda" else "cpu")[0]

                vehicle_pixels = 0
                centers = []
                center_to_class = {}

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    if cls_name in selected_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1 += x
                        x2 += x
                        y1 += y
                        y2 += y
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        centers.append(center)
                        center_to_class[center] = cls_name

                        color = class_colors.get(cls_name, (0, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        vehicle_pixels += (x2 - x1) * (y2 - y1)

                objects = ct.update(np.array(centers))

                if use_threshold:
                    cv2.line(frame, (0, threshold_y), (frame.shape[1], threshold_y), (0, 255, 255), 2)

                for object_id, centroid in objects.items():
                    cx, cy = centroid
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

                    if object_id not in ct.counted_ids and cy > threshold_y:
                        matched_cls = None
                        for center, cls_name in center_to_class.items():
                            if np.linalg.norm(np.array(center) - np.array(centroid)) < 20:
                                matched_cls = cls_name
                                break

                        if matched_cls:
                            total_counts[matched_cls] += 1
                        total_counts["vehicles"] += 1
                        ct.counted_ids.add(object_id)

                density_ratio = vehicle_pixels / frame_area
                cv2.putText(frame, f"Density: {density_ratio:.2%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(frame, f"Vehicles Counted: {total_counts['vehicles']}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                y_offset = 120
                for cls in vehicle_classes:
                    count = total_counts.get(cls, 0)
                    label = f"{cls.capitalize()}: {count}"
                    cv2.putText(frame, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[cls], 2)
                    y_offset += 30

                out.write(frame)
                stframe.image(frame, channels="BGR", use_container_width=True)

            cap.release()
            out.release()
            st.success("✅ Processing complete!")

            st.subheader("📊 Vehicle Crossed Threshold")
            st.write(f"**Total Count (All Vehicles):** {total_counts['vehicles']}")
            for cls in vehicle_classes:
                st.write(f"- **{cls.capitalize()}s:** {total_counts.get(cls, 0)}")

            # Export to CSV
            summary_data = {k: [v] for k, v in total_counts.items()}
            df_summary = pd.DataFrame(summary_data)
            csv = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button("📄 Export Summary to CSV", csv, "vehicle_summary.csv", "text/csv")

            # Download video
            with open(output_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Output Video",
                    data=file,
                    file_name="vehicle_count_result.mp4",
                    mime="video/mp4"
                )

        elif not st.session_state.roi:
            st.warning("Please select an ROI before running detection.")

elif input_source == "Live Camera":
    run_live_camera()
