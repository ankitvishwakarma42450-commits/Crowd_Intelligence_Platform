import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np
import datetime
import time

from pathlib import Path
from ultralytics import YOLO


# ==========================================
# IMPORTS
# ==========================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from analysis.density import calculate_density
from analysis.movement import calculate_movement
from analysis.risk import calculate_risk


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crowd Intelligence Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #FFFFFF;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}

h1, h2, h3 {
    color: #38BDF8;
}

p {
    color: #CBD5E1;
}

div[data-testid="stMetric"] {
    background-color: #172338;
    border: 1px solid #263B57;
    padding: 20px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8;
}

.stButton > button {
    background-color: #0284C7;
    color: white;
    border-radius: 8px;
    border: none;
    width: 100%;
    padding: 10px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0369A1;
    color: white;
}

hr {
    border-color: #263247;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path(__file__).with_name("yolo11s.pt")

HAZARD_MODEL_PATH = Path(__file__).with_name("fire_smoke.pt")

MAX_CAPACITY = 10
ZONE_AREA = 20

HAZARD_CONFIDENCE = 0.25

# Prevent repeated alerts for every video frame
ALERT_COOLDOWN = 20


# ==========================================
# SESSION STATE
# ==========================================

if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = None

if "previous_positions" not in st.session_state:
    st.session_state.previous_positions = {}

if "hazard_alerts" not in st.session_state:
    st.session_state.hazard_alerts = []

if "last_hazard_alert" not in st.session_state:
    st.session_state.last_hazard_alert = {}


# ==========================================
# LOAD YOLO MODELS
# ==========================================

@st.cache_resource
def load_model():

    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_hazard_model():

    return YOLO(str(HAZARD_MODEL_PATH))


try:

    model = load_model()

except Exception as e:

    st.error(f"Unable to load Person Detection Model: {e}")
    st.stop()


try:

    hazard_model = load_hazard_model()

except Exception as e:

    st.error(f"Unable to load Fire/Smoke Model: {e}")
    st.stop()


# ==========================================
# AUTOMATIC HAZARD ALERT
# ==========================================

def generate_hazard_alert(hazard_name, confidence):

    current_time = time.time()

    last_time = st.session_state.last_hazard_alert.get(
        hazard_name,
        0
    )

    if current_time - last_time >= ALERT_COOLDOWN:

        alert = {

            "id": len(st.session_state.hazard_alerts) + 1,

            "type": hazard_name.title() + " Detected",

            "confidence": round(confidence * 100, 2),

            "status": "Pending Review",

            "source": "AI Detection",

            "time": datetime.datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )

        }

        st.session_state.hazard_alerts.append(alert)

        st.session_state.last_hazard_alert[hazard_name] = current_time


# ==========================================
# FIRE AND SMOKE DETECTION
# ==========================================

def detect_hazards(frame, annotated_frame):

    results = hazard_model.predict(

        source=frame,

        conf=HAZARD_CONFIDENCE,

        imgsz=640,

        verbose=False

    )

    detections = []

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = hazard_model.names[class_id]

            hazard_name = str(class_name).lower()

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Draw hazard bounding box
            cv2.rectangle(

                annotated_frame,

                (x1, y1),

                (x2, y2),

                (0, 0, 255),

                2

            )

            label = (
                f"{class_name}: {confidence * 100:.1f}%"
            )

            # Draw label
            cv2.putText(

                annotated_frame,

                label,

                (x1, max(y1 - 10, 20)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2

            )

            detections.append({

                "type": str(class_name),

                "confidence": confidence

            })

            # Generate automatic alert
            generate_hazard_alert(
                hazard_name,
                confidence
            )

    return annotated_frame, detections


# ==========================================
# DETECTION FUNCTION
# ==========================================

def analyze_frame(frame):

    results = model.predict(

        source=frame,

        classes=[0],

        conf=0.20,

        imgsz=960,

        iou=0.45,

        verbose=False

    )

    annotated_frame = results[0].plot()

    person_count = len(results[0].boxes)

    # Fire and smoke detection
    annotated_frame, hazard_detections = detect_hazards(

        frame,

        annotated_frame

    )

    density, density_status = calculate_density(

        person_count,

        ZONE_AREA,

        MAX_CAPACITY

    )

    occupancy = (

        person_count / MAX_CAPACITY

    ) * 100

    risk_score, risk_level = calculate_risk(

        occupancy=occupancy,

        density_status=density_status

    )

    st.session_state.latest_metrics = {

        "people": person_count,

        "density": density,

        "density_status": density_status,

        "occupancy": occupancy,

        "risk_score": risk_score,

        "risk_level": risk_level,

        "hazards": hazard_detections

    }

    return (

        annotated_frame,

        person_count,

        density,

        density_status,

        occupancy,

        risk_score,

        risk_level

    )


# ==========================================
# MOVEMENT TRACKING FUNCTION
# ==========================================

def analyze_tracking_frame(frame):

    results = model.track(

        source=frame,

        persist=True,

        classes=[0],

        conf=0.20,

        imgsz=640,

        iou=0.45,

        verbose=False

    )

    result = results[0]

    annotated_frame = result.plot()

    boxes = result.boxes

    person_count = len(boxes)

    moving_count = 0

    stationary_count = 0

    directions = []

    current_positions = {}

    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()

        coordinates = boxes.xyxy.cpu().tolist()

        for track_id, box in zip(track_ids, coordinates):

            x1, y1, x2, y2 = box

            current_position = (

                (x1 + x2) / 2,

                (y1 + y2) / 2

            )

            previous_position = (

                st.session_state.previous_positions.get(track_id)

            )

            direction, distance = calculate_movement(

                previous_position,

                current_position

            )

            current_positions[track_id] = current_position

            if direction == "STATIONARY":

                stationary_count += 1

            elif direction != "UNKNOWN":

                moving_count += 1

                directions.append(direction)

    st.session_state.previous_positions = current_positions

    # Fire and smoke detection
    annotated_frame, hazard_detections = detect_hazards(

        frame,

        annotated_frame

    )

    density, density_status = calculate_density(

        person_count,

        ZONE_AREA,

        MAX_CAPACITY

    )

    occupancy = (

        person_count / MAX_CAPACITY

    ) * 100

    risk_score, risk_level = calculate_risk(

        occupancy=occupancy,

        density_status=density_status

    )

    st.session_state.latest_metrics = {

        "people": person_count,

        "density": density,

        "density_status": density_status,

        "occupancy": occupancy,

        "risk_score": risk_score,

        "risk_level": risk_level,

        "moving": moving_count,

        "stationary": stationary_count,

        "hazards": hazard_detections

    }

    return (

        annotated_frame,

        person_count,

        density,

        density_status,

        occupancy,

        risk_score,

        risk_level,

        moving_count,

        stationary_count,

        directions

    )


# ==========================================
# HAZARD RESULT DISPLAY
# ==========================================

def display_hazard_results():

    metrics = st.session_state.latest_metrics

    if metrics and metrics.get("hazards"):

        st.error("⚠️ Potential Hazard Detected!")

        for hazard in metrics["hazards"]:

            st.write(

                f"**{hazard['type']}** — "

                f"Confidence: {hazard['confidence'] * 100:.2f}%"

            )

    else:

        st.success("No Fire or Smoke Detected in the Latest Frame.")


# ==========================================
# RESULT DISPLAY
# ==========================================

def show_results(frame):

    (

        annotated_frame,

        person_count,

        density,

        density_status,

        occupancy,

        risk_score,

        risk_level

    ) = analyze_frame(frame)

    st.subheader("Detection Results")

    col1, col2 = st.columns(2)

    with col1:

        st.image(

            cv2.cvtColor(

                annotated_frame,

                cv2.COLOR_BGR2RGB

            ),

            channels="RGB",

            use_container_width=True

        )

    with col2:

        st.metric("People Detected", person_count)

        st.metric("Crowd Density", f"{density:.2f}")

        st.metric("Occupancy", f"{occupancy:.1f}%")

        st.metric("Density Status", density_status)

        st.metric("Risk Level", risk_level)

        st.metric("Risk Score", f"{risk_score}/100")

        st.progress(risk_score / 100)

    st.divider()

    st.subheader("🔥 Fire & Smoke Analysis")

    display_hazard_results()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.title("👥 CIP")

    st.caption("CROWD INTELLIGENCE PLATFORM")

    st.divider()

    st.subheader("Navigation")

    mode = st.radio(

        "Select Module",

        [

            "Overview",

            "Image Detection",

            "Video Detection",

            "Live Camera",

            "Crowd Analytics",

            "Risk Assessment",

            "Movement Analysis",

            "Hazard Alerts"

        ]

    )

    st.divider()

    st.subheader("System Information")

    st.success("Person Detection Model Loaded")

    st.success("Fire/Smoke Model Loaded")

    st.caption("Person Model: YOLO11 Small")

    st.caption("Hazard Model: Fire & Smoke YOLO")

    st.caption("Platform: Computer Vision")


# ==========================================
# HEADER
# ==========================================

st.title("👥 Crowd Intelligence Platform")

st.caption(

    "AI-Based Crowd Monitoring and Risk Assessment System"

)

st.divider()


# ==========================================
# OVERVIEW DASHBOARD
# ==========================================

if mode == "Overview":

    st.header("📊 Monitoring Dashboard")

    st.write(

        "Welcome to the Crowd Intelligence Platform. "

        "Monitor people, crowd density, potential risks "

        "and environmental hazards."

    )

    metrics = st.session_state.latest_metrics

    if metrics:

        c1, c2, c3 = st.columns(3)

        c1.metric("People Detected", metrics["people"])

        c2.metric("Crowd Density", f'{metrics["density"]:.2f}')

        c3.metric("Occupancy", f'{metrics["occupancy"]:.1f}%')

        c4, c5, c6 = st.columns(3)

        c4.metric("Density Status", metrics["density_status"])

        c5.metric("Risk Score", f'{metrics["risk_score"]}/100')

        c6.metric("Risk Level", metrics["risk_level"])

        st.subheader("Latest Hazard Information")

        if metrics.get("hazards"):

            for hazard in metrics["hazards"]:

                st.error(

                    f"{hazard['type']} detected — "

                    f"{hazard['confidence'] * 100:.2f}% confidence"

                )

        else:

            st.success("No hazard detected in the latest frame.")

    else:

        st.info(

            "No analysis available yet. "

            "Select Image Detection, Video Detection "

            "or Live Camera to begin."

        )

    st.divider()

    st.subheader("Available Modules")

    c1, c2, c3 = st.columns(3)

    c1.info("🖼️ Image Detection\n\nDetect people, fire and smoke.")

    c2.info("🎥 Video Detection\n\nAnalyse uploaded videos.")

    c3.info("📹 Live Camera\n\nMonitor live camera footage.")


# ==========================================
# IMAGE DETECTION
# ==========================================

elif mode == "Image Detection":

    st.header("🖼️ Image Detection")

    st.write(

        "Upload an image to detect people, fire and smoke."

    )

    uploaded_image = st.file_uploader(

        "Upload Image",

        type=["jpg", "jpeg", "png"]

    )

    if uploaded_image is not None:

        image_bytes = np.frombuffer(

            uploaded_image.getvalue(),

            dtype=np.uint8

        )

        frame = cv2.imdecode(

            image_bytes,

            cv2.IMREAD_COLOR

        )

        if frame is not None:

            show_results(frame)

        else:

            st.error("Unable to read uploaded image.")


# ==========================================
# VIDEO DETECTION + MOVEMENT TRACKING
# ==========================================

elif mode == "Video Detection":

    st.header("🎥 Video Detection")

    st.write(

        "Upload a video to detect people, fire, smoke "

        "and analyse crowd movement."

    )

    uploaded_video = st.file_uploader(

        "Upload Video",

        type=["mp4", "avi", "mov"]

    )

    if uploaded_video is not None:

        if st.button("Start Video Analysis"):

            temp_path = None

            cap = None

            st.session_state.previous_positions = {}

            try:

                suffix = Path(uploaded_video.name).suffix

                with tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=suffix

                ) as temp_file:

                    temp_file.write(

                        uploaded_video.getvalue()

                    )

                    temp_path = temp_file.name

                cap = cv2.VideoCapture(temp_path)

                video_display = st.empty()

                metrics_display = st.empty()

                if not cap.isOpened():

                    st.error("Unable to open uploaded video.")

                else:

                    while True:

                        ret, frame = cap.read()

                        if not ret:

                            break

                        (

                            annotated_frame,

                            person_count,

                            density,

                            density_status,

                            occupancy,

                            risk_score,

                            risk_level,

                            moving_count,

                            stationary_count,

                            directions

                        ) = analyze_tracking_frame(frame)

                        video_display.image(

                            cv2.cvtColor(

                                annotated_frame,

                                cv2.COLOR_BGR2RGB

                            ),

                            channels="RGB",

                            use_container_width=True

                        )

                        with metrics_display.container():

                            c1, c2, c3 = st.columns(3)

                            c1.metric(

                                "People Detected",

                                person_count

                            )

                            c2.metric(

                                "Crowd Density",

                                f"{density:.2f}"

                            )

                            c3.metric(

                                "Occupancy",

                                f"{occupancy:.1f}%"

                            )

                            c4, c5, c6 = st.columns(3)

                            c4.metric(

                                "Density Status",

                                density_status

                            )

                            c5.metric(

                                "Risk Score",

                                f"{risk_score}/100"

                            )

                            c6.metric(

                                "Risk Level",

                                risk_level

                            )

                            c7, c8 = st.columns(2)

                            c7.metric(

                                "Moving People",

                                moving_count

                            )

                            c8.metric(

                                "Stationary People",

                                stationary_count

                            )

                            st.write(

                                "Movement Directions:",

                                ", ".join(directions)

                                if directions

                                else "No movement detected"

                            )

                            st.divider()

                            st.subheader("🔥 Hazard Detection")

                            display_hazard_results()

                    st.success(

                        "Video analysis completed."

                    )

            except Exception as e:

                st.error(

                    f"Video processing error: {e}"

                )

            finally:

                if cap is not None:

                    cap.release()

                if temp_path and os.path.exists(temp_path):

                    try:

                        os.remove(temp_path)

                    except PermissionError:

                        st.warning(

                            "Temporary video file could not be deleted."

                        )


# ==========================================
# LIVE CAMERA
# ==========================================

elif mode == "Live Camera":

    st.header("📹 Live Camera Monitoring")

    st.write(
        "Monitor people, crowd conditions, fire and smoke "
        "using your connected webcam."
    )

    st.info("Camera access requires a connected webcam.")

    hazard_frequency = st.selectbox(
        "Hazard Detection Frequency",
        [5, 10, 15],
        index=1,
        help="Run fire/smoke detection after every selected number of frames."
    )

    if st.button("Start Live Camera"):

        cap = cv2.VideoCapture(0)

        # Reduce camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():

            st.error("Camera could not be opened.")

        else:

            video_display = st.empty()
            metrics_display = st.empty()

            frame_count = 0

            last_hazard_detections = []

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Resize frame for faster processing
                frame = cv2.resize(
                    frame,
                    (640, 480)
                )

                # Person detection
                results = model.predict(
                    source=frame,
                    classes=[0],
                    conf=0.25,
                    imgsz=640,
                    verbose=False
                )

                annotated_frame = results[0].plot()

                person_count = len(results[0].boxes)

                # Fire and smoke detection periodically
                if frame_count % hazard_frequency == 0:

                    annotated_frame, last_hazard_detections = detect_hazards(
                        frame,
                        annotated_frame
                    )

                else:

                    # Display the latest hazard results
                    for hazard in last_hazard_detections:

                        st.session_state.latest_metrics = (
                            st.session_state.latest_metrics or {}
                        )

                # Crowd calculations
                density, density_status = calculate_density(
                    person_count,
                    ZONE_AREA,
                    MAX_CAPACITY
                )

                occupancy = (
                    person_count / MAX_CAPACITY
                ) * 100

                risk_score, risk_level = calculate_risk(
                    occupancy=occupancy,
                    density_status=density_status
                )

                st.session_state.latest_metrics = {
                    "people": person_count,
                    "density": density,
                    "density_status": density_status,
                    "occupancy": occupancy,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "hazards": last_hazard_detections
                }

                # Display camera
                video_display.image(
                    cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    ),
                    channels="RGB",
                    use_container_width=True
                )

                with metrics_display.container():

                    c1, c2, c3 = st.columns(3)

                    c1.metric("People Detected", person_count)

                    c2.metric("Density Status", density_status)

                    c3.metric("Risk Level", risk_level)

                    st.subheader("🔥 Hazard Detection")

                    if last_hazard_detections:

                        for hazard in last_hazard_detections:

                            st.error(
                                f"{hazard['type']} detected "
                                f"({hazard['confidence'] * 100:.1f}%)"
                            )

                    else:

                        st.success(
                            "No hazard detected in the latest hazard scan."
                        )

            cap.release()

            st.success("Camera stopped.")


# ==========================================
# CROWD ANALYTICS
# ==========================================

elif mode == "Crowd Analytics":

    st.header("📊 Crowd Analytics")

    metrics = st.session_state.latest_metrics

    if metrics:

        c1, c2, c3 = st.columns(3)

        c1.metric("Total People", metrics["people"])

        c2.metric("Crowd Density", f'{metrics["density"]:.2f}')

        c3.metric("Occupancy", f'{metrics["occupancy"]:.1f}%')

        st.subheader("Crowd Occupancy")

        st.progress(

            min(metrics["occupancy"] / 100, 1.0)

        )

        st.write(

            f'Current occupancy: {metrics["occupancy"]:.1f}%'

        )

        if "moving" in metrics:

            c4, c5 = st.columns(2)

            c4.metric("Moving People", metrics["moving"])

            c5.metric("Stationary People", metrics["stationary"])

        st.info(

            "Analytics are based on the most recently processed frame."

        )

    else:

        st.warning(

            "Perform image, video or camera analysis first."

        )


# ==========================================
# RISK ASSESSMENT
# ==========================================

elif mode == "Risk Assessment":

    st.header("⚠️ Crowd Risk Assessment")

    metrics = st.session_state.latest_metrics

    if metrics:

        st.metric(

            "Risk Score",

            f'{metrics["risk_score"]}/100'

        )

        st.progress(

            metrics["risk_score"] / 100

        )

        st.metric("Risk Level", metrics["risk_level"])

        st.metric("Density Status", metrics["density_status"])

        st.metric("Occupancy", f'{metrics["occupancy"]:.1f}%')

        st.info(

            "Risk score is a rule-based estimate using "

            "occupancy and density status. It is not "

            "a prediction of an actual emergency."

        )

    else:

        st.warning(

            "Perform crowd analysis to calculate risk."

        )


# ==========================================
# MOVEMENT ANALYSIS
# ==========================================

elif mode == "Movement Analysis":

    st.header("📈 Movement Analysis")

    metrics = st.session_state.latest_metrics

    if metrics and "moving" in metrics:

        c1, c2, c3 = st.columns(3)

        c1.metric("People Detected", metrics["people"])

        c2.metric("Moving People", metrics["moving"])

        c3.metric("Stationary People", metrics["stationary"])

        st.info(

            "Movement statistics are based on the "

            "latest processed video frame."

        )

    else:

        st.warning(

            "Run Video Detection first to generate "

            "movement tracking data."

        )


# ==========================================
# HAZARD ALERT MANAGEMENT
# ==========================================

elif mode == "Hazard Alerts":

    st.header("🚨 Hazard Alert Management")

    st.write(

        "View and manage potential fire and smoke "

        "detection alerts."

    )

    st.info(

        "AI alerts are generated when the Fire/Smoke model "

        "detects a potential hazard. Demo alerts are simulated."

    )

    st.divider()

    # Demo alert
    st.subheader("Generate Demo Alert")

    if st.button("🚨 Generate Demo Hazard Alert"):

        alert = {

            "id": len(st.session_state.hazard_alerts) + 1,

            "type": "Potential Hazard - Demo",

            "confidence": None,

            "status": "Pending Review",

            "source": "Demo",

            "time": datetime.datetime.now().strftime(

                "%d-%m-%Y %H:%M:%S"

            )

        }

        st.session_state.hazard_alerts.append(alert)

        st.success("Demo alert generated successfully!")

    st.divider()

    # Alert statistics
    st.subheader("Alert Statistics")

    alerts = st.session_state.hazard_alerts

    total_alerts = len(alerts)

    pending_alerts = sum(

        1 for a in alerts

        if a["status"] == "Pending Review"

    )

    verified_alerts = sum(

        1 for a in alerts

        if a["status"] == "Verified"

    )

    dismissed_alerts = sum(

        1 for a in alerts

        if a["status"] == "Dismissed"

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Alerts", total_alerts)

    c2.metric("Pending", pending_alerts)

    c3.metric("Verified", verified_alerts)

    c4.metric("Dismissed", dismissed_alerts)

    st.divider()

    # Alert history
    st.subheader("Alert History")

    if alerts:

        for alert in reversed(alerts):

            with st.container(border=True):

                st.write(f"### 🚨 Alert #{alert['id']}")

                st.write(f"**Type:** {alert['type']}")

                st.write(f"**Source:** {alert['source']}")

                if alert.get("confidence") is not None:

                    st.write(

                        f"**Confidence:** "

                        f"{alert['confidence']:.2f}%"

                    )

                st.write(f"**Time:** {alert['time']}")

                st.write(f"**Status:** {alert['status']}")

                if alert["status"] == "Pending Review":

                    col1, col2 = st.columns(2)

                    with col1:

                        if st.button(

                            "Verify Alert",

                            key=f"verify_{alert['id']}"

                        ):

                            alert["status"] = "Verified"

                            st.rerun()

                    with col2:

                        if st.button(

                            "Dismiss Alert",

                            key=f"dismiss_{alert['id']}"

                        ):

                            alert["status"] = "Dismissed"

                            st.rerun()

    else:

        st.info(

            "No alerts available. Run image, video or live "

            "detection to generate AI alerts."

        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(

    "Crowd Intelligence Platform | "

    "AI-Based Crowd Monitoring System"

)

st.caption(

    "Developed using Python, OpenCV, YOLO and Streamlit."

)