import streamlit as st
try:
    import cv2
except:
    st.error("OpenCV not installed properly")
import av
import time
import threading

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from risk_score import calculate_risk
from emotion_detection import detect_emotion
from adaptive_signal import get_signal_time
from accident_prediction import detect_accident
from database import insert_violation
from report_generator import generate_report
from number_plate import detect_number_plate

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Helmet Traffic System",
    layout="centered"
)

st.title("AI Helmet Traffic System")

st.write("Live Helmet Detection Dashboard")

# =========================
# REDUCE CAMERA SIZE
# =========================

st.markdown(
    """
    <style>
    video {
        width: 500px !important;
        height: 350px !important;
        border-radius: 10px;
        margin: auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================

model = YOLO("best.pt")

print("Model Loaded Successfully")
print(model.names)

# =========================
# SETTINGS
# =========================

CONF_THRESHOLD = 0.50
SAVE_INTERVAL = 10

# =========================
# VIDEO PROCESSOR
# =========================

# =========================
# ASYNC VIOLATION PROCESSOR
# =========================

def process_violation_async(img_copy, helmet_detected, no_helmet_detected, current_time, processor_instance):
    try:
        # Run OCR in the background thread (takes 1-3 seconds, would freeze camera otherwise)
        _, plate_number = detect_number_plate(img_copy)
        
        # Run Emotion Detection in the background thread
        emotion = detect_emotion(img_copy)
        
        # Calculate Risk Score
        risk_score = calculate_risk(
            helmet=helmet_detected,
            triple_riding=False,
            mobile_usage=False,
            overspeed=False
        )
        
        # Get Adaptive Signal Green Time
        green_time = get_signal_time(risk_score)
        
        # Accident Prediction
        accident_risk = detect_accident(90, True)
        
        # Write to SQLite Database
        insert_violation(f"NO HELMET | Plate: {plate_number}")
        
        # Generate PDF Violation Report
        generate_report(
            helmet_status="No Helmet Detected",
            risk_score=risk_score,
            emotion=emotion,
            signal="RED",
            plate_number=plate_number,
            accident_risk=accident_risk
        )
        
        print(f"Async Violation Logged: Plate={plate_number}, Emotion={emotion}")
        
        # Cache results for rendering on the Streamlit display
        processor_instance.cached_plate_number = plate_number
        processor_instance.cached_emotion = emotion
        processor_instance.cached_risk_score = risk_score
        processor_instance.cached_green_time = green_time
        processor_instance.cached_accident_risk = accident_risk
        
    except Exception as e:
        print(f"Error in background violation logging: {e}")
    finally:
        processor_instance.is_processing_violation = False


class HelmetProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_save_time = 0
        self.last_process_time = 0
        
        # Threading lock
        self.is_processing_violation = False
        
        # Detection cache (for frame rate stabilization)
        self.cached_boxes = []
        self.cached_helmet_detected = False
        self.cached_no_helmet_detected = False
        
        # Text/Logic overlay cache
        self.cached_plate_number = "UNKNOWN"
        self.cached_emotion = "Unknown"
        self.cached_risk_score = 0
        self.cached_green_time = 0
        self.cached_accident_risk = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # ===================================================
        # 1. RATE LIMIT YOLO INFERENCE (Run at ~7 FPS)
        # ===================================================
        if current_time - self.last_process_time > 0.15:
            self.last_process_time = current_time
            results = model(img, imgsz=320, verbose=False)

            helmet_detected = False
            no_helmet_detected = False
            boxes_to_draw = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == 1:
                        helmet_detected = True
                        color = (0, 255, 0)
                        label = "Helmet"
                    else:
                        no_helmet_detected = True
                        color = (0, 0, 255)
                        label = "No Helmet"

                    boxes_to_draw.append((x1, y1, x2, y2, label, conf, color))

            self.cached_boxes = boxes_to_draw
            self.cached_helmet_detected = helmet_detected
            self.cached_no_helmet_detected = no_helmet_detected
        else:
            # Use cached YOLO boxes to maintain 30 FPS visual stream
            helmet_detected = self.cached_helmet_detected
            no_helmet_detected = self.cached_no_helmet_detected
            boxes_to_draw = self.cached_boxes

        # ===================================================
        # 2. RENDER DETECTION BOXES
        # ===================================================
        for x1, y1, x2, y2, label, conf, color in boxes_to_draw:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # ===================================================
        # 3. INTERACTIVE SIGNAL LOGIC
        # ===================================================
        if helmet_detected:
            signal = "GREEN"
            message = "Helmet Detected"
            signal_color = (0, 255, 0)
            risk_score = calculate_risk(helmet=True)
            green_time = get_signal_time(risk_score)
            accident_risk = detect_accident(90, True)
        elif no_helmet_detected:
            signal = "RED"
            message = "No Helmet Detected"
            signal_color = (0, 0, 255)
            # Use background-updated risk and accident scores
            risk_score = self.cached_risk_score if self.cached_risk_score > 0 else calculate_risk(helmet=False)
            green_time = self.cached_green_time if self.cached_green_time > 0 else get_signal_time(risk_score)
            accident_risk = self.cached_accident_risk
        else:
            signal = "YELLOW"
            message = "No Detection"
            signal_color = (0, 255, 255)
            risk_score = 0
            green_time = 5
            accident_risk = False

        # ===================================================
        # 4. ASYNC VIOLATION LOGGING (OCR, DB, PDF, EMOTION)
        # ===================================================
        if no_helmet_detected:
            if current_time - self.last_save_time > SAVE_INTERVAL:
                if not self.is_processing_violation:
                    self.is_processing_violation = True
                    self.last_save_time = current_time
                    
                    # Create deep copy of the image to send to thread
                    img_copy = img.copy()
                    
                    # Update screen UI indicator
                    self.cached_plate_number = "SCANNING OCR..."
                    self.cached_emotion = "SCANNING..."
                    
                    # Offload to background thread to prevent camera stutter
                    threading.Thread(
                        target=process_violation_async,
                        args=(img_copy, helmet_detected, no_helmet_detected, current_time, self),
                        daemon=True
                    ).start()

            plate_number = self.cached_plate_number
            emotion = self.cached_emotion
        else:
            plate_number = "UNKNOWN"
            emotion = "Unknown"
            # Reset scanning cache when frame is clear
            if not self.is_processing_violation:
                self.cached_plate_number = "UNKNOWN"
                self.cached_emotion = "Unknown"
                self.cached_risk_score = 0
                self.cached_green_time = 0
                self.cached_accident_risk = False

        # ===================================================
        # 5. DRAW GRAPHICAL OVERLAYS AND TEXT
        # ===================================================
        # Draw light indicator circle
        cv2.circle(
            img,
            (80, 80),
            30,
            signal_color,
            -1
        )

        cv2.putText(
            img,
            f"Signal: {signal}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            signal_color,
            2
        )

        cv2.putText(
            img,
            message,
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            signal_color,
            2
        )

        cv2.putText(
            img,
            f"Plate: {plate_number}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Risk Score: {risk_score}",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Emotion: {emotion}",
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            img,
            f"Timer: {green_time}s",
            (20, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if accident_risk:
            cv2.putText(
                img,
                "ACCIDENT RISK!",
                (20, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        cv2.putText(
            img,
            "AI Helmet Traffic System",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================
# START WEBCAM
# =========================

webrtc_streamer(
    key="helmet-system",
    video_processor_factory=HelmetProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    desired_playing_state=True
)