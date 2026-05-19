import streamlit as st
import cv2
import av
import time

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from risk_score import calculate_risk
from emotion_detection import detect_emotion
from adaptive_signal import get_signal_time
from accident_prediction import detect_accident
from database import insert_violation
from report_generator import generate_report

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

class HelmetProcessor(VideoProcessorBase):

    def __init__(self):

        self.last_save_time = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # =========================
        # NUMBER PLATE
        # =========================

        plate_number = "TN01AB1234"

        # =========================
        # YOLO DETECTION
        # =========================

        results = model(img, verbose=False)

        helmet_detected = False
        no_helmet_detected = False

        # =========================
        # DETECTION LOOP
        # =========================

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # =========================
                # MODEL LABELS
                # 0 = no_helmet
                # 1 = helmet
                # =========================

                if cls == 1:

                    helmet_detected = True

                    color = (0, 255, 0)

                    label = "Helmet"

                else:

                    no_helmet_detected = True

                    color = (0, 0, 255)

                    label = "No Helmet"

                # =========================
                # DRAW BOX
                # =========================

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        # =========================
        # SIGNAL LOGIC
        # =========================

        if helmet_detected:

            signal = "GREEN"

            message = "Helmet Detected"

            signal_color = (0, 255, 0)

        elif no_helmet_detected:

            signal = "RED"

            message = "No Helmet Detected"

            signal_color = (0, 0, 255)

        else:

            signal = "YELLOW"

            message = "No Detection"

            signal_color = (0, 255, 255)

        # =========================
        # RISK SCORE
        # =========================

        risk_score = calculate_risk(
            helmet=helmet_detected,
            triple_riding=False,
            mobile_usage=False,
            overspeed=False
        )

        # =========================
        # EMOTION DETECTION
        # =========================

        emotion = "Unknown"

        if no_helmet_detected:

            emotion = detect_emotion(img)

        # =========================
        # SIGNAL TIMER
        # =========================

        green_time = get_signal_time(
            risk_score
        )

        # =========================
        # ACCIDENT PREDICTION
        # =========================

        accident_risk = detect_accident(
            90,
            True
        )

        # =========================
        # DATABASE + PDF GENERATION
        # =========================

        current_time = time.time()

        if no_helmet_detected:

            if current_time - self.last_save_time > SAVE_INTERVAL:

                # SAVE DATABASE

                insert_violation(
                    f"NO HELMET | Plate: {plate_number}"
                )

                # GENERATE PDF

                generate_report(

                    helmet_status=message,

                    risk_score=risk_score,

                    emotion=emotion,

                    signal=signal,

                    plate_number=plate_number,

                    accident_risk=accident_risk
                )

                print("Violation Saved")
                print("PDF Generated")

                self.last_save_time = current_time

        # =========================
        # DRAW SIGNAL
        # =========================

        cv2.circle(
            img,
            (80, 80),
            30,
            signal_color,
            -1
        )

        # =========================
        # DISPLAY TEXT
        # =========================

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

        # =========================
        # ACCIDENT ALERT
        # =========================

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

        # =========================
        # PROJECT TITLE
        # =========================

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
    desired_playing_state=True
)