import cv2
import time
from ultralytics import YOLO

from database import insert_violation
from risk_score import calculate_risk
from emotion_detection import detect_emotion
from adaptive_signal import get_signal_time
from accident_prediction import detect_accident
from report_generator import generate_report
from number_plate import detect_number_plate

# =========================
# LOAD MODEL
# =========================

model = YOLO("best.pt")

print("Model Loaded Successfully")
print("Class Names:", model.names)

# ============================================
# CLASS FIX
# ============================================
#
# YOUR MODEL:
# 0 = helmet
# 1 = no_helmet
#
# BUT MODEL OUTPUT IS REVERSED
#
# SO:
# cls == 0  -> TREAT AS NO HELMET
# cls == 1  -> TREAT AS HELMET
#
# ============================================

# =========================
# SETTINGS
# =========================

CONF_THRESHOLD = 0.50
SAVE_INTERVAL = 10

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera not working")
    exit()

print("Camera Started Successfully")

# =========================
# VARIABLES
# =========================

plate_number = "UNKNOWN"

last_save_time = 0

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # =========================
    # RUN MODEL
    # =========================

    results = model(frame, verbose=False)

    helmet_detected = False
    no_helmet_detected = False

    best_conf = 0
    best_cls = -1
    best_box = None

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

            print("Class:", cls,
                  "| Label:", model.names[cls],
                  "| Confidence:", conf)

            # =========================
            # BEST CONFIDENCE ONLY
            # =========================

            if conf > best_conf:

                best_conf = conf
                best_cls = cls
                best_box = (x1, y1, x2, y2)

    # =========================
    # FINAL DETECTION
    # =========================

    if best_box is not None:

        x1, y1, x2, y2 = best_box

        # =====================================
        # REVERSED CLASSES
        # =====================================

        # cls 1 = HELMET
        if best_cls == 1:

            helmet_detected = True

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Helmet {best_conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # cls 0 = NO HELMET
        elif best_cls == 0:

            no_helmet_detected = True

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"No Helmet {best_conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    # =========================
    # NUMBER PLATE OCR DETECTION
    # =========================

    if no_helmet_detected:
        frame, plate_number = detect_number_plate(frame)

    # =========================
    # SIGNAL LOGIC
    # =========================

    if helmet_detected:

        signal = "GREEN"
        message = "Helmet Detected"
        color = (0, 255, 0)

    elif no_helmet_detected:

        signal = "RED"
        message = "No Helmet Detected"
        color = (0, 0, 255)

    else:

        signal = "YELLOW"
        message = "No Detection"
        color = (0, 255, 255)

    # =========================
    # RISK SCORE
    # =========================

    risk_score = calculate_risk(
        helmet=helmet_detected,
        triple_riding=False,
        mobile_usage=False,
        overspeed=False
    )

    emotion = "Unknown"

    if no_helmet_detected:
        emotion = detect_emotion(frame)

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
    # SAVE DATABASE + PDF
    # =========================

    now = time.time()

    if signal == "RED":

        if now - last_save_time > SAVE_INTERVAL:

            insert_violation(
                f"NO HELMET | Plate: {plate_number}"
            )

            generate_report(

                helmet_status=message,

                risk_score=risk_score,

                emotion=emotion,

                signal=signal,

                plate_number=plate_number,

                accident_risk=accident_risk
            )

            print("Violation Stored")
            print("PDF Generated")

            last_save_time = now

    # =========================
    # DRAW SIGNAL
    # =========================

    cv2.circle(
        frame,
        (100, 100),
        40,
        color,
        -1
    )

    # =========================
    # DISPLAY TEXT
    # =========================

    cv2.putText(
        frame,
        f"Signal: {signal}",
        (40, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        message,
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Plate: {plate_number}",
        (40, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Risk Score: {risk_score}",
        (40, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (40, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Signal Timer: {green_time}s",
        (40, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =========================
    # ACCIDENT WARNING
    # =========================

    if accident_risk:

        cv2.putText(
            frame,
            "ACCIDENT RISK!",
            (40, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =========================
    # PROJECT TITLE
    # =========================

    cv2.putText(
        frame,
        "AI Helmet Traffic System",
        (40, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # =========================
    # SHOW OUTPUT
    # =========================

    cv2.imshow(
        "AI Helmet Traffic System",
        frame
    )

    # =========================
    # EXIT
    # =========================

    if cv2.waitKey(1) == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()