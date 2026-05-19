import cv2
import easyocr
import imutils

# Initialize OCR
reader = easyocr.Reader(['en'])

def detect_number_plate(frame):

    plate_text = "UNKNOWN"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:10]

    screenCnt = None

    for c in contours:

        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(
            c,
            0.018 * peri,
            True
        )

        # Number plate rectangle
        if len(approx) == 4:

            screenCnt = approx
            break

    if screenCnt is not None:

        x, y, w, h = cv2.boundingRect(screenCnt)

        plate = frame[y:y + h, x:x + w]

        # OCR text extraction
        result = reader.readtext(plate)

        for res in result:

            plate_text = res[1]

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            plate_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return frame, plate_text