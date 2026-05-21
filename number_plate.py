import cv2
import easyocr
import imutils

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        # Initialize EasyOCR lazily on CPU to guarantee thread-safety inside Streamlit WebRTC threads
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

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
        result = get_reader().readtext(plate)

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

    # ====================================================
    # FALLBACK: SCAN ENTIRE FRAME IF CONTOURS FAILED OR UNKNOWN
    # ====================================================
    if plate_text == "UNKNOWN" or len(plate_text.strip()) < 4:
        results = get_reader().readtext(frame)
        best_candidate = None
        best_box = None
        
        # Phase 1: Try to find a strict license plate format (e.g. MH12AB1234)
        for res in results:
            text = res[1].strip().upper()
            alphanumeric = "".join([c for c in text if c.isalnum()])
            has_letter = any(c.isalpha() for c in alphanumeric)
            has_digit = any(c.isdigit() for c in alphanumeric)
            
            if len(alphanumeric) >= 5 and len(alphanumeric) <= 15 and has_letter and has_digit:
                best_candidate = text
                best_box = res[0]
                break
                
        # Phase 2: If no strict format, take the longest text block with at least some numbers/letters
        if not best_candidate:
            for res in results:
                text = res[1].strip().upper()
                alphanumeric = "".join([c for c in text if c.isalnum()])
                if len(alphanumeric) >= 4:
                    best_candidate = text
                    best_box = res[0]
                    break
                    
        # If we successfully found a fallback candidate, draw and save it
        if best_candidate:
            plate_text = best_candidate
            box = best_box
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            tx, ty = int(min(xs)), int(min(ys))
            tw, th = int(max(xs) - tx), int(max(ys) - ty)
            
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
            cv2.putText(
                frame,
                plate_text,
                (tx, ty - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    return frame, plate_text