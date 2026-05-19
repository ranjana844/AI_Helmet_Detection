import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame not captured")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()

try:
    cv2.destroyAllWindows()
except:
    pass