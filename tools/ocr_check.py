
import cv2

from visionassist.ocr import read_text_from_frame

cap = cv2.VideoCapture(0, cv2.CAP_ANY)

if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    raise SystemExit(1)

window = "OCR Test - O=Read Text | Q/Esc=Quit"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

print("Camera started.")
print("Click the camera window first.")
print("Press O to read text, Q or Esc to quit.")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Camera frame unavailable.")
            break

        cv2.imshow(window, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("o"), ord("O")):
            print("Reading text...")
            result = read_text_from_frame(frame)
            print("OCR RESULT:", repr(result))

        elif key in (ord("q"), ord("Q"), 27):
            print("Closing camera...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")
