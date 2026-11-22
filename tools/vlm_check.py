
import cv2

from visionassist.vlm import describe_scene


def main():
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            print("CAMERA ERROR: Could not open webcam.")
            return 1

        ok, frame = cap.read()

        if not ok or frame is None:
            print("CAMERA ERROR: Could not capture a frame.")
            return 1
    finally:
        cap.release()

    print("Calling OpenRouter VLM (may take several seconds)...")

    try:
        desc = describe_scene(frame)
        print(f"VLM RESULT: {desc!r}")
        return 0
    except Exception as exc:
        print(f"VLM ERROR: {type(exc).__name__}: {exc}")

        response = getattr(exc, "response", None)
        if response is not None:
            print(f"Response body: {response.text[:500]}")

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
