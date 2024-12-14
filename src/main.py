import os
import logging
import threading
import time
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
from notification.select_notification import SelectNotification
from capture.capture_rstp import RSTPCapture
from detectors.detect_default import DetectDefault

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

MAX_COUNT_CAPTURE = int(os.getenv("MAX_COUNT_CAPTURE", 5))
RESET_CAPTURE_COUNT_TIMER = int(os.getenv("RESET_CAPTURE_COUNT_TIMER", 60))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")
DETECTORS = os.getenv("DETECTORS", "falls").lower()
SAVE_FRAMES_TO_DISK = bool(os.getenv("SAVE_FRAMES_TO_DISK", "false").lower())
NOTIFICATION_SYSTEMS = os.getenv("NOTIFY_SYSTEMS", "").lower()
NOTIFY_IF = os.getenv("NOTIFY_IF", "dog").lower()
RTSP_URLS = os.getenv("RTSP_URLS", "").split(",")

# Initialize notification system
notify_selector = SelectNotification(notification_systems=NOTIFICATION_SYSTEMS)

# Load YOLO model
try:
    model = YOLO("yolov11-coco.pt")  # Replace with the correct model path
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    exit()

# Target classes
target_classes = [cls.strip() for cls in NOTIFY_IF.split(",")]
logger.info(f"Target classes for detection: {target_classes}")

# Output directory setup
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

def process_stream(rtsp_url):
    """
    Process an RTSP stream and perform object detection.
    """
    frame_count = 0
    detected_count = 0
    start_time = None

    while True:
        capture = RSTPCapture(rtsp_url)
        ret, frame = capture.open_stream()
        if not ret:
            logger.error(f"Failed to open stream: {rtsp_url}")
            break

        try:
            results = model(frame)
            target_detected = False
            detected_class = None

            for result in results:
                detect = DetectDefault(results=result, target_class=target_classes, confidence_threshold=CONFIDENCE_THRESHOLD)
                target_detected = detect.detect_target_class()
               
            
            if target_detected:
                detected_class = detect.get_detected_class()  # Assuming get_detected_class() returns the detected class
                if detected_count <= MAX_COUNT_CAPTURE:
                    if SAVE_FRAMES_TO_DISK:
                        logger.info(f"Max frame count {MAX_COUNT_CAPTURE}")
                        logger.info(f"Detected count: {detected_count} ")
                        output_path = detect.save_image_to_disk(frame, output_dir, detected_count)

                    if NOTIFICATION_SYSTEMS != "":
                        logger.info(f"Sending notification for target class '{detected_class}' detected.")
                        notify_selector.send_message(f"Target class '{detected_class}' detected!", output_path)

                    detected_count += 1
                else:
                    logger.info(f"Max count capture reached: {MAX_COUNT_CAPTURE}. Stopping save to disk and notifications.")

                if detected_count == 1:
                    start_time = time.time()
                elif time.time() - start_time >= RESET_CAPTURE_COUNT_TIMER:
                    detected_count = 0
                    logger.info("Detected count reset after timer elapsed.")

            frame_count += 1
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            break

    capture.close_stream()
    logger.info(f"Stream processing completed: {rtsp_url}")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Frames with target classes saved: {detected_count}")

def main():
    """
    Main function to start threads for each RTSP URL.
    """
    threads = []
    for url in RTSP_URLS:
        thread = threading.Thread(target=process_stream, args=(url,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()