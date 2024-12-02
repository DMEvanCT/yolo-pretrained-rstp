import os
from ultralytics import YOLO
import cv2
import requests
import time
from dotenv import load_dotenv
import threading
import logging
from notification.slack import SlackNotifier
load_dotenv()

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
X_APP_BEARER_TOKEN = os.getenv("X_APP_BEARER_TOKEN", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "")
NOTIFY_SLACK = os.getenv("NOTIFY_SLACK", "false").lower()
# How many iamges it will capture and send to slack
MAX_COUNT_CAPTURE = os.getenv("MAX_COUNT_CAPTURE", 5)
RESET_CAPTURE_COUNT_TIMER = os.getenv("RESET_CAPTURE_COUNT_TIMER", 60)
CONFIDENCE_THRESHOLD = os.getenv("CONFIDENCE_THRESHOLD", 0.5)
# mps for mac cuda for nvidia and cpu for none
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")

slack_notifier = SlackNotifier(SLACK_WEBHOOK_URL, X_APP_BEARER_TOKEN, SLACK_CHANNEL)

try:
    model = YOLO("yolov11-coco.pt")  # Replace with your model file
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.info(f"Error loading YOLO model: {e}")
    exit()

# Get the classes to detect from the environment variable
notify_if = os.getenv("NOTIFY_IF", "dog")  # Default to "dog" if not set
target_classes = [cls.strip().lower() for cls in notify_if.split(",")]
logger.info(f"Target classes for detection: {target_classes}")

# RTSP Stream URL
rtsp_url = os.getenv("RSTP_URLS") # Replace with your RTSP link
# List of urls
rstp_urls = [cls.strip().lower() for cls in rtsp_url.split(",")]

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    logger.info("Error: Unable to open RTSP stream. Check the URL or network connection.")
    exit()

# Create output directory
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory: {output_dir}")

frame_count = 0
detected_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        logger.info("Error: Unable to read frame from stream. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)  # Reinitialize the stream
        continue

    try:
        # Perform inference
        results = model(frame, device=INFERENCE_DEVICE)

        # Check for target class detection
        target_detected = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                class_name = result.names[class_id].lower()
                # get the probability from tensor  tensor([0.6411])
                probablity = box.conf.item()
                logger.info(f"Detected: {class_name} with probability: {probablity}")
                if class_name in target_classes and probablity >= float(CONFIDENCE_THRESHOLD):  # Check if the class is in target_classes
                    logger.info(f"Detected: {class_name}")
                    target_detected = True
                    break
            if target_detected:
                break

        # If a target class is detected, log the frame
        if target_detected:
            annotated_frame = results[0].plot()
            output_path = os.path.join(output_dir, f"detected_frame_{detected_count:04d}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            # @TODO: add upload to S3
            logger.info(f"Target class detected! Saved frame {detected_count} to {output_path}")
            if NOTIFY_SLACK == "true" and detected_count < int(MAX_COUNT_CAPTURE):
                logger.info(NOTIFY_SLACK)
                slack_notifier.upload_image_to_slack(output_path, text=f"Target class detected: {class_name}!")
            detected_count += 1

            # Reset detected count after RESET_CAPTURE_COUNT_TIMER seconds
            if detected_count == 1:
                start_time = time.time()
            elif time.time() - start_time >= int(RESET_CAPTURE_COUNT_TIMER):
                detected_count = 0
                logger.info(f"Detected count reset after {RESET_CAPTURE_COUNT_TIMER} seconds.")

        # Increment frame count
        frame_count += 1
    except Exception as e:
        logger.info(f"Error during inference or saving frame: {e}")
        break

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Exit requested by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

logger.info(f"Total frames processed: {frame_count}")
logger.info(f"Frames with target classes saved: {detected_count}")
def process_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.info(f"Error: Unable to open RTSP stream {rtsp_url}. Check the URL or network connection.")
        return

    frame_count = 0
    detected_count = 0
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Error: Unable to read frame from stream {rtsp_url}. Retrying...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)  # Reinitialize the stream
            continue

        try:
            # Perform inference
            results = model(frame)

            # Check for target class detection
            target_detected = False
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])  # Get class ID
                    class_name = result.names[class_id].lower()
                    probablity = box.conf.item()
                    logger.info(f"Detected: {class_name} with probability: {probablity}")
                    if class_name in target_classes and probablity >= float(CONFIDENCE_THRESHOLD):
                        logger.info(f"Detected: {class_name}")
                        target_detected = True
                        break
                if target_detected:
                    break

            # If a target class is detected, log the frame
            if target_detected:
                annotated_frame = results[0].plot()
                output_path = os.path.join(output_dir, f"detected_frame_{detected_count:04d}.jpg")
                cv2.imwrite(output_path, annotated_frame)
                logger.info(f"Target class detected! Saved frame {detected_count} to {output_path}")
                if NOTIFY_SLACK == "true" and detected_count < int(MAX_COUNT_CAPTURE):
                    logger.info(NOTIFY_SLACK)
                    slack_notifier.upload_image_to_slack(output_path, text=f"Target class detected: {class_name}!")
                detected_count += 1

                # Reset detected count after RESET_CAPTURE_COUNT_TIMER seconds
                if detected_count == 1:
                    start_time = time.time()
                elif time.time() - start_time >= int(RESET_CAPTURE_COUNT_TIMER):
                    detected_count = 0
                    logger.info(f"Detected count reset after {RESET_CAPTURE_COUNT_TIMER} seconds.")

            # Increment frame count
            frame_count += 1
        except Exception as e:
            logger.info(f"Error during inference or saving frame: {e}")
            break

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Exit requested by user.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"Total frames processed: {frame_count} for stream {rtsp_url}")
    logger.info(f"Frames with target classes saved: {detected_count} for stream {rtsp_url}")

# Create and start threads for each RTSP URL
threads = []
for url in rstp_urls:
    thread = threading.Thread(target=process_stream, args=(url,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
