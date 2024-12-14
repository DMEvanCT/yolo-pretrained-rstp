
import logging
import cv2

logger = logging.getLogger(__name__)

class RSTPCapture():
    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
    
    def open_stream(self):
        logger.info(f"Opening RTSP stream: {self.url}")
        ret, frame = self.cap.read()
        if not ret:
            logger.info("Error: Unable to read frame from stream. Retrying...")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.url)  # Reinitialize the stream
            if not self.cap.isOpened():
                logger.info("Error: Unable to open RTSP stream. Check the URL or network connection.")   
        return ret, frame 
    
    def close_stream(self):
        self.cap.release()
        cv2.destroyAllWindows()

