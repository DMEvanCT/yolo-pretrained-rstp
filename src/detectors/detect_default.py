import logging
import cv2

logger = logging.getLogger(__name__)

class DetectDefault():
    def __init__(self, detected_class=None, results=None, confidence_threshold=0.5, target_class="person"):
        self.target_detected = False
        self.detected_class = detected_class
        self.results = results
        self.confidence_threshold = confidence_threshold
        self.target_class = target_class

    def get_class_id(self, box):
            return int(box.cls[0])
    
    def get_class_name(self, result, class_id):
        return result.names[class_id].lower()
    
    def get_probability(self, box):
        return box.conf.item()
    
    def detect_target_class(self):
        for result in self.results:
            for box in result.boxes:
                class_id = self.get_class_id(box)
                class_name = self.get_class_name(result, class_id)
                probability = self.get_probability(box)
                if class_name in self.target_class and probability >= self.confidence_threshold:
                    self.target_detected = True
                    self.detected_class = class_name
                    break
            if self.target_detected:
                break
        return self.target_detected
    
    def save_image_to_disk(self, output_dir, detected_count):
        annotated_frame = self.results[0].plot()
        output_file = f"{output_dir}/frame_{detected_count}.jpg"
        cv2.imwrite(output_file, annotated_frame)
        logger.info(f"Frame saved to disk: {output_file}")
        return output_file
    
