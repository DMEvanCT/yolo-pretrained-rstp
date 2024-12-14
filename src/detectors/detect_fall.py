import logging

logger = logging.getLogger(__name__)

def detect_fall(results, target_class, confidence_threshold):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int()
            fall = False 
            # check if width is greater than height
            if box.cls[0] == target_class and box.conf.item() >= confidence_threshold:
                if x2 - x1 > y2 - y1:
                    return fall             
    return fall