import detect_fall as detect_fall

class DetectorSelector:
    def __init__(self, detectors, results, confidence_threshold):
        self.detectors = detectors
        self.results = results
        self.confidence_threshold = confidence_threshold

    def select(self, target_classes):
        if "fall" in self.detectors and "person" in target_classes:
            detect_fall.detect_fall(self.results, target_classes, self.confidence_threshold)
            


            
            