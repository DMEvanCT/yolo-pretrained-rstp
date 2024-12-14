import cv2 
import os
import boto3

def write_to_disk(results, output_dir, detected_count):
        annotated_frame = results[0].plot()
        image_name = f"detected_frame_{detected_count:04d}.jpg"
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, annotated_frame)
        return output_path
