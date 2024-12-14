import sagemaker
from dotenv import load_dotenv
import os

load_dotenv()
# Replace with your bucket and dataset paths

bucket = os.getenv("S3_BUCKET")
dataset_path = os.getenv("DATASET_PATH")
output_path = f"s3://{bucket}/output"
# Retrieve YOLOv10 image URI
image_uri = os.getenv("IMAGE_URI")
# Define SageMaker role
role = os.getenv("SAGEMAKER_ROLE") 
instance_type = os.getenv("INSTANCE_TYPE", "ml.m5.xlarge")
# Create SageMaker Estimator
estimator = sagemaker.estimator.Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type=instance_type,
    output_path=output_path,
    use_spot_instances=True,
    max_wait=3600,  # Maximum wait time for spot instances in seconds
    max_run=3600    # Maximum run time for the training job in seconds
)

# Define input channels
inputs = {
    "all": f"s3://{bucket}/{dataset_path}/"
}

# Launch the training job
estimator.fit(inputs)