import os 
import logging
import requests

logger = logging.getLogger(__name__)

class SlackNotifier:
    def __init__(self, slack_webhook_url, x_app_token, slack_channel):
        self.slack_webhook_url = slack_webhook_url
        self.x_app_token = x_app_token
        self.slack_channel = slack_channel

    def upload_image_to_slack(self, image_path, text=None):
        """
        Upload an image to a Slack channel using the new Slack API methods.

        Args:
            image_path (str): Path to the image file to upload.
            channel_id (str): Slack channel ID to upload the image to.
            token (str): Slack Bot User OAuth Token.
            text (str, optional): Initial comment to accompany the image.

        Returns:
            dict: Response from Slack API with upload details.
        """
        try:
            # Step 1: Get the upload URL
            file_size = os.path.getsize(image_path)
            file_name = os.path.basename(image_path)
            headers = {
                "Authorization": f"Bearer {self.x_app_token}",
            }
            params = {
                "filename": file_name,
                "length": file_size
            }
            response = requests.get(
                "https://slack.com/api/files.getUploadURLExternal",
                headers=headers,
                params=params
            )
            response_data = response.json()
            if not response_data.get("ok"):
                logger.info("Failed to get upload URL:", response_data.get("error"))
                return response_data

            upload_url = response_data["upload_url"]
            file_id = response_data["file_id"]

            # Step 2: Upload the file
            with open(image_path, "rb") as file_content:
                upload_response = requests.post(
                    upload_url,
                    files={"file": file_content}
                )
            if upload_response.status_code != 200:
                logger.info("Failed to upload file:", upload_response.text)
                return {"ok": False, "error": upload_response.text}

            # Step 3: Complete the upload
            complete_payload = {
                "files": [
                    {
                        "id": file_id,
                        "title": file_name
                    }
                ],
                "channel_id": self.slack_channel,
                "initial_comment": text or ""
            }
            complete_response = requests.post(
                "https://slack.com/api/files.completeUploadExternal",
                headers={
                    "Authorization": f"Bearer {self.x_app_token}",
                    "Content-Type": "application/json"
                },
                json=complete_payload
            )
            complete_data = complete_response.json()
            if complete_data.get("ok"):
                logger.info(f"Image uploaded successfully! File ID: {file_id}")
            else:
                logger.info("Failed to complete upload:", complete_data.get("error"))
            return complete_data

        except Exception as e:
            logger.info("An error occurred:", str(e))
            return {"ok": False, "error": str(e)}