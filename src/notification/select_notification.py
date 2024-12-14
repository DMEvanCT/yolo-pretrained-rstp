from notification.slack import SlackNotifier
import os
import logging


log = logging.getLogger(__name__)


class SelectNotification:
    def __init__(self, notification_systems: str):
        self.notification_systems = notification_systems
        self.SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
        self.X_APP_BEARER_TOKEN = os.getenv("X_APP_BEARER_TOKEN", "")
        self.SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "")
    
    def send_message(self, message, image_path=None):
        if "slack" in self.notification_systems:
            slack_notifier = SlackNotifier(slack_webhook_url=self.SLACK_WEBHOOK_URL, x_app_token=self.X_APP_BEARER_TOKEN, 
                                           slack_channel=self.SLACK_CHANNEL)
            if image_path is None:
                log.exception("Image path is required for slack notification")
            slack_notifier.upload_image_to_slack(image_path, message)


