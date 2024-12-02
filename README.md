# COCO YOLO w/ RSTP. 

*Note* This is a trash heap I threw together just to test. See env.example for variables you need to define. Error handling is sub par and I did not like the slack class method in main among other glaring issues. Feel free to fork and make improvements. 

## Variable definition:
* `NOTIFY_IF` is a list of items you want to detect, e.g., `dog,cat`.
```bash
NOTIFY_IF="dog,cat"
```
* `RSTP_URLS` is a list of camera RTSP feeds to watch.
```bash
RSTP_URLS="rtsps://192.168.1.100:7441/ABCD,rtsps://192.168.1.100:7441/EFGH"
```
* `SLACK_WEBHOOK_URL` is the webhook URL provided to you by Slack. You need to create an app for this to work.
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/ABCDEFG/HIJKLMNOP/QrStUvWxYz
```

* `SLACK_CHANNEL` is the Slack channel where notifications will be sent.
* `X_APP_BEARER_TOKEN` is the bearer token for the application.
* `NOTIFY_IF` is a list of items you want to detect, e.g., `dog,cat,person`.
* `NOTIFY_SLACK` is a boolean to enable or disable Slack notifications.
* `RESET_CAPTURE_COUNT_TIMER` is the timer in seconds to reset the capture count.
* `MAX_COUNT_CAPTURE` is the maximum number of captures before resetting.
* `CONFIDENCE_THRESHOLD` is the confidence threshold for detections, e.g., `0.7`.
* `LOG_LEVEL` is the logging level, e.g., `DEBUG`.
* `INFERENCE_DEVICE` is the device to run inference on, e.g., `cpu`.

## Build 
```bash
docker build -t yolo-rtsp-app . 
```
## Run
```bash 
docker run --rm -it yolo-rtsp-app
```

You can change things up when you run the docker container by setting env variables on the fly.
```bash
 docker run --rm -e NOTIFY_IF="tv,chair" -e CONFIDENCE_THRESHOLD=0.6  -e RSTP_URLS="rtsps://192.168.1.100:7441/ABCD" -it yolo-rtsp-app
 ```

 # Sagemaker training
  There is a seperate folder called dataset for custom training I omitted. You can label custom images using a tool like cvat and then export them. Ex: If you wanted to detect a specific person.  