import os
from loguru import logger
import sys
import boto3
import datetime

ENV = os.getenv("ENV")
AWS_REGION = os.getenv("AWS_REGION")
AWS_LOG_GROUP_NAME = os.getenv("AWS_LOG_GROUP_NAME")
AWS_LOG_STREAM_NAME = os.getenv("AWS_LOG_STREAM_NAME")

TIMESTAMP = int(datetime.datetime.utcnow().timestamp() * 1000)
LOG_STREAM_NAME = f"{AWS_LOG_STREAM_NAME}{TIMESTAMP}"

if ENV == "PRODUCTION":
    try:
        client = boto3.client("logs", region_name=AWS_REGION)
    except Exception as e:
        logger.critical(f"Failed to initialize AWS CloudWatch client: {e}")
        sys.exit(1)

    try:
        client.create_log_group(logGroupName=AWS_LOG_GROUP_NAME)
    except client.exceptions.ResourceAlreadyExistsException:
        pass
    except Exception as e:
        logger.critical(f"Failed to create log group: {e}")
        sys.exit(1)

    try:
        client.create_log_stream(logGroupName=AWS_LOG_GROUP_NAME, logStreamName=LOG_STREAM_NAME)
    except client.exceptions.ResourceAlreadyExistsException:
        pass
    except Exception as e:
        logger.critical(f"Failed to create log stream: {e}")
        sys.exit(1)

    sequence_token = None

def send_to_cloudwatch(message, retry=True):
    """Send logs to AWS CloudWatch, failing the app if logging is unavailable."""
    global sequence_token

    log_event = {
        "logGroupName": AWS_LOG_GROUP_NAME,
        "logStreamName": LOG_STREAM_NAME,
        "logEvents": [
            {
                "timestamp": int(datetime.datetime.utcnow().timestamp() * 1000),
                "message": message.strip()
            }
        ]
    }

    if sequence_token:
        log_event["sequenceToken"] = sequence_token

    try:
        response = client.put_log_events(**log_event)
        sequence_token = response.get("nextSequenceToken")

    except client.exceptions.InvalidSequenceTokenException as e:
        if retry:
            sequence_token = e.response["expectedSequenceToken"]
            send_to_cloudwatch(message, retry=False)  # Retry with updated token
        else:
            logger.critical("CloudWatch sequence token error, logging permanently failed.")
            sys.exit(1)  # Stop the application

    except Exception as e:
        logger.critical(f"Failed to send logs to CloudWatch: {e}")
        sys.exit(1)  # Stop the application

class CloudWatchSink:
    """Custom Loguru Sink to send logs to AWS CloudWatch. Fails the app if logging fails."""

    def write(self, message):
        send_to_cloudwatch(message)

logger.remove()
logger.add(sys.stdout, level="DEBUG")
# logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")
# logger.add("logs/app.log", rotation="1 MB", retention="10 days", level="DEBUG")
if ENV == "PRODUCTION":
    logger.add(CloudWatchSink(), level="DEBUG")

def setup_logger():
    """Set up global exception handling for unhandled errors."""
    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.exit(1)

    sys.excepthook = log_exception
    logger.info("CloudWatch Logging Initialized Successfully")
