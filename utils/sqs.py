import json
from dotenv import load_dotenv # type: ignore
import os
import boto3
from loguru import logger
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse, SuccessResponse

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")

class SQS():
    def __init__(self):
        try:
            logger.info("SQS client initialized successfully.")
            self.client = boto3.client(
                'sqs',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION
            )
            logger.info("SQS client initialized successfully.")
        except Exception as e:
            logger.critical(f"Exception at SQS __init__", e)

    async def get_sqs_messages(self)-> AppResponse:
        try:
            logger.info("Receiving messages from SQS...")
            response = self.client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=5,
                AttributeNames=['All']
            )

            messages = response.get('Messages', [])
            logger.info("Received messages from SQS", messages)
            
            if not messages:
                logger.info("No messages in queue")
                
            return SuccessResponse("No messages available", messages)

        except Exception as e:
            logger.critical(f"Exception at SQS get_sqs_messages", e)
            return ServerErrorResponse(f"Exception at SQS get_sqs_messages", e)
        
    async def delete_sqs_message(self, message_id, receipt_handle):
        try:
            result = self.client.delete_message(
                QueueUrl=SQS_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )

            if not result:
                logger.critical(f"Failed to delete sqs message with message id: {message_id}", result)
                return ErrorResponse(f"Failed to delete sqs message with message id: {message_id}", result)
            
            logger.info(f"Successfully deleted sqs message with message id: {message_id}")
            return SuccessResponse(f"Successfully deleted sqs message with message id: {message_id}", result)

        except Exception as e:
            logger.critical(f"Exception at SQS delete_sqs_message", e)
            return ServerErrorResponse(f"Exception at SQS delete_sqs_message", e)

    async def send_sqs_message(self, message_body: dict) -> AppResponse:
        try:
            logger.info("Sending message to SQS...")

            response = self.client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(message_body)
            )

            logger.info(f"Message sent successfully. MessageId: {response['MessageId']}")
            return SuccessResponse("Message sent successfully", response)

        except Exception as e:
            logger.critical(f"Exception at SQS send_sqs_message: {e}", exc_info=True)
            return ServerErrorResponse("Exception at SQS send_sqs_message", str(e))
