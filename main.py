import json
from loguru import logger
import asyncio
from process_image import ProcessImage
from utils.logger import setup_logger
from utils.models import Models
from utils.mongodb import MongodbDatabase
from utils.pinecone import PineconeDatabase
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse
from utils.sqs import SQS

async def process_individual_message(message, process_image: ProcessImage, sqs: SQS, models: Models) -> AppResponse:
    try:
        print("message", message)
        receipt_handle = message.get('ReceiptHandle')
        message_id = message.get('MessageId')
        body = message.get('Body', "{}")

        try:
            logger.info(f"Processing message {message_id}")
            message_dict = json.loads(body)
            action = message_dict.get("action")

            # Handle "EXPERIENCE_IMAGE_UPLOADED" action
            if action == "EXPERIENCE_IMAGE_UPLOADED":
                result = await process_image.handle_request(message_dict)
            else:
                logger.exception(f"Invalid action: {action} at process_individual_message")
                return ErrorResponse(f"Invalid action: {action} at process_individual_message")

            if not result.success:
                logger.exception(f"action {action} failed at process_individual_message: {result}")
                return ErrorResponse(f"action {action} failed at process_individual_message", result)

            sqs_delete_response = await sqs.delete_sqs_message(message_id, receipt_handle)

            return sqs_delete_response

        except Exception as e:
            logger.critical(f"Exception at main process_individual_message", exc_info=True)
            return ServerErrorResponse(f"Exception at main process_individual_message", e)


    except Exception as e:
        logger.critical(f"Exception at process_individual_message", exc_info=True)

async def process_messages():
    try:
        while True:
            sqs = SQS()
            models = Models()
            pinecone_database = PineconeDatabase()
            mongodb_database = MongodbDatabase()
            process_image = ProcessImage(models, pinecone_database, mongodb_database)
            
            response = await sqs.get_sqs_messages()
            if not response.success:
                logger.exception(f"SQS message receiver failed: {response}")
                break
            
            if not isinstance(response.data, list) or len(response.data) == 0:
                logger.info(f"No more messages in sqs to process: {response}")
                break

            for message in response.data:
                await process_individual_message(message, process_image, sqs, models)

    except Exception as e:
        print("e", e)
        logger.exception(f"Exception at process_messages")

if __name__ == "__main__":
    setup_logger()
    asyncio.run(process_messages())
