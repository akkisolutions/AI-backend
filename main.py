import datetime
import json
from album_memory_creation import AlbumMemoryCreation
from face_classification import FaceClassification
from loguru import logger # type: ignore
import asyncio
from process_image import ProcessImage
from utils.groq import GroqApi
from utils.logger import setup_logger
from utils.models import Models
from utils.mongodb import MongodbDatabase
from utils.pinecone import PineconeDatabase
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse
from utils.sqs import SQS
from utils.step_function import StepFunction

async def process_individual_message(message, sqs: SQS, process_image: ProcessImage, album_creation: AlbumMemoryCreation) -> AppResponse:
    try:
        print("message", message)
        receipt_handle = message.get('ReceiptHandle')
        message_id = message.get('MessageId')
        body = message.get('Body', "{}")

        try:
            logger.info(f"Processing message {message_id}")
            message_dict = json.loads(body)
            action = message_dict.get("action")

            if action == "EXPERIENCE_IMAGE_UPLOADED":
                result = await process_image.handle_request(message_dict)
            elif action == "ALBUM_CREATION_REQUEST":
                result = await album_creation.handle_album_request(message_dict)
            elif action == "MEMORY_CREATION_AI":
                result = await album_creation.handle_memory_request(message_dict)
            else:
                logger.exception(f"Invalid action: {action} at process_individual_message")
                return ErrorResponse(f"Invalid action: {action} at process_individual_message")
            print("result", result.success, result.message)
            if not result.success:
                logger.exception(f"{action} action failed at process_individual_message: {result}")
                return ErrorResponse(f"{action} action failed at process_individual_message", result)

            sqs_delete_response = await sqs.delete_sqs_message(message_id, receipt_handle)

            print("sqs_delete_response", sqs_delete_response.success, sqs_delete_response.data)
            return sqs_delete_response

        except Exception as e:
            logger.exception(f"Exception at main process_individual_message")
            return ServerErrorResponse(f"Exception at main process_individual_message", e)

    except Exception as e:
        logger.exception(f"Exception at process_individual_message")

async def process_messages():
    try:
        sqs = SQS()
        step_function = StepFunction()
        models = Models()
        pinecone_database = PineconeDatabase()
        mongodb_database = MongodbDatabase()
        groq_api = GroqApi()
        process_image = ProcessImage(models, pinecone_database, mongodb_database)
        album_creation = AlbumMemoryCreation(models, pinecone_database, mongodb_database, groq_api, step_function)
        while True:
            response = await sqs.get_sqs_messages()
            if not response.success:
                logger.exception(f"SQS message receiver failed: {response}")
                break
            
            if not isinstance(response.data, list) or len(response.data) == 0:
                logger.info(f"No more messages in sqs to process: {response}")
                break

            for message in response.data:
                await process_individual_message(
                    message = message,
                    sqs = sqs,
                    process_image = process_image,
                    album_creation=album_creation
                )

    except Exception as e:
        print("e", e)
        logger.exception(f"Exception at process_messages")

async def handle_face_classification():
    try:
        mongodb_database = MongodbDatabase()
        face_classification = FaceClassification(mongodb_database)
        await face_classification.handle_request()
    except Exception as e:
        logger.exception(f"Exception at handle_face_classification")

if __name__ == "__main__":
    setup_logger()

    asyncio.run(process_messages())
    asyncio.run(handle_face_classification())
