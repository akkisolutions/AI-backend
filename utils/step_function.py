import json
from dotenv import load_dotenv  # type: ignore
import os
import boto3
from loguru import logger
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse, SuccessResponse

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
STEP_FUNCTION_ARN = os.getenv("STEP_FUNCTION_ARN")


class StepFunction:
    def __init__(self):
        try:
            logger.info("Step Functions client initializing...")
            self.client = boto3.client(
                "stepfunctions",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION,
            )
            logger.info("Step Functions client initialized successfully.")
        except Exception as e:
            logger.critical(f"Exception at StepFunction __init__: {e}")

    async def start_execution(self, payload: dict) -> AppResponse:
        try:
            logger.info("Starting a new Step Function execution...")
            response = self.client.start_execution(
                stateMachineArn=STEP_FUNCTION_ARN,
                name=f"execution-{os.urandom(8).hex()}",
                input=json.dumps(payload),
            )
            logger.info("Step Function execution started successfully.", response)

            return SuccessResponse("Step Function started successfully", response)

        except Exception as e:
            logger.critical(f"Exception at StepFunction start_execution: {e}")
            return ServerErrorResponse(f"Exception at StepFunction start_execution", e)
