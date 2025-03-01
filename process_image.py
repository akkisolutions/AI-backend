from loguru import logger # type: ignore
from utils.models import Models
from utils.mongodb import MongodbDatabase
from utils.pinecone import PineconeDatabase
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse, SuccessResponse
from PIL import Image
import cv2
import torch
import requests
from io import BytesIO
import numpy as np
import face_recognition
from bson import ObjectId
import datetime

video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm")

class ProcessImage():

    def __init__(self, models: Models, pinecone: PineconeDatabase, mongodb: MongodbDatabase):
        self.models = models
        self.pinecone = pinecone
        self.mongodb = mongodb

    async def extract_image_embedding_from_opencv(self, rgb_image) -> AppResponse:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

            image = self.models.preprocess(pil_image).unsqueeze(0).to(self.models.device)

            with torch.no_grad():
                image_features = self.models.open_clip_model.encode_image(image)

            return SuccessResponse("Successfully generated embeddings from image", image_features.cpu().numpy().flatten())
        
        except Exception as e:
            logger.critical(f"Exception at ProcessImage extract_image_embedding_from_opencv")
            return ServerErrorResponse(f"Exception at ProcessImage extract_image_embedding_from_opencv", e)        

    async def handle_request(self, request: dict)-> AppResponse:
        try:
            print("request", request)
            if not isinstance(request, dict):
                logger.exception("request is not a dict at ProcessImage handle_request", request)
                return ErrorResponse("request is not a dict at  ProcessImage handle_request", request)
            
            image_url = request.get("image_url")
            experience_id = request.get("experience_id")
            uploader_id = request.get("uploader_id")
            img_id = request.get("img_id")
            is_video = image_url.lower().endswith(video_extensions)

            if not image_url or not experience_id or not uploader_id or not img_id or is_video:
                logger.exception("Not all required fields provided for processing image at ProcessImage handle_request", request)
                return ErrorResponse("Not all required fields provided for processing image at ProcessImage handle_request", request)

            response = requests.get(image_url)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_array = np.array(img)
            rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            image_embedding = await self.extract_image_embedding_from_opencv(rgb)

            if not image_embedding.success:
                logger.exception("Faield to generate image embeddings at ProcessImage handle_request", request)
                return ErrorResponse("Faield to generate image embeddings at ProcessImage handle_request", request)

            self.pinecone.index.upsert([(img_id, image_embedding.data.tolist(), {"experience_id": experience_id, "img_id": img_id, "image_url": image_url})])

            boxes = face_recognition.face_locations(rgb, model="cnn")
            encodings = face_recognition.face_encodings(rgb, boxes)

            update = {
                "experience_id": ObjectId(experience_id),
                "uploader_id": ObjectId(uploader_id),
                "img_id": ObjectId(img_id),
                "image_url": image_url,
                "faces":[
                    {
                        "bounding_box": box,
                        "encoding": enc.tolist(),
                        "face_id": None
                    }
                    for box, enc in zip(boxes, encodings)
                ],
                "created_at": datetime.datetime.now() 
            }

            update_one_response = await self.mongodb.face_embeddings_collection.update_one({"_id": ObjectId(img_id)}, {"$set": update}, upsert=True)

            if not update_one_response.acknowledged or not update_one_response.modified_count > 0:
                return ErrorResponse("Faield to update image embeddings to mongodb", update_one_response)
                
            return SuccessResponse("Image processed again sucessfully", None)
        except Exception as e:

            logger.exception(f"Exception at ProcessImage handle_request", exc_info=True)
            return ServerErrorResponse(f"Exception at ProcessImage handle_request", e)
        
    