from typing import Literal, Optional
from loguru import logger # type: ignore
from utils.groq import GroqApi
from utils.models import Models
from utils.mongodb import MongodbDatabase
from utils.pinecone import PineconeDatabase
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse, SuccessResponse
import torch
from bson import ObjectId
import datetime


class AlbumCreation():
     
     def __init__(self, models: Models, pinecone: PineconeDatabase, mongodb: MongodbDatabase, groqApi: GroqApi):
        self.models = models
        self.pinecone = pinecone
        self.mongodb = mongodb
        self.groqApi = groqApi

     async def generate_text_embedding(self, text: str) -> AppResponse:
          try:
               with torch.no_grad():
                    text_tokens = self.models.tokenizer([text]).to(self.models.device)
                    text_features = self.models.open_clip_model.encode_text(text_tokens)

                    embeddings = text_features.cpu().numpy().flatten()
          
                    return SuccessResponse("Generated text embedding successfully", embeddings)
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation generate_text_embedding", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation generate_text_embedding", e)
     
     async def get_names_from_query(self, known_names, search_prompt, names_dict) -> AppResponse:
          try:
               names_response = self.groqApi.identify_names_from_prompt(known_names, search_prompt)

               selected_names_list = names_response.data.get("user_names")

               cluster_keys = [names_dict[name] for name in selected_names_list if name in names_dict]
          
               return SuccessResponse("Successfully fetched names from query", {
                    "selected_names_list": selected_names_list,
                    "cluster_keys": cluster_keys
               })
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation get_names_from_query", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation get_names_from_query", e)
          
     async def search_images(self, text_prompt, top_k=5, experience_ids = None, image_ids = None) -> AppResponse:
          try:
               text_embedding = self.generate_text_embedding(text_prompt)

               filter_condition = {}
               
               if image_ids:
                    filter_condition["img_id"] = {"$in": image_ids}

               if experience_ids:
                    filter_condition["experience_id"] = {"$in": experience_ids}

               results = self.pinecone.index.query(
                    vector=text_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_condition
               )

               if results and "matches" in results:
                    matched_images = []
                    for match in results["matches"]:
                         print(f"Image metadata: {match['metadata']} | Score: {match['score']:.4f}")
                         matched_images.append(match.get('metadata', None).get("image_url", None))

                    return SuccessResponse("Successfully fetched images from pinecone", matched_images)

               else:
                    return SuccessResponse("No images matched", None)
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation search_images", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation search_images", e)
          
     async def search_cluster_names(self, user_id) -> AppResponse:
          try:
               response = await self.mongodb.face_cluster_collection.aggregate([
                    {
                         '$match': {
                              'user_id': ObjectId(user_id)
                         }
                    }, {
                         '$project': {
                              '_id': 0, 
                              'clusters': {
                                   '$map': {
                                        'input': {
                                        '$objectToArray': '$clusters'
                                        }, 
                                        'as': 'cluster', 
                                        'in': {
                                        'k': '$$cluster.v.name',
                                        'v': '$$cluster.k'
                                        }
                                   }
                              }
                         }
                    }, {
                         '$replaceRoot': {
                              'newRoot': {
                                   '$arrayToObject': '$clusters'
                              }
                         }
                    }
                    ]).to_list()
               
               names_dict = {}
               names_list = []

               if isinstance(response, list) and len(response) >0:
                    names_dict = response[0]
                    names_list = names_dict.keys()                   

               return SuccessResponse("Successfully fetched names from mongodb", {
                    "names_dict": names_dict, 
                    "names_list": names_list
               })
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation search_cluster_names", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation search_cluster_names", e)
          
     async def fetch_applicable_experience_ids(self, user_id: str) -> AppResponse:
          try:
               response = await self.mongodb.experience_participant.aggregate([
                    {
                         '$match': {
                              'participants': {
                                   '$elemMatch': {
                                   'user_id': ObjectId(user_id), 
                                   'status': {
                                        '$in': [
                                             'approved', 'accepted'
                                        ]
                                   }
                                   }
                              }
                         }
                    }, {
                         '$group': {
                              '_id': None, 
                              'experience_ids': {
                                   '$push': '$experience_id'
                              }
                         }
                    }
               ]).to_list()

               experience_ids = []
               
               if isinstance(response, list) and len(response)> 0:
                    experience_ids = response[0].get("experience_ids")   
                    experience_ids = [str(experience_id) for experience_id in experience_ids]                 

               return SuccessResponse("Successfully fetched experience_ids from mongodb", experience_ids)
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation fetch_applicable_experience_ids", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation fetch_applicable_experience_ids", e)

     async def fetch_applicable_image_ids(self, user_id: str, cluster_keys: list[str]) -> AppResponse:
          try:
               response = await self.mongodb.face_cluster_collection.aggregate([
                    {
                         '$match': {
                              'user_id': ObjectId(user_id)
                         }
                    }, {
                         '$project': {
                              'images': {
                                   '$objectToArray': '$clusters'
                              }
                         }
                    }, {
                         '$unwind': {
                              'path': '$images', 
                              'preserveNullAndEmptyArrays': False
                         }
                    }, {
                         '$match': {
                              'images.k': {
                                   '$in': cluster_keys
                              }
                         }
                    }, {
                         '$unwind': {
                              'path': '$images.v.faces', 
                              'preserveNullAndEmptyArrays': False
                         }
                    }, {
                         '$group': {
                              '_id': None, 
                              'image_ids': {
                                   '$addToSet': '$images.v.faces.img_id'
                              }
                         }
                    }
               ]).to_list()

               if isinstance(response, list) and len(response)> 0:
                    image_ids = response[0].get("image_ids")
                    image_ids = [str(image_id) for image_id in image_ids]                  

               return SuccessResponse("Successfully fetched image_ids from mongodb", image_ids)
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation fetch_applicable_image_ids", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation fetch_applicable_image_ids", e)
          
     async def update_status_of_album_in_mongodb(self, album_id: str, images: list[str], status: Literal["created", "failure"], failure_reason: Optional[str] = None) -> AppResponse:
          try:
               update_one_response = await self.mongodb.album_collection.update_one(
                    {"_id": ObjectId(album_id)},
                    {
                         "$set": {
                              "images": images,
                              "status": status,
                              "failure_reason": failure_reason,
                              "updated_at": datetime.utcnow(),
                         }
                    }
               )
                    
               if  update_one_response.acknowledged and update_one_response.modified_count > 0:
                    return SuccessResponse("Successfully updated album status", None)
               
               return ServerErrorResponse("Failed to update album status")
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation update_status_of_album_in_mongodb", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation update_status_of_album_in_mongodb", e)
          
     async def fetch_album_from_mongodb(self, album_id: str) -> AppResponse:
          try:
               find_one_response = await self.mongodb.album_collection.find_one(
                    {"_id": ObjectId(album_id)},
               )
                    
               if find_one_response:
                    return SuccessResponse("Successfully fetched album", find_one_response)
               
               return ServerErrorResponse("Failed to fetch album")
          except Exception as e:
               logger.exception(f"Exception at AlbumCreation fetch_album_from_mongodb", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation fetch_album_from_mongodb", e)
          
     async def handle_request(self, request: dict)-> AppResponse:
          try:
               if not isinstance(request, dict):
                    logger.exception("request is not a dict at AlbumCreation handle_request", request)
                    return ErrorResponse("request is not a dict at  AlbumCreation handle_request", request)
            
               album_id = request.get("album_id")

               if not album_id:
                    logger.exception("Not all required fields provided for processing image at AlbumCreation handle_request", request)
                    return ErrorResponse("album_id not provided for processing image at AlbumCreation handle_request", request)
               
               fetch_album_from_mongodb = await self.fetch_album_from_mongodb(album_id)

               if not fetch_album_from_mongodb.success:
                    return fetch_album_from_mongodb
               
               prompt = fetch_album_from_mongodb.data.get("prompt")
               user_id = fetch_album_from_mongodb.data.get("user_id")
               
               search_cluster_names_result = await self.search_cluster_names(user_id)

               if not search_cluster_names_result.success:
                    return search_cluster_names_result
               
               names_dict = search_cluster_names_result.data.get("names_dict")
               names_list = search_cluster_names_result.data.get("names_list")

               get_names_from_query_result = await self.get_names_from_query(names_list, prompt, names_dict)

               if not get_names_from_query_result.success:
                    return get_names_from_query_result

               cluster_keys = get_names_from_query_result.data.get("cluster_keys")

               if isinstance(cluster_keys, list) and len(cluster_keys)> 0:
                    fetch_applicable_image_ids_result = await self.fetch_applicable_image_ids(user_id, cluster_keys)

                    if not fetch_applicable_image_ids_result.success:
                         return fetch_applicable_image_ids_result

                    image_ids = fetch_applicable_image_ids_result.data
               else:
                    fetch_applicable_experience_ids_result = await self.fetch_applicable_experience_ids(user_id)

                    if not fetch_applicable_experience_ids_result.success:
                         return fetch_applicable_experience_ids_result
                    
                    experience_ids = fetch_applicable_experience_ids_result.data

               search_images_result = await self.search_images(prompt, experience_ids=experience_ids, image_ids=image_ids)

               if not search_images_result.success:
                    return search_images_result
               
               update_status_of_album_in_mongodb_result = await self.update_status_of_album_in_mongodb(
                    album_id=album_id, 
                    images=search_images_result.data if search_images_result.data else [],
                    status="created" if search_images_result.data else "failure",
                    failure_reason=search_images_result.message
               )

               if not update_status_of_album_in_mongodb_result.success:
                    return update_status_of_album_in_mongodb_result

               return SuccessResponse("Image processed again sucessfully", None)
          except Exception as e:

               logger.exception(f"Exception at AlbumCreation handle_request", exc_info=True)
               return ServerErrorResponse(f"Exception at AlbumCreation handle_request", e)
          
# {"action": "ALBUM_CREATION_REQUEST", "prompt": "happy images", "album_id": "67be1a5e16bca4c9cf243eda", "timestamp": "2025-02-25T19:30:38.837116"}