from collections import defaultdict
from datetime import datetime, timedelta
from loguru import logger # type: ignore
from utils.models import Models
from utils.mongodb import MongodbDatabase
from utils.pinecone import PineconeDatabase
from utils.response import AppResponse, ErrorResponse, ServerErrorResponse, SuccessResponse
from bson import ObjectId
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN

class FaceClassification():

    def __init__(self, mongodb: MongodbDatabase):
        self.mongodb = mongodb

    async def fetch_face_embeddings_from_mongodb(self, user_id: str) -> AppResponse:
        try:
            aggregate_result = await self.mongodb.experience_participant_collection.aggregate([
                {
                    '$match': {
                        'participants': {
                            '$elemMatch': {
                                'user_id': ObjectId(user_id)
                            }
                        }
                    }
                }, {
                    '$lookup': {
                        'from': 'face_embeddings', 
                        'localField': 'experience_id', 
                        'foreignField': 'experience_id', 
                        'as': 'face_embeddings'
                    }
                }, {
                    '$unwind': '$face_embeddings'
                }, {
                    '$group': {
                        '_id': None, 
                        'all_embeddings': {
                            '$push': '$face_embeddings'
                        }
                    }
                }
            ]).to_list()

            all_images_doc = []
            
            if aggregate_result and isinstance(aggregate_result, list) and len(aggregate_result) > 0:
                all_images_doc = aggregate_result[0].get("all_embeddings")    

            return SuccessResponse("Successfully fetched face embeddings from mongodb", all_images_doc)
        except Exception as e:
           logger.exception(f"Exception at FaceClassification fetch_face_embeddings_from_mongodb", exc_info=True)
           return ServerErrorResponse(f"Exception at FaceClassification fetch_face_embeddings_from_mongodb", e)
        

    async def fetch_cluster_document(self, user_id: str) -> AppResponse:
        try:
            exisiting_data = await self.mongodb.face_cluster_collection.find_one({"_id": ObjectId(user_id)})

            return SuccessResponse("Successfully fetched cluster dict", exisiting_data)
        except Exception as e:
            logger.exception(f"Exception at FaceClassification fetch_cluster_document", exc_info=True)
            return ServerErrorResponse(f"Exception at FaceClassification fetch_cluster_document", e)
        
    async def update_cluster_document(self, user_id: str, cluster_doc: dict) -> AppResponse:
        try:
            replace_one_response = await self.mongodb.face_cluster_collection.replace_one(
                {"_id": ObjectId(user_id), "user_id": ObjectId(user_id)},
                {
                    **cluster_doc, 
                    "user_id": ObjectId(user_id)
                },
                upsert=True
            )

            if  replace_one_response.acknowledged and replace_one_response.modified_count > 0:
                    return SuccessResponse("Successfully updated cluster docuemnt", None)
            
            return ServerErrorResponse("Failed to update cluster document")
        except Exception as e:
            logger.exception(f"Exception at FaceClassification update_cluster_document", exc_info=True)
            return ServerErrorResponse(f"Exception at FaceClassification update_cluster_document", e)

    async def cluster_embeddings(self, images_list: list, exisiting_data: dict) -> AppResponse:
        try:
            all_faces = []
            face_refs = []
            for img in images_list:
                image_id = img["_id"]
                image_url = img["image_url"]

                for face_index, face in enumerate(img["faces"]):
                    all_faces.append(np.array(face["encoding"]))
                    face_refs.append((image_id, face_index, image_url, face["bounding_box"]))
    
            if not all_faces:
                cluster_doc = {
                    "clusters": {},
                    "cluster_count": 0,
                    "total_faces": 0,
                    "updated_at": datetime.now()
                }

                return SuccessResponse("Successfully clustered images", cluster_doc)

            encodings = np.array(all_faces)
            clustering = DBSCAN(metric="euclidean", n_jobs=-1).fit(encodings)
            labels = [int(label) for label in clustering.labels_]
    
            cluster_data = defaultdict(list)
            for (img_id, _, image_url, bounding_box), label in zip(face_refs, labels):
                cluster_data[label].append({
                    "img_id": img_id,
                    "image_url": image_url,
                    "bounding_box": bounding_box
                })

            clusters = {
                str(cluster_id): sorted(members, key=lambda x: x["img_id"])
                for cluster_id, members in cluster_data.items()
            }

            if not exisiting_data:
                cluster_name = {}
            else:
                cluster_name = {key: value["name"] for key, value in exisiting_data["clusters"].items()}

            cluster_doc = {
                "clusters": {
                    str(cluster_id): { 
                        "faces": sorted(
                            members,
                            key=lambda x: x["img_id"]
                        ),
                        "name": cluster_name.get(str(cluster_id), "")
                    }
                    for cluster_id, members in cluster_data.items()
                },
                "cluster_count": len(clusters),
                "total_faces": len(all_faces),
                "updated_at": datetime.now()
            }

            return SuccessResponse("Successfully clustered images", cluster_doc)
        
        except Exception as e:
            logger.exception(f"Exception at FaceClassification cluster_embeddings", exc_info=True)
            return ServerErrorResponse(f"Exception at FaceClassification cluster_embeddings", e)
        
    async def handle_individual_request(self, request: dict)-> AppResponse:
        try:
            if not isinstance(request, dict):
                    logger.exception("request is not a dict at FaceClassification handle_individual_request", request)
                    return ErrorResponse("request is not a dict at  FaceClassification handle_individual_request", request)
            
            user_id = request.get("user_id")

            if not user_id:
                logger.exception("Not all required fields provided for processing image at FaceClassification handle_individual_request", request)
                return ErrorResponse("user_id not provided for processing image at FaceClassification handle_individual_request", request)
            
            fetch_face_embeddings_from_mongodb_result = await self.fetch_face_embeddings_from_mongodb(user_id)

            if not fetch_face_embeddings_from_mongodb_result.success:
                return fetch_face_embeddings_from_mongodb_result
            
            images__docs_list = fetch_face_embeddings_from_mongodb_result.data
            
            fetch_cluster_document_result = await self.fetch_cluster_document(user_id)

            if not fetch_cluster_document_result.success:
                return fetch_cluster_document_result
            
            exisiting_cluster_data = fetch_cluster_document_result.data
            
            cluster_embeddings_result = await self.cluster_embeddings(images__docs_list, exisiting_cluster_data)

            if not cluster_embeddings_result.success:
                return cluster_embeddings_result
            
            cluster_doc = cluster_embeddings_result.data
            print("cluster_doc", cluster_doc)

            update_cluster_document_result = await self.update_cluster_document(user_id, cluster_doc)

            if not update_cluster_document_result.success:
                return update_cluster_document_result
            
            return SuccessResponse(f"Successfully clustered all faces for the user {user_id}")

            
        except Exception as e:
               logger.exception(f"Exception at FaceClassification handle_individual_request", exc_info=True)
               return ServerErrorResponse(f"Exception at FaceClassification handle_individual_request", e)
        
    async def handle_request(self)-> AppResponse:
        try:
            time_threshold = datetime.utcnow() - timedelta(hours=24)

            face_cluster_list = await self.mongodb.face_cluster_collection.find(
                {
                    # "$or": [
                    #     {"updated_at": None}, 
                    #     {"updated_at": {"$lt": time_threshold}} 
                    # ]
                },
                projection={
                    "user_id": 1,
                }
            ).to_list(None)

            for data in face_cluster_list:
                await self.handle_individual_request({"user_id": str(data.get("user_id"))})

        except Exception as e:
               logger.exception(f"Exception at FaceClassification handle_individual_request", exc_info=True)
               return ServerErrorResponse(f"Exception at FaceClassification handle_individual_request", e)
