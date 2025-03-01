from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
import os
import sys
from dotenv import load_dotenv
from utils.response import SuccessResponse  # type: ignore

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
FACE_ENCODINGS_COLLECTION = os.getenv("FACE_ENCODINGS_COLLECTION")
FACE_EMBEDDINGS_COLLECTION = os.getenv("FACE_EMBEDDINGS_COLLECTION")
FACE_CLUSTER_COLLECTION = os.getenv("FACE_CLUSTER_COLLECTION")
CAPTION_SEARCH_COLLECTION = os.getenv("CAPTION_SEARCH_COLLECTION")
EXPERIENCE_COLLECTION = os.getenv("EXPERIENCE_COLLECTION")
EXPERIENCE_PARTICIPANT_COLLECTION = os.getenv("EXPERIENCE_PARTICIPANT_COLLECTION")
ALBUM_COLLECTION =  os.getenv("ALBUM_COLLECTION")
EXPERIENCE_IMAGE_COLLECTION = os.getenv("EXPERIENCE_IMAGE_COLLECTION")

INDEX_NAME = "quickstart"

class MongodbDatabase():
    def __init__(self):

        if not MONGO_URL or not DATABASE_NAME:
            logger.exception("Error: Missing MongoDB configuration. Check environment variables.")    
            sys.exit(1)

        self.client = AsyncIOMotorClient(MONGO_URL)
        self.database = self.client[DATABASE_NAME]

        self.client.admin.command("ping")

        self.face_encodings_collection = self.database[os.getenv("FACE_ENCODINGS_COLLECTION")]
        self.face_embeddings_collection = self.database[os.getenv("FACE_EMBEDDINGS_COLLECTION")]
        self.face_cluster_collection = self.database[os.getenv("FACE_CLUSTER_COLLECTION")]
        self.caption_search_collection = self.database[os.getenv("CAPTION_SEARCH_COLLECTION")]
        self.experience_collection = self.database[os.getenv("EXPERIENCE_COLLECTION")]
        self.experience_participant_collection = self.database[os.getenv("EXPERIENCE_PARTICIPANT_COLLECTION")]
        self.album_collection = self.database[os.getenv("ALBUM_COLLECTION")]
        self.experience_participant = self.database[os.getenv("EXPERIENCE_PARTICIPANT")]
        self.experience_image = self.database[os.getenv("EXPERIENCE_IMAGE_COLLECTION")]

        # logger.info("Connected to MongoDB successfully.")
