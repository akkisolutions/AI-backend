from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX")

print("pinecone INDEX_NAME", INDEX_NAME)

class PineconeDatabase():
    def __init__(self):
        self.pc = Pinecone(api_key="pcsk_Cg2nG_GPDdNp46REdehHiggiDvzaTVUbWA45Phk6fbiDCE4NVZBnhcsD3rkbocEbq97TT")

        existing_indexes = self.pc.list_indexes()

        existing_indexes_names = [index.name for index in  existing_indexes]

        if INDEX_NAME not in existing_indexes_names:
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(name=INDEX_NAME)
