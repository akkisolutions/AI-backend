import asyncio


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.mongodb import MongodbDatabase
from utils.sqs import SQS

async def upload_all_images_to_sqs():
    sqs = SQS()
    mongodb_database = MongodbDatabase()

    images = await mongodb_database.experience_image.aggregate([
        {
            '$unwind': {
                'path': '$uploaders'
            }
        }, {
            '$unwind': {
                'path': '$uploaders.images'
            }
        }, {
            '$project': {
                '_id': 0, 
                'image_url': '$uploaders.images.key', 
                'action': {
                    '$literal': 'EXPERIENCE_IMAGE_UPLOADED'
                }, 
                'uploader_id': {
                    '$toString': '$uploaders.uploader_id'
                }, 
                'experience_id': {
                    '$toString': '$experience_id'
                }, 
                'img_id': {
                    '$toString': '$uploaders.images.img_id'
                }, 
                'timestamp': {
                    '$dateToString': {
                        'format': '%Y-%m-%dT%H:%M:%S.%L', 
                        'date': '$created_at'
                    }
                }
            }
        }
    ]).to_list()

    print("images", len(images), images)

    for image in images:
        response = await sqs.send_sqs_message(image)
        print("response status", response.success)

if __name__ == "__main__":
    asyncio.run(upload_all_images_to_sqs())
