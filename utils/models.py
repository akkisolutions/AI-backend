import asyncio
import gc
import torch
import open_clip
from loguru import logger # type: ignore

class Models:
    def __init__(self):
        logger.info("Models initilization started.")
        logger.info("starting open_clip_model and preprocess initilization.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.open_clip_model, self.preprocess, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
        self.open_clip_model.to(self.device)
        logger.info("Starting tokenizer initilization.")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("Models initilization Completed successfully.")

    def cleanup(self, signum=None, frame=None):
        """Cleanup function to free memory before termination"""
        logger.info("Received termination signal. Cleaning up model...")
        print("Received termination signal. Cleaning up model...")
        
        del self.open_clip_model
        del self.preprocess
        del self.tokenizer

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Cleanup complete. Exiting now.")
        print("Cleanup complete. Exiting now.")

    def __del__(self):
        """Destructor to ensure cleanup when the instance is deleted"""
        logger.info("Deleting Models instance. Performing cleanup...")
        asyncio.run(self.cleanup())