import torch
import open_clip

class Models:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.open_clip_model, self.preprocess, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
        self.open_clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
