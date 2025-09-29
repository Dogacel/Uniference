from simsuite.device import Device
from simsuite.world import Program

import fire

from models.clip import clip
from PIL import Image

import os
import torch


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


class ClipProgram(Program):
    def __init__(self):
        super().__init__()

    def initialize(self, me: Device) -> None:
        def __initialize_model(
            **kwargs,
        ) -> None:
            pass

        fire.Fire(__initialize_model)

        self.me = me
        self.device = get_device()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def run(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model

        image = self.preprocess(Image.open("./models/clip/CLIP.png")).unsqueeze(0).to(self.device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
