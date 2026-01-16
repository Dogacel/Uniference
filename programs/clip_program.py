from simsuite.device import Device
from simsuite.world import Program

import fire
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

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
        self.model, self.preprocess = clip.load("RN50x64", device=self.device)

    def run(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model

        device = get_device()
        if device is not Device:
            device = torch.device(device)

        if not torch.distributed.is_initialized():
            if device.type == "cuda":
                backend = os.environ.get("DIST_BACKEND", "nccl")
                torch.distributed.init_process_group(backend)
            else:
                torch.distributed.init_process_group("gloo")

        if not model_parallel_is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            pp_size = int(os.environ.get("PP_SIZE", 1))
            initialize_model_parallel(
                model_parallel_size_=world_size // pp_size,
                pipeline_length=pp_size,
            )

        image = self.preprocess(Image.open("./models/clip/CLIP.png")).unsqueeze(0).to(self.device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
