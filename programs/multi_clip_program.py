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


class MultiClipProgram(Program):
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

        self.me.world.chan("image_features").subscribe(self.me)
        self.me.world.chan("text_features").subscribe(self.me)

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

    def warmup(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model

        for i in range(1):
            if get_pp_rank(me) == 0:
                print(f"Device {me.name} is image processor")
                image = self.preprocess(Image.open("./models/clip/CLIP.png")).unsqueeze(0).to(self.device)
                image_features = model.encode_image(image)

                pp_send(me, image_features, target_rank=1)
                text_features = pp_recv(me, source_rank=1, shape=(3, 1024), dtype=image_features.dtype, device=self.device)

            else:
                print(f"Device {me.name} is text processor")
                text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(self.device)
                text_features = model.encode_text(text)

                pp_send(me, text_features, target_rank=0)
                image_features = pp_recv(me, source_rank=0, shape=(1, 1024), dtype=text_features.dtype, device=self.device)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    def run(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model

        if get_pp_rank(me) == 0:
            print(f"Device {me.name} is image processor")
            image = self.preprocess(Image.open("./models/clip/CLIP.png")).unsqueeze(0).to(self.device)
            image_features = model.encode_image(image)

            pp_send(me, image_features, target_rank=1)
            text_features = pp_recv(me, source_rank=1, shape=(3, 1024), dtype=image_features.dtype, device=self.device)

        else:
            print(f"Device {me.name} is text processor")
            text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(self.device)
            text_features = model.encode_text(text)

            pp_send(me, text_features, target_rank=0)
            image_features = pp_recv(me, source_rank=0, shape=(1, 1024), dtype=text_features.dtype, device=self.device)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_world_size, 
    get_model_parallel_rank,
    get_pipeline_parallel_group,
    get_pipeline_parallel_ranks,
)
import torch.distributed as dist
import threading
from time import perf_counter

def background_wait(req, tensor_ref):
    """
    Waits for the request to finish in the background.
    tensor_ref is passed just to ensure the tensor isn't garbage collected 
    before the send completes.
    """
    req.wait()
    # Optional: print(f"Finished sending tensor of shape {tensor_ref.shape}")

def pp_send(me: Device, data, target_rank: int):
    if me.world.backend == "pytorch":
        group = get_pipeline_parallel_group()
        pp_ranks = get_pipeline_parallel_ranks()

        # print(f"Device {me.name} sending data to PP rank {target_rank}...")
        # print(f"SEND: my global rank={dist.get_rank()}, pp_ranks={pp_ranks}, target_rank={target_rank}, dst_global={pp_ranks[target_rank]}")
        # print(f"Sent data shape: {data.shape}, dtype={data.dtype}, device={data.device}")

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_start",
                "op": "pp_send",
                "size": data.element_size() * data.nelement(),
            }
        )

        start = perf_counter()
        # print(f"Device {me.name} is sending data to PP rank {target_rank} (global rank {pp_ranks[target_rank]})...")
        data_to_send = data.clone().cpu()
        req = dist.isend(data_to_send, dst=get_pipeline_parallel_ranks()[target_rank], group=group)

        t = threading.Thread(target=background_wait, args=(req, data_to_send))
        t.start()

        # print(f"Device {me.name} sent data to PP rank {target_rank} (global rank {pp_ranks[target_rank]}).")
        end = perf_counter()

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_end",
                "op": "pp_send",
                "duration": end - start,
            }
        )
        return

    rank = get_pp_rank(me)
    tp_rank = 0

    # print(f"Device {me.name} sending data to PP rank {target_rank}...")
    # print(f"Subscribers: {[d.name for d in me.world.chan(f'pp_{target_rank}').subscribers]}")
    target_device = me.world.chan(f"pp_{target_rank}").subscribers[tp_rank]

    me.pp_chan().send(me, data.cpu(), f"pp_send_{rank}_{target_rank}_{tp_rank}", target_device)


def pp_recv(me: Device, source_rank: int, shape, dtype=torch.bfloat16, device="cpu"):
    if me.world.backend == "pytorch":
        group = get_pipeline_parallel_group()
        pp_ranks = get_pipeline_parallel_ranks()

        data = torch.empty(shape, dtype=dtype, device="cpu")

        # print(f"Device {me.name} receiving data from PP rank {source_rank}...")
        # print(f"RECV: my global rank={dist.get_rank()}, pp_ranks={pp_ranks}, source_rank={source_rank}, src_global={pp_ranks[source_rank]}")
        # print(f"RECV: group={group}, group.size()={group.size()}")
        # print(f"Receiving data shape: {data.shape}, dtype={data.dtype}, device={data.device}")

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_start",
                "op": "pp_recv",
                "size": data.element_size() * data.nelement(),
            }
        )

        start = perf_counter()
        # print(f"Device {me.name} is receiving data from PP rank {source_rank} (global rank {pp_ranks[source_rank]})...")
        dist.recv(data, src=get_pipeline_parallel_ranks()[source_rank], group=group)
        # print(f"Device {me.name} received data from PP rank {source_rank} (global rank {pp_ranks[source_rank]}).")
        end = perf_counter()

        data_size = data.element_size() * data.nelement()
        print(f"Device {me.name} received data from PP rank {source_rank} in {end - start:.4f} seconds. Effective bandwidth: {data_size / (end - start) / (1024 ** 2):.2f} MB/s")

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_end",
                "op": "pp_recv",
                "duration": end - start,
            }
        )

        data = data.to(device)
        return data

    rank = get_pp_rank(me)
    tp_rank = 0

    start = me.state.sync_clock()
    data = me.pp_chan().receive(me, f"pp_send_{source_rank}_{rank}_{tp_rank}").to(device)
    end = me.state.sync_clock()

    data_size = data.element_size() * data.nelement()
    # print(f"Device {me.name} received data from PP rank {source_rank} in {end - start:.4f} seconds. Effective bandwidth: {data_size / (end - start) / (1024 ** 2):.2f} MB/s")

    return data

def get_pp_size(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} size: {len(get_pipeline_parallel_ranks())}")
        return len(get_pipeline_parallel_ranks())

    return me.pp_size

def get_pp_rank(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} rank: {get_pipeline_parallel_group().rank()}")
        return get_pipeline_parallel_group().rank()

    return me.pp_rank

def get_tp_size(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} TP size: {fs_init.get_model_parallel_world_size()}")
        return get_model_parallel_world_size()

    return len(me.tp_chan().subscribers)

def get_tp_rank(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} TP rank: {fs_init.get_model_parallel_rank()}")
        return get_model_parallel_rank()

    return me.tp_chan().subscribers.index(me)
