from simsuite.device import Device
from simsuite.world import Program
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMessage
from models.llama3_ha.generation import Llama3

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


class TextGenerationHAProgram(Program):
    def __init__(self):
        super().__init__()
        self.model: Llama3
        self.ckpt_dir: str
        self.temperature: float
        self.top_p: float
        self.max_seq_len: int
        self.max_batch_size: int
        self.world_size: Optional[int]
        self.quantization_mode: Optional[str]
        self.disable_kv_cache: bool

    def initialize(self, me: Device) -> None:
        def __initialize_model(
            ckpt_dir: str,
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_seq_len: int = 512,
            max_batch_size: int = 4,
            world_size: Optional[int] = None,
            quantization_mode: Optional[str] = None,
            disable_kv_cache: bool = False,
            **kwargs,
        ) -> None:
            self.ckpt_dir = ckpt_dir
            self.temperature = temperature
            self.top_p = top_p
            self.max_seq_len = max_seq_len
            self.max_batch_size = max_batch_size
            self.world_size = world_size
            self.quantization_mode = quantization_mode
            self.disable_kv_cache = disable_kv_cache

        fire.Fire(__initialize_model)

        self.me = me
        self.model = Llama3.build(
            ckpt_dir=self.ckpt_dir,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            world_size=self.world_size,
            quantization_mode=self.quantization_mode,
            device=get_device(),
            me=me,
            use_kv_cache=not self.disable_kv_cache,
        )

    def run(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model

        world.chan("cache").subscribe(me)
        world.chan("gen").subscribe(me)
        world.chan("input").subscribe(me)
        world.chan("input_fallback").subscribe(me)

        # Non client machines will be listening for cache updates
        input: Optional[list[RawMessage]] = None
        if not me.client:
            world.chan("cache").add_listener(model.sync_kv_cache)
            world.chan("gen").add_listener(model.sync_gen_cache)
            input = world.chan("input_fallback").receive(me)
        else:
            input = world.chan("input").receive(me)

        def evaluate(model: Llama3, dialog: list[RawMessage]):
            batch = [dialog]

            generated_token_count = 0

            for token_results in model.chat_completion(
                batch,
                temperature=self.temperature,
                top_p=self.top_p,
                max_gen_len=self.max_seq_len,
            ):
                result = token_results[0]
                generated_token_count += 1

                if result.finished:
                    break

                world.event_logger.log_event(
                    {
                        "device": me.name,
                        "action": "generate",
                        "time": world.device_states[me].clock,
                        "token": result.text,
                    }
                )
                cprint(result.text, color="yellow", end="", flush=True)
            print("\n")

        if input is not None:
            for msg in input:
                print(f"{msg.role.capitalize()}: {msg.content}\n")
                evaluate(model, [msg])
                model.clean_cache()
