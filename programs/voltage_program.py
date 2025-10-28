from simsuite.device import Device
from simsuite.world import Program
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMessage

from models.llama3_voltage.generation import Llama3 as Llama3Voltage
from models.llama3_voltage_improv.generation import Llama3 as Llama3VoltageImprov

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


class VoltageProgram(Program):
    def __init__(
        self,
        ckpt_dir: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        world_size: Optional[int] = None,
        quantization_mode: Optional[str] = None,
        disable_kv_cache: bool = False,
        max_tokens: int = 1024,
        model_type: str = "voltage",
        **kwargs,
    ):
        super().__init__()
        self.ckpt_dir: str = ckpt_dir
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.max_seq_len: int = max_seq_len
        self.max_batch_size: int = max_batch_size
        self.world_size: Optional[int] = world_size
        self.quantization_mode: Optional[str] = quantization_mode
        self.disable_kv_cache: bool = disable_kv_cache
        self.max_tokens: int = max_tokens
        self.model_type: str = model_type
        self.initialized: bool = False

    def initialize(self, me: Device) -> None:
        self.me = me

        if "voltage" == self.model_type:
            Llama3 = Llama3Voltage
        elif "voltage_improv" == self.model_type:
            Llama3 = Llama3VoltageImprov
        else:
            raise ValueError("Checkpoint directory must contain either 'voltage' or 'voltage_improv' in its name.")

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

    def warmup(self) -> None:
        me = self.me
        model = self.model
        world = me.world

        world.chan("input").subscribe(me)
        world.chan("pre_processed_input").subscribe(me)
        world.chan("forward").subscribe(me)
        world.chan("logits").subscribe(me)

        # print(f"Device {me.name} warming up...")
        # print(f"[{me.name}] Generating 20 tokens for warmup...")
        # result = ""

        # if me.client:
        #     for token_results in model.chat_completion(
        #         [[RawMessage(role="user", content="Count from 1 to 10.")]],
        #         temperature=0.0,
        #         top_p=1.0,
        #         max_gen_len=1,
        #     ):
        #         result += token_results[0].text
        #         world.chan("pre_processed_input").broadcast(me, None, "tokens", force_send=True)
        #         break

        #     print(f"[{me.name}] Warmup complete. Generated text: {result}")
        # else:
        #     model.serve_forever()

    def run(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model

        def evaluate(model: Llama3Voltage | Llama3VoltageImprov, dialog: list[RawMessage]):
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

                world.event_logger.log_event(
                    {
                        "device": me.name,
                        "action": "generate",
                        "time": world.device_states[me].clock,
                        "token": result.text,
                    }
                )
                cprint(result.text, color="yellow", end="", flush=True)

                if result.finished or generated_token_count >= self.max_tokens:
                    # Termination signal
                    world.chan("pre_processed_input").broadcast(me, None, "tokens", source=0)
                    break
            print("\n")

        if world.backend == "pytorch":
            me.client = torch.distributed.get_rank() == 0

        if me.client:
            input = self.input
            # Convert tensor to list of string
            for msg in input:
                msg = RawMessage(role="user", content=msg)
                print(f"{msg.role.capitalize()}: {msg.content}\n")
                evaluate(model, [msg])
                model.clean_cache()
        else:
            model.serve_forever()
