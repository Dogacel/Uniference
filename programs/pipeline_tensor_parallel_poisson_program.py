from simsuite.device import Device
from simsuite.world import Program
from typing import Optional

import fire
import time
from termcolor import cprint

from models.datatypes import RawMessage
from models.llama3_tp_pp.generation import Llama3
from models.llama3_tp_pp.model import pp_broadcast, get_pp_rank, get_tp_rank, pp_recv, pp_send

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


class PipelineTensorParallelPoissonProgram(Program):
    def __init__(
        self,
        ckpt_dir: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        world_size: Optional[int] = None,
        quantization_mode: Optional[str] = None,
        disable_kv_cache: bool = False,
        max_tokens=256,
        tp_group=0,
        pp_group=0,
        **kwargs,
    ):
        super().__init__()
        self.model: Llama3
        self.ckpt_dir: str = ckpt_dir
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.max_seq_len: int = max_seq_len
        self.max_batch_size: int = max_batch_size
        self.world_size: Optional[int] = world_size
        self.quantization_mode: Optional[str] = quantization_mode
        self.disable_kv_cache: bool = disable_kv_cache
        self.max_tokens: int = max_tokens
        self.tp_group: int = tp_group
        self.pp_group: int = pp_group

    def initialize(self, me: Device) -> None:
        self.me = me

        me.tp_group = self.tp_group
        me.pp_group = self.pp_group

        self.model = Llama3.build(
            ckpt_dir=self.ckpt_dir,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            world_size=self.world_size,
            quantization_mode=self.quantization_mode,
            device=get_device(),
            me=me,
        )

    def warmup(self) -> None:
        me = self.me
        model = self.model

        me.world.chan("time").subscribe(me)

        print(f"Device {me.name} warming up...")
        print(f"[{me.name}] Generating 20 tokens for warmup...")
        result = ""

        for token_results in model.chat_completion(
            [[RawMessage(role="user", content="Count from 1 to 10.")]],
            temperature=0.0,
            top_p=1.0,
            max_gen_len=20,
        ):
            if token_results[0].finished:
                break
            result += token_results[0].text

        print(f"[{me.name}] Warmup complete. Generated text: {result}")

        model.clean_cache()

    def run(self) -> None:
        world = self.me.world
        me = self.me
        model = self.model
        max_gen_len = 1

        # Non client machines will be listening for cache updates

        def evaluate(model: Llama3, dialog: list[RawMessage], exit_early: bool = False):
            batch = dialog

            generated_token_count = 0

            # pp_rank != 0 doesn't know when to stop generation, so we just run max_gen_len steps
            # if me.pp_rank != 0:
            #     for i in range(max_gen_len):
            #         model.model.forward(None, 0)
            #     return

            for token_results in model.chat_completion(
                batch,
                temperature=self.temperature,
                top_p=self.top_p,
                max_gen_len=max_gen_len,
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
                    world.runtime_params["total_generated_tokens"] = generated_token_count
                    break
            print("\n")

        all_messages = me.world.inputs

        delays = []

        start_time = 0

        while len(all_messages) > 0:
            now = (time.time() if world.backend == "pytorch" else me.state.sync_clock()) - start_time

            now = torch.tensor([now], dtype=torch.float32, device="cpu")
            if get_pp_rank(me) == 0:
                pp_send(me, now, target_rank=1)
            else:
                now = pp_recv(me, source_rank=0, shape=(1,), dtype=torch.float32)

            if start_time == 0:
                start_time = now.item()

            print(f"{me.state.sync_clock()}: Device {me.name} synced time: {now.item():.4f} seconds.")

            # Make sure all devices get the same view of all_messages

            visible_messages = [msg for msg in all_messages if msg["timestamp"] <= now]
            all_messages = [msg for msg in all_messages if msg["timestamp"] > now]

            print(f"Total messages in queue: {len(all_messages)}. Visible messages: {len(visible_messages)}")
            print(f"Device {me.name} processing {len(visible_messages)} messages.")

            if len(visible_messages) == 0:
                time.sleep(0.1)
                continue

            # Group visible_messages into batches of max_batch_size
            batches = [
                visible_messages[i:i + self.max_batch_size]
                for i in range(0, len(visible_messages), self.max_batch_size)
            ]

            for batch in batches:
                dialogs = [[RawMessage(role="user", content=msg["content"])] for msg in batch]
                evaluate(model, dialogs)
                model.clean_cache()

            end_time = (time.time() if world.backend == "pytorch" else me.state.sync_clock()) - start_time

            for msg in visible_messages:
                delays.append(end_time - msg["timestamp"])

            time.sleep(0.1)

        print(f"Average delay: {sum(delays)/len(delays):.4f} seconds over {len(delays)} messages.")