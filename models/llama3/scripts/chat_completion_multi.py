# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from time import time, perf_counter
from models.llama3.comm.realm import Device
from models.llama3.comm.realm import InMemoryChan
from models.llama3.comm.realm import Chan
from models.llama3.comm.realm import Realm
from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMediaItem, RawMessage, RawTextItem, StopReason
from models.llama3.generation import Llama3

import os
import torch

THIS_DIR = Path(__file__).parent


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


cache_chan = InMemoryChan()
gen_chan = InMemoryChan()

class TestRealm(Realm):
    def __init__(self, role):
        super().__init__()

        self.role = role

    def chan(self, tag: str) -> Chan:
        if tag == "cache":
            return cache_chan
        elif tag == "gen":
            return gen_chan
        raise ValueError(f"Unknown channel tag: {tag}")

    def me(self) -> Device:
        return Device(tag=self.role)

def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
    disable_kv_cache: bool = False,
):

    leader_realm = TestRealm(role="leader")
    leader: Llama3 = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
        realm=leader_realm,
        use_kv_cache=not disable_kv_cache,
    )

    follower_realm = TestRealm(role="follower")
    follower: Llama3 = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
        realm=follower_realm,
        use_kv_cache=not disable_kv_cache,
    )

    cache_chan.add_listener(follower.sync_kv_cache)
    cache_chan.start_worker()

    gen_chan.add_listener(follower.sync_gen_cache)
    gen_chan.start_worker()

    dialogs = [
        [RawMessage(role="user", content="what is the recipe of mayonnaise?")],
        # [
        #     RawMessage(role="system", content="Always answer with Haiku"),
        #     RawMessage(role="user", content="I am going to Paris, what should I see?"),
        # ],
        # [
        #     RawMessage(role="system", content="Always answer with emojis"),
        #     RawMessage(role="user", content="How to go from Beijing to NY?"),
        # ],
    ]


    def evaluate(model: Llama3, dialog: list[RawMessage]):
        batch = [dialog]

        start_time = perf_counter()
        generated_token_count = 0

        for token_results in model.chat_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_seq_len,
        ):
            result = token_results[0]
            generated_token_count += 1

            if model == leader and generated_token_count == 50:
                print("<Emulated Crash>\n")
                break

            if result.finished:
                end_time = perf_counter()
                print("\n")
                print(f"Time taken: {end_time - start_time:.2f} seconds")
                print(f"Total tokens generated: {generated_token_count}")
                print(f"Total tokens per second: {generated_token_count / (end_time - start_time):.2f}")

                print(f"cache_chan.total_transferred_bytes: {cache_chan.total_transferred_bytes / 1_000_000:.2f} MB")
                print(f"cache_chan.total_transferred_count: {cache_chan.total_transferred_count}")
                print(f"cache_chan bandwith used: {cache_chan.total_transferred_bytes / 1_000_000 / (end_time - start_time):.2f} MB/s")

                print(f"gen_chan.total_transferred_bytes: {gen_chan.total_transferred_bytes / 1_000:.2f} KB")
                print(f"gen_chan.total_transferred_count: {gen_chan.total_transferred_count}")
                print(f"gen_chan bandwith used: {gen_chan.total_transferred_bytes / 1_000 / (end_time - start_time):.2f} KB/s")

                cache_chan.reset_counters()
                gen_chan.reset_counters()

                break

            cprint(result.text, color="yellow", end="", flush=True)
        print("\n")

    for dialog in dialogs:
        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

            leader.clean_cache()
            follower.clean_cache()

            evaluate(leader, dialog)
            
            evaluate(follower, dialog)




def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
