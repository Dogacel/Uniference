# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Any
from models.llama3.comm.realm import Realm
from models.llama3.comm.realm import Program
from time import perf_counter
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMessage
from models.llama3.generation import Llama3

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

    def runnable(self, realm: Realm) -> None:
        def program(
            ckpt_dir: str,
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_seq_len: int = 512,
            max_batch_size: int = 4,
            world_size: Optional[int] = None,
            quantization_mode: Optional[str] = None,
            disable_kv_cache: bool = False,
        ) -> None:
            model: Llama3 = Llama3.build(
                ckpt_dir=ckpt_dir,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                world_size=world_size,
                quantization_mode=quantization_mode,
                device=get_device(),
                realm=realm,
                use_kv_cache=not disable_kv_cache,
            )

            world = realm.world
            me = realm.me

            # Non client machines will be listening for cache updates
            if not me.client:
                world.chan("cache").add_listener(model.sync_kv_cache)
                world.chan("gen").add_listener(model.sync_gen_cache)
            else:
                input = world.chan("input").receive(me)

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

                    if result.finished:
                        break

                    cprint(result.text, color="yellow", end="", flush=True)
                print("\n")

            for msg in input:
                print(f"{msg.role.capitalize()}: {msg.content}\n")
                model.clean_cache()
                evaluate(model, [msg])

        fire.Fire(program)
