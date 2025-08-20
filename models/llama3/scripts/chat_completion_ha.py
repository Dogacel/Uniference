# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from models.llama3.comm.realm import Realm
from time import perf_counter
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMessage
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
    realm = Realm(world=World(), me=Device())
    leader: Llama3 = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
        realm=realm,
        use_kv_cache=not disable_kv_cache,
    )

    follower_realm = HAReplicationRealm(role="follower")
    follower: Llama3 = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
        realm=realm,
        use_kv_cache=not disable_kv_cache,
    )

    realm.cache_chan.add_listener(follower.sync_kv_cache)
    realm.cache_chan.start_worker()

    realm.gen_chan.add_listener(follower.sync_gen_cache)
    realm.gen_chan.start_worker()

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

                realm.print_stats(start_time)

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
