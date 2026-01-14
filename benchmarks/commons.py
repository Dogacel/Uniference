from programs.yield_perf_program import RawMessage
from simsuite.world import World, Program
from simsuite.device import DeviceArgs
from simsuite.network import NetworkArgs
from simsuite.device import DeviceSpec
from typing import Callable, List


def load_prompt(prompt_file: str) -> str:
    if not prompt_file:
        return ""
    with open(prompt_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_prompt_sequence_first_n(prompt: str, n: int) -> str:
    if n <= 0 or not prompt:
        return ""
    pos = -1
    for _ in range(n):
        pos = prompt.find(",", pos + 1)
        if pos == -1:
            return prompt
    return prompt[: pos + 1]


def setup_world(
    device_count: int,
    pp_size: int,
    seq_len: int,
    output_file: str,
    program: Callable[..., Program],
    batch_size: int = 1,
    program_kwargs: dict = {},
    network_params=[0.002, 8.5e-9],
    world_kwargs: dict = {},
) -> World:
    world = World(output_file=output_file, **world_kwargs)

    device_spec = DeviceSpec(speed_scale=1.0)

    devices = []
    for i in range(device_count):
        # 0 1 2 3 4 5 6 7
        # i % (device_count // tp_size) ->
        # 0 1 0 1 0 1 0 1 (tp_size=2, device_count=8)
        # 0 0 1 1 2 2 3 3 (tp_size=4, device_count=8)
        # i // tp_size ->
        # 0 0 1 1 2 2 3 3 (tp_size=2, device_count=8)
        # 0 0 0 0 1 1 1 1 (tp_size=4, device_count=8)

        # 8 / 2 = 4 stages
        tp_size = device_count // pp_size
        pp_rank = i % pp_size
        tp_group = i % pp_size # 0 -> 0, 1 -> 1, 2 -> 0, 3 -> 1

        print(f"For {i}-th device: tp_group={tp_group}, pp_rank={pp_rank}, pp_size={pp_size}")

        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=program(
                ckpt_dir="./checkpoints/Llama-3.2-1B-Instruct/original",
                temperature=0.0,
                top_p=1.0,
                max_seq_len=seq_len,
                max_tokens=1,
                max_batch_size=batch_size,
                tp_group=tp_group,
                pp_size=pp_size,
                pp_rank=pp_rank,
            ),
        )
        device.tp_group = tp_group
        device.pp_rank = pp_rank
        device.pp_size = pp_size

        device.tp_chan().subscribe(device)
        device.pp_chan().subscribe(device)

        devices.append(device)
        world.chan("input").subscribe(device)
        world.chan("all_gather").subscribe(device)

    world.network(
        NetworkArgs(
            devices=devices,
            network_params=network_params,
        )
    )

    world.set_runtime_params(
        {
            "device_count": device_count,
            "max_seq_len": seq_len,
        }
    )

    world.run(warmup=True)
    return world


def run_once(
    prompt: str,
    max_tokens: int,
    yield_probability: float,
    world: World,
):
    world.set_runtime_params(
        world.runtime_params
        | {
            "prompt_length": len(prompt),
            "max_tokens": max_tokens,
            "yield_probability": yield_probability,
        }
    )

    for device in world.devices:
        if device.remote:
            continue

        device.program.max_tokens = max_tokens
        device.program.yield_probability = yield_probability
        device.program.input = [RawMessage(role="user", content=prompt)]

    world.run()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]
