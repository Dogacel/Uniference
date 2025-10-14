from programs.yield_perf_program import RawMessage
from simsuite.world import World, Program
from simsuite.device import DeviceArgs
from simsuite.network import NetworkArgs
from simsuite.units import TFLOPs
from simsuite.units import ms
from simsuite.units import Gbps
from simsuite.units import GB
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


def setup_world(device_count: int, seq_len: int, output_file: str, program: Callable[..., Program]) -> World:
    world = World(debug_run=False, output_file=output_file)

    device_spec = DeviceSpec()

    devices = []
    for i in range(device_count):
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=program(
                ckpt_dir="./checkpoints/Llama-3.2-1B-Instruct/original",
                temperature=0.0,
                top_p=1.0,
                max_seq_len=seq_len,
                max_tokens=1,
            ),
        )
        devices.append(device)
        world.chan("input").subscribe(device)
        world.chan("all_gather").subscribe(device)

    world.network(
        NetworkArgs(
            devices=devices,
            network_params=[0.003223, 2 * 1.5654e-08],
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
        device.program.max_tokens = max_tokens
        device.program.yield_probability = yield_probability
        device.send("input", [RawMessage(role="user", content=prompt)], "input")

    world.run()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]
