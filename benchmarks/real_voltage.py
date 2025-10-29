import fire
import itertools
import torch

from commons import load_prompt, get_prompt_sequence_first_n, setup_world
from programs.voltage_program import VoltageProgram

from simsuite.units import Gbps, Mbps, ms


def main(
    output_file: str = "results/run_report.json",
    prompt_file: str = "checkpoints/prompt_5000.txt",
    text_lengths: list[int] = [8, 32, 256],
    speed: list[float] = [100 * Mbps, 1 * Gbps],
    latency: list[float] = [1 * ms, 10 * ms],
    repeats: int = 10,
    max_seq_len: int = 8192,
    device_count: int = 1,
    model_type: str = "voltage",
    debug_run: bool = False,
) -> int:
    prompt = load_prompt(prompt_file)

    world = setup_world(
        device_count=device_count,
        seq_len=max_seq_len,
        output_file=output_file,
        program=lambda **kwargs: VoltageProgram(**kwargs),
        batch_size=1,
        program_kwargs={"model_type": model_type},
        world_kwargs={"debug_run": debug_run},
    )

    # Generate Cartesian product
    combinations = list(itertools.product(text_lengths, speed, latency, range(repeats)))

    print(f"Going to run {len(combinations)} combinations")

    for combo in combinations:
        sequence_length, speed, latency, repeat_idx = combo

        print(
            f"Sequence length: {sequence_length}, tokens, speed: {speed}, latency: {latency}, Repeat index: {repeat_idx}"
        )

        sub_prompt = prompt[:sequence_length] # get_prompt_sequence_first_n(prompt, sequence_length)
        world.set_runtime_params(
            world.runtime_params
            | {
                "prompt_length": len(sub_prompt),
                "device_count": device_count,
                "max_seq_len": sequence_length,
                "max_tokens": 1,
                "network_bandwidth": speed,
                "network_latency": latency,
            }
        )

        world.networks[0].network_params = [latency, 1 / speed]

        client = True
        for device in world.devices:
            device.client = client
            device.spec.speed_scale = 1.30951488  # Adjust speed scale for Voltage devices
            if world.backend == "pytorch":
                device.client = torch.distributed.get_rank() == 0
                client = device.client
            if client:
                print("Device", device.name, "is client, sending input")
                device.program.input = [sub_prompt]
            client = False  # Only first device is client
            device.program.max_tokens = 1

        print("Start run...")
        world.run()

    world.destroy()

    return 0


if __name__ == "__main__":
    fire.Fire(main)
