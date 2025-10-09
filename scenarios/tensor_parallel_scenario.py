import fire
from programs.tensor_parallel_program import TensorParallelProgram
from simsuite.device import DeviceArgs
from models.datatypes import RawMessage
from simsuite.network import NetworkArgs
from simsuite.units import TFLOPs
from simsuite.units import ms
from simsuite.units import Gbps
from simsuite.units import GB
from simsuite.device import DeviceSpec
from simsuite.world import World


def run(
    device_count: int,
    prompt: str,
    **kwargs,
):
    world = World(**kwargs)
    world.set_runtime_params(
        {
            "device_count": device_count,
            "prompt_length": len(prompt),
            "max_seq_len": kwargs.get("max_seq_len", None),
            "max_tokens": kwargs.get("max_tokens", None),
            "performance_mode": world.performance_mode,
        }
    )

    device_spec = DeviceSpec()

    devices = []
    for i in range(device_count):
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=TensorParallelProgram(**kwargs),
        )
        devices.append(device)
        world.chan("input").subscribe(device)
        world.chan("all_gather").subscribe(device)

    world.network(
        NetworkArgs(
            devices=devices,
            network_params=[5.50000006e-04, 8.33730502e-09, 1.30408584e-08, 6.55360000e04],
        )
    )

    # Warmup run
    world.run(warmup=True)

    for device in devices:
        device.send("input", [RawMessage(role="user", content=prompt)], "starting_input", 0.0)

    world.run()
    world.destroy()


if __name__ == "__main__":
    fire.Fire(run)
