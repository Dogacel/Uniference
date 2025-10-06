import fire
from programs.yield_perf_program import YieldPerfProgram
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
            "yield_probability": kwargs.get("yield_probability", 1.0),
        }
    )

    device_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)

    devices = []
    for i in range(device_count):
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=YieldPerfProgram(),
        )
        devices.append(device)
        world.chan("input").subscribe(device)

    world.network(
        NetworkArgs(
            devices=devices,
            bandwidth=5 * Gbps,
            latency=10 * ms,
        )
    )

    world.run(warmup=True)

    for device in world.devices:
        device.send("input", [RawMessage(role="user", content=prompt)], "input")

    world.run()


if __name__ == "__main__":
    fire.Fire(run)
