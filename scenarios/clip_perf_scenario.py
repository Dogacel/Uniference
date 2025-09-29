import fire
from programs.clip_program import ClipProgram
from simsuite.device import DeviceArgs
from simsuite.network import NetworkArgs
from simsuite.units import TFLOPs
from simsuite.units import ms
from simsuite.units import Gbps
from simsuite.units import GB
from simsuite.device import DeviceSpec
from simsuite.world import World


def run(
    device_count: int,
    **kwargs,
):
    world = World(**kwargs)
    world.set_runtime_params(
        {
            "device_count": device_count,
        }
    )

    device_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)
    devices = []

    for i in range(device_count):
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=ClipProgram(),
        )
        devices.append(device)

    world.network(
        NetworkArgs(
            devices=devices,
            bandwidth=5 * Gbps,
            latency=10 * ms,
        )
    )

    world.run()


if __name__ == "__main__":
    fire.Fire(run)
