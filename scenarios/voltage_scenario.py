import fire
from simsuite.device import DeviceArgs
from programs.voltage_program import VoltageProgram
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
    world = World()

    device_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)

    devices = []
    for i in range(device_count):
        is_client = i == 0
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=is_client, name=f"phone_{i + 1}"),
            program=VoltageProgram(),
        )

        if is_client:
            device.send("input", [RawMessage(role="user", content=prompt)])

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
