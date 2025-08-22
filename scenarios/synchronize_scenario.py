import fire
from models.datatypes import RawMessage
from simsuite.device import DeviceArgs
from simsuite.network import NetworkArgs
from simsuite.units import TFLOPs
from simsuite.units import ms
from simsuite.units import Gbps
from simsuite.units import GB
from simsuite.device import DeviceSpec
from simsuite.world import World
from programs.ping_pong_program import PingPongProgram


def run(
    device_count: int,
    prompt: str,
    **kwargs,
):
    world = World()

    device_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)
    devices = []

    for i in range(device_count):
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=PingPongProgram(),
        )
        device.send("input", [RawMessage(role="user", content=prompt)])
        devices.append(device)

    world.network(
        NetworkArgs(
            devices=devices,
            bandwidth=5 * Gbps,
            latency=100 * ms,
        )
    )

    world.run()


if __name__ == "__main__":
    fire.Fire(run)
