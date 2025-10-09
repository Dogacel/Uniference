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

    device_spec = DeviceSpec()
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
            network_params=[5.50000006e-04, 8.33730502e-09, 1.30408584e-08, 6.55360000e04],
        )
    )

    world.run()


if __name__ == "__main__":
    fire.Fire(run)
