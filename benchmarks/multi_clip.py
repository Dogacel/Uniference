import sys

from typing import Optional, Sequence

import argparse

from programs.multi_clip_program import MultiClipProgram
from simsuite.device import DeviceArgs, DeviceSpec
from simsuite.network import NetworkArgs
from simsuite.world import World


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argv[1:] if argv else sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_count", type=int, default=2, help="Number of devices")
    parser.add_argument("output_file", nargs="?", default="run_report.json", help="Output file name")
    parsed_args = parser.parse_args(args)

    output_file = "results/" + parsed_args.output_file
    device_count = parsed_args.device_count

    world = World(output_file=output_file)

    def get_device(id):
        device = world.device(
            deviceArgs=DeviceArgs(spec=DeviceSpec(), client=True, name=f"phone_{id}"),
            program=MultiClipProgram()
        )
        device.tp_group = 0
        device.pp_rank = id
        device.pp_size = 2

        device.tp_chan().subscribe(device)
        device.pp_chan().subscribe(device)
        return device

    devices = []
    for i in range(device_count):
        devices.append(get_device(i))

    world.network(
        NetworkArgs(
            devices=devices,
            network_params=[0.0, 1e-6],
        )
    )

    world.set_runtime_params(
        {
            "device_count": 2,
            "max_seq_len": 1,
        }
    )

    world.run(warmup=True)

    world.run()

    world.destroy()

    return 0

if __name__ == "__main__":
    sys.exit(main())
