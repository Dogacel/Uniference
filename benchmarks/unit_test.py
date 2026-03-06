import sys
import torch

from typing import Optional, Sequence

import argparse

from simsuite.device import DeviceArgs, DeviceSpec
from simsuite.network import NetworkArgs
from simsuite.world import World, Program, Device


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argv[1:] if argv else sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_count", type=int, default=1, help="Number of devices")
    parser.add_argument("output_file", nargs="?", default="run_report.json", help="Output file name")
    parsed_args = parser.parse_args(args)

    output_file = "results/" + parsed_args.output_file

    world = World(output_file=output_file)

    def get_device(id):
        device = world.device(
            deviceArgs=DeviceArgs(spec=DeviceSpec(), client=True, name=f"phone_{id}"),
            program=UnitTestProgram()
        )
        device.tp_group = 0
        device.pp_rank = 0
        device.pp_size = 1

        device.tp_chan().subscribe(device)
        device.pp_chan().subscribe(device)
        return device

    global device0, device1

    device0 = get_device(0)
    device1 = get_device(1)


    world.network(
        NetworkArgs(
            devices=[device0, device1],
            network_params=[0.0, 1e-6],
        )
    )

    world.set_runtime_params(
        {
            "device_count": 2,
            "max_seq_len": 1,
        }
    )

    world.run()

    world.destroy()

    return 0

import time

class UnitTestProgram(Program):
    def initialize(self, me: Device) -> None:
        self.me = me
        pass

    def run(self) -> None:
        global device0, device1

        if self.me == device0:
            # 100 ^ 3 = 1,000,000 tensor = 4MB
            device0.tp_chan().send(device0, torch.zeros((100, 100, 100)), "t_0", device0)
            time.sleep(3)
            device0.tp_chan().send(device0, torch.zeros((100, 100, 100)), "t_0", device0) 
            time.sleep(3)
            device0.tp_chan().send(device0, torch.zeros((100, 100, 100)), "t_0", device0)
            time.sleep(3)

        else:
            device1.tp_chan().receive(device1, "t_0")
            device1.tp_chan().receive(device1, "t_0")
            device1.tp_chan().receive(device1, "t_0")


        return

    def warmup(self) -> None:
        pass


if __name__ == "__main__":
    sys.exit(main())
