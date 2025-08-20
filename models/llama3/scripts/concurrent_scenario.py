import fire
from models.llama3.scripts.chat_completion_program import TextGenerationHAProgram
from models.datatypes import RawMessage
from models.llama3.comm.realm import DeviceArgs
from models.llama3.comm.realm import TFLOPs
from models.llama3.comm.realm import ms
from models.llama3.comm.realm import Gbps
from models.llama3.comm.realm import GB
from models.llama3.comm.realm import DeviceSpec
from models.llama3.comm.realm import World


def run(
    device_count: int,
    prompt: str,
    **kwargs,
):
    world = World()

    device_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)

    for i in range(device_count):
        device = world.device(
            deviceArgs=DeviceArgs(spec=device_spec, client=True, name=f"phone_{i + 1}"),
            program=TextGenerationHAProgram(),
        )
        device.send("input", [RawMessage(role="user", content=prompt)])

    world.run()

if __name__ == "__main__":
    fire.Fire(run)
