from models.llama3.scripts.chat_completion_program import TextGenerationHAProgram
from models.llama3.comm.realm import s
from models.datatypes import RawMessage
from models.llama3.comm.realm import NetworkArgs
from models.llama3.comm.realm import DeviceArgs
from models.llama3.comm.realm import TFLOPs
from models.llama3.comm.realm import ms
from models.llama3.comm.realm import Gbps
from models.llama3.comm.realm import GB
from models.llama3.comm.realm import DeviceSpec
from models.llama3.comm.realm import World

world = World()


phone_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)

phone = world.device(
    deviceArgs=DeviceArgs(spec=phone_spec, client=True, name="user-phone"),
    program=TextGenerationHAProgram(),
)

spare_phone = world.device(
    deviceArgs=DeviceArgs(spec=phone_spec, client=False, name="user-spare-phone"),
    program=TextGenerationHAProgram(),
)

local_wifi = world.network(
    NetworkArgs(
        devices=[phone, spare_phone],
        bandwidth=4.8 * Gbps,
        latency=5 * ms,  # Maybe standard deviation?
    )
)

world.chan("input").send(0, [RawMessage(role="user", content="what is the recipe of mayonnaise?")])

def disconnect_hook():
    phone.terminate()
    world.chan("input_fallback").send(world.device_states[phone].clock, [RawMessage(role="user", content="what is the recipe of mayonnaise?")])

world.after_time(2 * s).hook("phone_disconnected", disconnect_hook)

if __name__ == "__main__":
    world.run()
