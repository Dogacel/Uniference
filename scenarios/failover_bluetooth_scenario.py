
from programs.chat_completion_program import TextGenerationHAProgram
from simsuite.units import s
from models.datatypes import RawMessage
from simsuite.units import Mbps
from simsuite.network import NetworkArgs
from simsuite.device import DeviceArgs
from simsuite.units import TFLOPs
from simsuite.units import ms
from simsuite.units import Gbps
from simsuite.units import GB
from simsuite.device import DeviceSpec
from simsuite.world import World

world = World()

phone_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)

phone = world.device(
    deviceArgs=DeviceArgs(spec=phone_spec, client=True, name="user-phone"),
    program=TextGenerationHAProgram(),
)

spare_phone = world.device(
    deviceArgs=DeviceArgs(spec=phone_spec, client=True, name="user-spare-phone"),
    program=TextGenerationHAProgram(),
)

unknown_phones = [
    world.device(
        deviceArgs=DeviceArgs(spec=phone_spec, client=True, name=f"user-unknown-phone-{i}"),
        program=TextGenerationHAProgram()
    ) for i in range(5)
]

local_wifi = world.network(
    NetworkArgs(
        devices=[phone, spare_phone],
        bandwidth=4.8 * Gbps,
        latency=5 * ms,  # Maybe standard deviation?
    )
)

bluetooth_mesh = world.network(
    NetworkArgs(
        devices=[phone, spare_phone] + unknown_phones,
        bandwidth=2.0 * Mbps,
        latency=10 * ms,
    )
)

world.chan("input").send(0, [RawMessage(role="user", content="what is the recipe of mayonnaise?")])

def disconnect_hook():
    phone.terminate()
    world.chan("input_fallback").send(0, [RawMessage(role="user", content="what is the recipe of mayonnaise?")])

world.after_time(5 * s).hook("phone_disconnected", disconnect_hook)

if __name__ == "__main__":
    world.run()
