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

phone_spec = DeviceSpec()

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
        program=TextGenerationHAProgram(),
    )
    for i in range(5)
]

local_wifi = world.network(
    NetworkArgs(
        devices=[phone, spare_phone],
        network_params=[5.50000006e-04, 8.33730502e-09, 1.30408584e-08, 6.55360000e04],
    )
)

bluetooth_mesh = world.network(
    NetworkArgs(
        devices=[phone, spare_phone] + unknown_phones,
        network_params=[5.50000006e-04, 8.33730502e-09, 1.30408584e-08, 6.55360000e04],
    )
)

world.chan("input").send(0, [RawMessage(role="user", content="what is the recipe of mayonnaise?")])


def disconnect_hook():
    phone.terminate()
    world.chan("input_fallback").send(0, [RawMessage(role="user", content="what is the recipe of mayonnaise?")])


world.after_time(5 * s).hook("phone_disconnected", disconnect_hook)

if __name__ == "__main__":
    world.run()
