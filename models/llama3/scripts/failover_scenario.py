from models.llama3.scripts.chat_completion_program import TextGenerationHAProgram
from models.llama3.comm.realm import s
from models.datatypes import RawMessage
from models.llama3.comm.realm import Mbps
from models.llama3.comm.realm import NetworkArgs
from models.llama3.comm.realm import DeviceArgs
from models.llama3.comm.realm import TFLOPs
from models.llama3.comm.realm import ms
from models.llama3.comm.realm import Gbps
from models.llama3.comm.realm import GB
from models.llama3.comm.realm import DeviceSpec
from models.llama3.comm.realm import World

world = World()

program = TextGenerationHAProgram()

phone_spec = DeviceSpec(flops=24 * TFLOPs, mem=8 * GB, max_bandwidth=5 * Gbps, inherent_latency=10 * ms)

phone = world.device(
    deviceArgs= DeviceArgs(spec=phone_spec, client=True, name="user-phone"),
    program=program,
)

spare_phone = world.device(
    deviceArgs=DeviceArgs(spec=phone_spec, client=True, name="user-spare-phone"),
    program=program,
)

# unknown_phones = [
#     world.device(
#         deviceArgs=DeviceArgs(spec=phone_spec, client=True, name=f"user-unknown-phone-{i}"),
#         program=program
#     ) for i in range(5)
# ]

local_wifi = world.network(
    NetworkArgs(
        devices=[phone, spare_phone],
        bandwidth=4.8 * Gbps,
        latency=5 * ms,  # Maybe standard deviation?
    )
)

# bluetooth_mesh = world.network(
#     NetworkArgs(
#         devices=[phone, spare_phone] + unknown_phones,
#         bandwidth=2.0 * Mbps,
#         latency=10 * ms,
#     )
# )

world.chan("input").send([RawMessage(role="user", content="what is the recipe of mayonnaise?")])

world.after_time(5 * s).hook("phone_disconnected", lambda _: spare_phone.terminate())

if __name__ == "__main__":
    world.run()
