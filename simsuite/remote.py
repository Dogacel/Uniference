import asyncio
import json
import struct

HEADER = struct.Struct("!I")  # 4-byte big-endian unsigned length


async def send_msg(writer: asyncio.StreamWriter, obj) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    writer.write(HEADER.pack(len(data)))
    writer.write(data)
    await writer.drain()


async def recv_msg(reader: asyncio.StreamReader):
    # Read 4-byte length, then the payload
    header = await reader.readexactly(HEADER.size)
    (length,) = HEADER.unpack(header)
    data = await reader.readexactly(length)
    return json.loads(data.decode("utf-8"))
