import asyncio
import json
import struct
import torch

HEADER = struct.Struct("!I")  # 4-byte big-endian unsigned length


async def send_msg(writer: asyncio.StreamWriter, obj) -> None:
    data = json.dumps(obj, separators=(",", ":"), cls=TensorEncoder).encode("utf-8")
    writer.write(HEADER.pack(len(data)))
    writer.write(data)
    await writer.drain()


async def recv_msg(reader: asyncio.StreamReader):
    # Read 4-byte length, then the payload
    header = await reader.readexactly(HEADER.size)
    (length,) = HEADER.unpack(header)
    data = await reader.readexactly(length)
    return json.loads(data.decode("utf-8"), object_hook=tensor_decoder)


class TensorEncoder(json.JSONEncoder):
    """JSON encoder that handles tensors at any depth"""

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "data": obj.cpu().detach().tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "requires_grad": obj.requires_grad,
            }
        elif isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "shape": list(obj.shape), "dtype": str(obj.dtype)}
        return super().default(obj)


def tensor_decoder(dct):
    """JSON decoder that reconstructs tensors"""

    if "__tensor__" in dct:
        tensor = torch.tensor(dct["data"])
        if dct.get("requires_grad", False):
            tensor.requires_grad = True
        return tensor

    elif "__ndarray__" in dct:
        return np.array(dct["data"], dtype=dct["dtype"])

    return dct
