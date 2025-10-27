# client.py
import asyncio
import os
import platform
import socket

from greenlet import greenlet
from time import perf_counter

from simsuite.device import Device
from simsuite.remote import send_msg, recv_msg

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8765"))
CLIENT_ID = os.environ.get("CLIENT_ID", f"{platform.node()}-{os.getpid()}")


class Client:
    def __init__(self, client_id: str, device: Device):
        self.client_id = client_id
        self.device = device
        device.remote_client = self

    async def send_state(self):
        state = {
            "clock": self.device.state.clock,
            "terminated": self.device.terminated,
        }
        print("Sending state to remote device: ", state)
        response = {"type": "get_state", "status": "ok", **state}
        await send_msg(self.writer, response)

    async def wait_for_continue_async(self):
        print("[2] Waiting for continue from remote device")
        cont = await recv_msg(self.reader)
        if cont.get("type") != "continue":
            raise RuntimeError(f"Failed to get continue from remote device: {cont}")
        await self.send_state()

    async def handle_commands(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Get the underlying socket
        sock = writer.get_extra_info("socket")

        # Enable TCP keepalive
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Identify to server
        await send_msg(writer, {"type": "hello", "client_id": self.client_id})

        self.writer = writer
        self.reader = reader

        # initialize -> warmup -> continue/

        async def run(warmup: bool):
            def device_run_wrapper(device: Device):
                device.state.last_run_time = perf_counter()
                device.run(warmup=warmup)
                if self.device.world.device_type == "cuda":
                    torch.cuda.synchronize()
                device.state.sync_clock()

            g = greenlet(lambda: device_run_wrapper(self.device))

            while not g.dead:
                g.switch()
                await self.wait_for_continue_async()

            self.device.terminate()
            await self.wait_for_continue_async()

        while True:
            cmd = await recv_msg(reader)
            if cmd["type"] == "ping":
                await send_msg(writer, {"type": "pong"})

            elif cmd["type"] == "initialize":
                self.device.initialize()
                await send_msg(writer, {"type": "initialize", "status": "ok"})

            elif cmd["type"] == "warmup":
                print("[client] received warmup command")
                await self.send_state()
                await run(warmup=True)
                self.device.terminated = False
                self.device.state.clock = 0.0

            elif cmd["type"] == "run":
                print("[client] received run command")
                await self.send_state()
                await run(warmup=False)
                self.device.terminated = False
                self.device.state.clock = 0.0

            elif cmd["type"] == "exit":
                print("[client] exiting as per server command")
                self.device.terminate()
                writer.close()
                await writer.wait_closed()
                break

            else:
                print(f"[client] unknown command: {cmd}")

    async def connect_with_retries(self):
        delay = 1
        while True:
            try:
                reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)  # , ssl=ssl_ctx
                self.reader_loop = asyncio.get_running_loop()

                await self.handle_commands(reader, writer)
                break
            except Exception as e:
                print(f"[client] connection error: {e}; retrying in {delay}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)  # backoff
