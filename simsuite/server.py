from __future__ import annotations

import asyncio
import socket
import threading
import uuid

from dataclasses import dataclass
from simsuite.remote import send_msg, recv_msg
from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from simsuite.world import World


@dataclass
class StateMessage:
    clock: float
    terminated: bool
    to_send: dict | None = None
    dependency: dict | None = None


def debug_event_loop(location=""):
    """Call this to see what's happening with event loops"""
    print(f"\n=== Debug at: {location} ===")

    try:
        loop = asyncio.get_running_loop()
        print(f"✓ Running loop found: {id(loop)}")
        print(f"  Loop is running: {loop.is_running()}")
    except RuntimeError as e:
        print(f"✗ No running loop: {e}")

    try:
        loop = asyncio.get_event_loop()
        print(f"✓ Event loop exists: {id(loop)}")
        print(f"  Loop is running: {loop.is_running()}")
    except RuntimeError as e:
        print(f"✗ No event loop: {e}")

    print(f"Current thread: {threading.current_thread().name}")
    print("=" * 50)


class WebClient:
    def __init__(self, loop, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, world: World):
        self.loop = loop
        self.reader = reader
        self.writer = writer
        self.world = world

    async def initialize_async(self):
        print("Initializing remote device")
        await send_msg(self.writer, {"type": "initialize"})
        response = await recv_msg(self.reader)
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to initialize remote device: {response}")

    def initialize(self):
        future = asyncio.run_coroutine_threadsafe(self.initialize_async(), self.loop)
        return future.result()

    async def get_state_async(self):
        response = await recv_msg(self.reader)
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to get state from remote device: {response}")

        # TODO: Maybe extra args such as status will cause an error?
        return StateMessage(
            clock=response["clock"],
            terminated=response["terminated"],
            to_send=response.get("to_send", None),
            dependency=response.get("dependency", None),
        )

    def get_state(self) -> StateMessage:
        future = asyncio.run_coroutine_threadsafe(self.get_state_async(), self.loop)
        return future.result()

    async def run_async(self, warmup: bool = False):
        if warmup:
            await send_msg(
                self.writer,
                {
                    "type": "warmup",
                    "device_count": len(self.world.devices),
                    "devices": [d.name for d in self.world.devices],
                },
            )
        else:
            await send_msg(
                self.writer,
                {
                    "type": "run",
                    "device_count": len(self.world.devices),
                    "devices": [d.name for d in self.world.devices],
                },
            )

    def run(self, warmup: bool = False):
        future = asyncio.run_coroutine_threadsafe(self.run_async(warmup), self.loop)
        return future.result()

    async def run_continue_async(self, data):
        print("Sending continue to remote device")
        await send_msg(
            self.writer,
            {
                "type": "continue",
                "dependency_data": data,
            },
        )

    def run_continue(self, data):
        future = asyncio.run_coroutine_threadsafe(self.run_continue_async(data), self.loop)
        return future.result()

    async def close_async(self):
        await send_msg(self.writer, {"type": "exit"})
        self.writer.close()
        await self.writer.wait_closed()

    def close(self):
        future = asyncio.run_coroutine_threadsafe(self.close_async(), self.loop)
        return future.result()


class BackgroundServer:
    def __init__(self):
        self.server = None
        self.server_task = None
        self.loop = None
        self.CLIENTS = {}  # client_id -> (reader, writer)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Get the underlying socket
        sock = writer.get_extra_info("socket")

        # Enable TCP keepalive
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        peer = writer.get_extra_info("peername")

        # First message expected from client: {"type":"hello","client_id": "..."}
        hello = await recv_msg(reader)
        client_id = hello.get("client_id") or str(uuid.uuid4())
        self.CLIENTS[client_id] = (reader, writer)
        print(f"[+] {client_id} connected from {peer}")
        await send_msg(writer, {"type": "ping"})
        await recv_msg(reader)

        # while True:
        #     await asyncio.sleep(10)  # Keep the connection alive

    async def _run_server(self):
        # Optional TLS (uncomment and provide certs if needed)
        # ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # ssl_ctx.load_cert_chain("server.crt", "server.key")
        self.server = await asyncio.start_server(self.handle_client, "0.0.0.0", 8765)
        addr = ", ".join(str(sock.getsockname()) for sock in self.server.sockets)
        print(f"Server listening on {addr}")

        async with self.server:
            await asyncio.gather(self.server.serve_forever())

    async def wait_for_clients_async(self, expected_count: int):
        print(f"Waiting for {expected_count} clients to connect...")
        while len(self.CLIENTS) < expected_count:
            await asyncio.sleep(1)
            print(f"{len(self.CLIENTS)}/{expected_count} clients connected...")
        print(f"All {expected_count} clients connected.")

    def wait_for_clients(self, expected_count: int):
        asyncio.run(self.wait_for_clients_async(expected_count))

    def start(self):
        """Start the server in the background"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.server_task = self.loop.create_task(self._run_server())

        # Run the loop in a background thread
        import threading

        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        print("Server started in background")

    def stop(self):
        """Stop the server and cleanup"""
        if self.server:
            self.server.close()
        if self.server_task:
            self.server_task.cancel()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        print("Server stopped")
