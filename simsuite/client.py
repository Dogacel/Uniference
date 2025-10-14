import fire
import json
import socket


class StatefulClient:
    def __init__(
        self,
        device_name: str = "client",
        server_host="0.0.0.0",
        server_port=25002,
    ):
        self.device_name = device_name
        self.server_host = server_host
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        self.socket.connect((self.server_host, self.server_port))
        try:
            while True:
                server_msg = self.socket.recv(1024)
                if not server_msg:
                    break
                msg = json.loads(server_msg.decode().strip())
                action = msg.get("action")
                data = msg.get("data")
                if action == "register":
                    self.socket.sendall(json.dumps({"action": "register", "data": self.device_name}).encode() + b"\n")
                elif action == "registered":
                    print(f"Registered as: {data}")
                elif action == "bye":
                    print(data)
                    break
                else:
                    print(f"Unknown action: {action}")

                # self.socket.sendall(json.dumps(msg).encode() + b"\n")
        finally:
            self.socket.close()


if __name__ == "__main__":
    fire.Fire(StatefulClient)
