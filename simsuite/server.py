import fire
import socket
import threading
import json


class StatefulServer:
    def __init__(self, host="0.0.0.0", port=25002):
        self.host = host
        self.port = port
        self.clients = {}
        self.lock = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

    def handle_client(self, client_socket, address):
        try:
            client_socket.sendall(b'{"action": "register", "data": "Enter your name"}\n')
            name_msg = client_socket.recv(1024).decode().strip()
            try:
                name_data = json.loads(name_msg)
                name = name_data.get("data", "Unknown")
            except Exception:
                name = "Unknown"
            with self.lock:
                self.clients[address] = name
            client_socket.sendall(json.dumps({"action": "registered", "data": name}).encode() + b"\n")
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                try:
                    msg = json.loads(data.decode().strip())
                    action = msg.get("action")
                    payload = msg.get("data")
                except Exception as e:
                    response = {"action": "error", "data": str(e)}
                client_socket.sendall(json.dumps(response).encode() + b"\n")
        finally:
            with self.lock:
                self.clients.pop(address, None)
            client_socket.close()
            print(f"Connection closed: {address}")

    def start(self):
        try:
            while True:
                client_socket, address = self.server_socket.accept()
                print(f"Connection from {address}")
                threading.Thread(target=self.handle_client, args=(client_socket, address), daemon=True).start()
        except KeyboardInterrupt:
            print("Server shutting down.")
        finally:
            self.server_socket.close()


if __name__ == "__main__":
    fire.Fire(StatefulServer)
