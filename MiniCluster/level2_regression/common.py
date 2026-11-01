"""Socket helpers: length-prefixed JSON messages (worker + coordinator dono use karte hain)."""
import json
import struct


def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf


def recv_msg(sock):
    (length,) = struct.unpack("!I", recv_exact(sock, 4))
    return json.loads(recv_exact(sock, length).decode())
