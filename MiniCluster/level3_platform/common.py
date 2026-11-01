"""Socket helpers: length-prefixed JSON messages + numpy arrays (base64, pickle ke baghair)."""
import base64
import json
import struct

import numpy as np

_DTYPES = {"float64", "float32", "int64", "int32"}


def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!Q", len(data)) + data)


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed")
        got += k
    return bytes(buf)


def recv_msg(sock):
    (length,) = struct.unpack("!Q", recv_exact(sock, 8))
    return json.loads(recv_exact(sock, length).decode())


def enc(a):
    """numpy array -> JSON-safe dict."""
    a = np.ascontiguousarray(a)
    return {"b64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": str(a.dtype)}


def dec(d):
    if d["dtype"] not in _DTYPES:
        raise ValueError("dtype allowed nahi")
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"]).copy()
