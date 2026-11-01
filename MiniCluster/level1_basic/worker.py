"""MiniCluster worker: har lab PC par chalayein.
Usage: python worker.py --port 5001
"""
import argparse
import json
import socket
import struct
import threading
import time


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


def compute_squares(start, end, work):
    """start..end (inclusive) ke squares. `work` > 1 ho to har number par
    extra CPU kaam hota hai taake speedup dikhe (demo ke liye)."""
    out = []
    for x in range(start, end + 1):
        val = x * x
        for _ in range(work - 1):
            val = x * x  # simulated heavy computation
        out.append(val)
    return out


def handle(conn, addr):
    try:
        with conn:
            msg = recv_msg(conn)
            if msg["type"] == "ping":
                send_msg(conn, {"type": "pong", "host": socket.gethostname()})
            elif msg["type"] == "square":
                t0 = time.perf_counter()
                res = compute_squares(msg["start"], msg["end"], msg.get("work", 1))
                elapsed = time.perf_counter() - t0
                print(f"[{addr[0]}] {msg['start']}-{msg['end']} done in {elapsed:.3f}s")
                send_msg(conn, {"type": "result", "result": res, "time": elapsed,
                                "host": socket.gethostname()})
            else:
                send_msg(conn, {"type": "error", "error": "unknown task"})
    except Exception as e:
        print("error:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(5)
    print(f"Worker listening on {args.host}:{args.port}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
