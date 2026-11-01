"""MiniCluster Level 2 worker: har lab PC par chalayein.
Usage: python worker.py --port 5001

Worker ke paas apna data shard hota hai. Coordinator har epoch mein model weights
bhejta hai, worker apne shard par gradient nikal kar wapas bhejta hai.
"""
import os

# Har worker ko default 1 CPU thread (taake 1 laptop par 3 workers fair compare hon).
# Asli PCs par sab cores chahiye to: set MC_THREADS=8 (Windows) / MC_THREADS=8 python worker.py
_t = os.environ.get("MC_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _t)

import argparse
import socket
import threading
import time

import numpy as np

from common import recv_msg, send_msg
from data import generate

try:
    import psutil
except ImportError:
    psutil = None

STATE = {"Xb": None, "y": None}
LOCK = threading.Lock()


def do_ping(_msg):
    out = {"type": "pong", "host": socket.gethostname()}
    if psutil:
        out["cpu"] = psutil.cpu_percent(interval=0.1)
        out["ram"] = psutil.virtual_memory().percent
    return out


def do_load(msg):
    X, y = generate(msg["start"], msg["end"], msg.get("seed", 42))
    Xb = np.hstack([X, np.ones((len(X), 1))])  # bias column
    with LOCK:
        STATE["Xb"], STATE["y"] = Xb, y
    print(f"loaded rows {msg['start']}-{msg['end']} ({len(y)} rows)")
    return {"type": "loaded", "rows": len(y)}


def do_gradient(msg):
    with LOCK:
        Xb, y = STATE["Xb"], STATE["y"]
    if Xb is None:
        return {"type": "error", "error": "data load nahi hua"}
    w = np.array(msg["w"])
    t0 = time.perf_counter()
    for _ in range(msg.get("work", 1) - 1):  # simulated heavy compute (demo ke liye)
        Xb.T @ (Xb @ w - y)
    err = Xb @ w - y
    grad = Xb.T @ err  # gradient ka sum (coordinator N se divide karega)
    sse = float(err @ err)
    return {"type": "gradient", "grad": grad.tolist(), "sse": sse,
            "rows": len(y), "time": time.perf_counter() - t0}


HANDLERS = {"ping": do_ping, "load": do_load, "gradient": do_gradient}


def handle(conn, addr):
    try:
        with conn:
            msg = recv_msg(conn)
            fn = HANDLERS.get(msg.get("type"))
            send_msg(conn, fn(msg) if fn else {"type": "error", "error": "unknown task"})
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
    srv.listen(8)
    print(f"Worker listening on {args.host}:{args.port} (threads={_t})")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
