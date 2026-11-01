"""MiniCluster Level 3 worker: har lab PC par chalayein.
Usage: python worker.py --port 5001 [--slow 2.0]

Worker do tarah ka kaam karta hai:
  * linreg / logreg : apne data shard ka gradient (coordinator har epoch weights bhejta hai)
  * rf              : coordinator ke diye hue tasks par Random Forest ke trees train karta hai
--slow F : is worker ko F guna slow bana deta hai (heterogeneous cluster / load balancing demo)
"""
import os

# Default 1 CPU thread per worker (ek laptop par fair comparison). Asli PCs: set MC_THREADS=8
_t = os.environ.get("MC_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _t)

import argparse
import socket
import threading
import time

import numpy as np

from common import dec, enc, recv_msg, send_msg

try:
    import psutil
except ImportError:
    psutil = None

try:  # startup par import, taake pehle RF task ka time import cost se na bhare
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError:
    RandomForestClassifier = RandomForestRegressor = None

STATE = {"mode": None, "Xb": None, "X": None, "y": None, "Y1h": None, "K": 1}
MODELS = {}
LOCK = threading.Lock()
SLOW = 1.0


def slowdown(t0):
    """Simulated slow machine: extra wait, jo reported time mein bhi shamil hota hai."""
    elapsed = time.perf_counter() - t0
    if SLOW > 1.0:
        time.sleep((SLOW - 1.0) * elapsed)
    return time.perf_counter() - t0


def do_ping(_msg):
    out = {"type": "pong", "host": socket.gethostname()}
    if psutil:
        out["cpu"] = psutil.cpu_percent(interval=0.1)
        out["ram"] = psutil.virtual_memory().percent
    return out


def do_load(msg):
    X, y = dec(msg["X"]), dec(msg["y"])
    K = msg.get("K", 1)
    with LOCK:
        MODELS.clear()
        STATE.update(mode=msg["mode"], K=K, y=y, X=None, Xb=None, Y1h=None)
        if msg["mode"] == "shard":
            STATE["Xb"] = np.hstack([X, np.ones((len(X), 1))])
            if msg.get("algo") == "logreg":
                STATE["Y1h"] = np.eye(K)[y.astype(int)]
        else:
            STATE["X"] = X
    print(f"loaded {msg['mode']}: {len(y)} rows x {X.shape[1]} features")
    return {"type": "loaded", "rows": len(y)}


def do_gradient(msg):
    with LOCK:
        Xb, y, Y1h = STATE["Xb"], STATE["y"], STATE["Y1h"]
    if Xb is None:
        return {"type": "error", "error": "shard load nahi hua"}
    algo, w, work = msg["algo"], np.array(msg["w"]), msg.get("work", 1)

    def compute():
        if algo == "linreg":
            err = Xb @ w - y
            return Xb.T @ err, float(err @ err), 0
        W = w.reshape(Xb.shape[1], -1)
        Z = Xb @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -float(np.log(P[np.arange(len(y)), y] + 1e-12).sum())
        return Xb.T @ (P - Y1h), loss, int((P.argmax(1) == y).sum())

    t0 = time.perf_counter()
    for _ in range(work - 1):  # simulated heavy compute
        compute()
    grad, loss, correct = compute()
    return {"type": "gradient", "grad": grad.ravel().tolist(), "loss": loss, "correct": correct,
            "rows": len(y), "time": slowdown(t0)}


def do_rf_train(msg):
    if RandomForestClassifier is None:
        return {"type": "error", "error": "scikit-learn install karein (pip install scikit-learn)"}
    with LOCK:
        X, y = STATE["X"], STATE["y"]
    if X is None:
        return {"type": "error", "error": "full data load nahi hua"}
    Model = RandomForestClassifier if msg["kind"] == "classification" else RandomForestRegressor
    t0 = time.perf_counter()
    rf = Model(n_estimators=msg["trees"], max_depth=msg["max_depth"] or None,
               n_jobs=msg.get("cores", 1), random_state=msg["seed"])
    rf.fit(X, y)
    with LOCK:
        MODELS[msg["task"]] = rf
    t = slowdown(t0)
    print(f"task {msg['task']} ({msg['trees']} trees) done in {t:.2f}s")
    return {"type": "trained", "task": msg["task"], "time": t}


def do_rf_predict(msg):
    X = dec(msg["X"])
    total, trees = None, 0
    t0 = time.perf_counter()
    for tid in msg["tasks"]:
        with LOCK:
            rf = MODELS.get(tid)
        if rf is None:
            return {"type": "error", "error": f"task {tid} ke trees nahi mile"}
        part = (rf.predict_proba(X) if hasattr(rf, "predict_proba") else rf.predict(X)) * rf.n_estimators
        total = part if total is None else total + part
        trees += rf.n_estimators
    return {"type": "predicted", "sum": enc(total), "trees": trees, "time": slowdown(t0)}


HANDLERS = {"ping": do_ping, "load": do_load, "gradient": do_gradient,
            "rf_train": do_rf_train, "rf_predict": do_rf_predict}


def handle(conn, addr):
    try:
        with conn:
            msg = recv_msg(conn)
            fn = HANDLERS.get(msg.get("type"))
            send_msg(conn, fn(msg) if fn else {"type": "error", "error": "unknown task"})
    except Exception as e:
        print("error:", e)


def main():
    global SLOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--slow", type=float, default=1.0)
    args = ap.parse_args()
    SLOW = max(1.0, args.slow)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(16)
    print(f"Worker listening on {args.host}:{args.port} (threads={_t}, slow={SLOW}x)")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
