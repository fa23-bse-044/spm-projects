"""MiniCluster Level 2 coordinator (FastAPI): distributed Linear Regression.
Usage: uvicorn coordinator:app --host 0.0.0.0 --port 8000

Har epoch:
  1. current weights sab workers ko bhejo
  2. har worker apne shard ka gradient wapas bheje
  3. coordinator gradients ka sum kar ke (aggregation) weights update kare
Agar koi worker mar jaye to data shards bache hue workers mein dobara baant diye jate hain
aur wahi epoch dobara chalta hai (fault tolerance).
"""
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common import recv_msg, send_msg
from data import D, FEATURES, TRUE_B, TRUE_W, generate

BASE = Path(__file__).parent
WORKERS = json.loads((BASE / "workers.json").read_text())
SEED = 42

app = FastAPI(title="MiniCluster Level 2")
JOB = None
JOB_LOCK = threading.Lock()


class TrainRequest(BaseModel):
    n: int = Field(300_000, ge=1000, le=2_000_000)
    workers: int = Field(3, ge=1)
    epochs: int = Field(60, ge=1, le=2000)
    lr: float = Field(0.15, gt=0, le=1.0)
    work: int = Field(10, ge=1, le=2000)
    slow_ms: int = Field(0, ge=0, le=5000)


# ---------------------------------------------------------------- helpers
def call_worker(w, msg, timeout=60):
    with socket.create_connection((w["host"], w["port"]), timeout=timeout) as s:
        s.settimeout(timeout)
        send_msg(s, msg)
        return recv_msg(s)


def parallel(workers, make_msg, timeout=60):
    """Sab workers ko ek saath message bhejo. Returns (ok, failed)."""
    def f(w):
        try:
            r = call_worker(w, make_msg(w), timeout)
            if r.get("type") == "error":
                return w, None, r.get("error", "error")
            return w, r, None
        except Exception as e:
            return w, None, str(e) or type(e).__name__

    if not workers:
        return [], []
    with ThreadPoolExecutor(len(workers)) as ex:
        out = list(ex.map(f, workers))
    return [(w, r) for w, r, e in out if e is None], [(w, e) for w, r, e in out if e is not None]


def shard_ranges(n, k):
    size, rem = divmod(n, k)
    ranges, start = [], 0
    for i in range(k):
        end = start + size + (1 if i < rem else 0)
        ranges.append((start, end))
        start = end
    return ranges


def event(job, text):
    job["events"].append({"t": round(time.perf_counter() - job["t0"], 2), "msg": text})


def mark_offline(job, failed):
    for w, err in failed:
        job["workers"][w["name"]].update(state="offline", rows=0)
        event(job, f"{w['name']} OFFLINE ({err})")


def distribute(job, alive, n):
    """Alive workers mein data shards load karo. Koi fail ho to hata kar dobara baanto."""
    while True:
        if not alive:
            raise RuntimeError("Koi worker available nahi")
        ranges = dict(zip([w["name"] for w in alive], shard_ranges(n, len(alive))))
        ok, failed = parallel(alive, lambda w: {"type": "load", "start": ranges[w["name"]][0],
                                                "end": ranges[w["name"]][1], "seed": SEED})
        for w, r in ok:
            job["workers"][w["name"]].update(state="online", rows=r["rows"])
        if not failed:
            return alive
        mark_offline(job, failed)
        dead = {w["name"] for w, _ in failed}
        alive = [w for w in alive if w["name"] not in dead]
        event(job, f"Data {len(alive)} workers mein dobara baanta ja raha hai")


# ---------------------------------------------------------------- training
def train(job, req):
    try:
        chosen = WORKERS[: req.workers]
        job["workers"] = {w["name"]: {"state": "pending", "rows": 0, "compute_ms": 0} for w in chosen}
        ok, failed = parallel(chosen, lambda w: {"type": "ping"}, timeout=2)
        mark_offline(job, failed)
        alive = [w for w, _ in ok]

        t_load = time.perf_counter()
        alive = distribute(job, alive, req.n)
        job["load_time"] = time.perf_counter() - t_load
        event(job, f"Data load: {', '.join(w['name'] for w in alive)}")

        w_vec = np.zeros(D + 1)
        epoch_wall, epoch_compute = [], []
        t_train = time.perf_counter()
        epoch = 0
        while epoch < req.epochs:
            if job["stop"]:
                job["state"] = "stopped"
                return
            t_ep = time.perf_counter()
            ok, failed = parallel(alive, lambda w: {"type": "gradient", "w": w_vec.tolist(), "work": req.work})
            if failed:  # fault tolerance: re-shard aur isi epoch ko dobara chalao
                mark_offline(job, failed)
                dead = {w["name"] for w, _ in failed}
                alive = [w for w in alive if w["name"] not in dead]
                event(job, f"Epoch {epoch + 1} dobara chalega, task reassignment")
                alive = distribute(job, alive, req.n)
                event(job, "Re-shard done: " + ", ".join(
                    f"{w['name']}={job['workers'][w['name']]['rows']:,} rows" for w in alive))
                continue
            grad = sum(np.array(r["grad"]) for _, r in ok)
            sse = sum(r["sse"] for _, r in ok)
            rows = sum(r["rows"] for _, r in ok)
            w_vec -= req.lr * grad / rows
            for w, r in ok:
                job["workers"][w["name"]]["compute_ms"] = round(r["time"] * 1000, 1)
            epoch_compute.append(max(r["time"] for _, r in ok))
            epoch += 1
            job["losses"].append(sse / rows)
            job["epoch"] = epoch
            if req.slow_ms:
                time.sleep(req.slow_ms / 1000)
            epoch_wall.append(time.perf_counter() - t_ep - req.slow_ms / 1000)
        train_time = time.perf_counter() - t_train - req.slow_ms / 1000 * req.epochs

        # reference: poore data par closed-form (least squares) solution
        X, y = generate(0, req.n, SEED)
        Xb = np.hstack([X, np.ones((req.n, 1))])
        w_ref = np.linalg.lstsq(Xb, y, rcond=None)[0]
        sse_final = float(np.sum((Xb @ w_vec - y) ** 2))
        sse_ref = float(np.sum((Xb @ w_ref - y) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        job["result"] = {
            "features": FEATURES + ["bias"],
            "learned": w_vec.tolist(),
            "true": TRUE_W.tolist() + [TRUE_B],
            "closed_form": w_ref.tolist(),
            "max_diff": float(np.max(np.abs(w_vec - w_ref))),
            "r2": 1 - sse_final / sst,
            "final_mse": sse_final / req.n,
            "optimal_mse": sse_ref / req.n,
            "train_time": train_time,
            "load_time": job["load_time"],
            "avg_epoch_ms": float(np.mean(epoch_wall)) * 1000,
            "avg_compute_ms": float(np.mean(epoch_compute)) * 1000,
            "workers_used": len(alive),
            "epochs": req.epochs,
        }
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)
        event(job, f"ERROR: {e}")


# ---------------------------------------------------------------- API
@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/status")
def status():
    def ping(w):
        try:
            r = call_worker(w, {"type": "ping"}, timeout=2)
            return {**w, "online": True, "cpu": r.get("cpu"), "ram": r.get("ram")}
        except Exception:
            return {**w, "online": False}

    with ThreadPoolExecutor(len(WORKERS)) as ex:
        return list(ex.map(ping, WORKERS))


@app.post("/api/train")
def start_train(req: TrainRequest):
    global JOB
    if req.workers > len(WORKERS):
        raise HTTPException(400, f"Sirf {len(WORKERS)} workers configured hain")
    with JOB_LOCK:
        if JOB and JOB["state"] == "running":
            raise HTTPException(409, "Ek training pehle se chal rahi hai")
        JOB = {"state": "running", "epoch": 0, "epochs": req.epochs, "n": req.n, "requested": req.workers,
               "losses": [], "events": [], "workers": {}, "result": None, "stop": False,
               "t0": time.perf_counter()}
        threading.Thread(target=train, args=(JOB, req), daemon=True).start()
    return {"ok": True}


@app.get("/api/job")
def get_job():
    if JOB is None:
        return {"state": "idle"}
    return {k: v for k, v in JOB.items() if k not in ("stop", "t0")}


@app.post("/api/stop")
def stop():
    if JOB and JOB["state"] == "running":
        JOB["stop"] = True
    return {"ok": True}
