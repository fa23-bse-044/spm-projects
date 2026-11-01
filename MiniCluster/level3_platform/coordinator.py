"""MiniCluster Level 3 coordinator (FastAPI): web-based distributed ML platform.
Usage: uvicorn coordinator:app --host 0.0.0.0 --port 8000

Algorithms
  linreg / logreg : data-parallel gradient descent (har epoch: weights bhejo, gradients sum karo)
  rf              : Random Forest, trees tasks mein bante hain; idle worker ko agla task milta hai
                    (dynamic load balancing). Worker mar jaye to uske tasks dobara schedule hote hain.
"""
import csv
import io
import json
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import datasets
from common import dec, enc, recv_msg, send_msg
from datasets import DatasetError

BASE = Path(__file__).parent
WORKERS = json.loads((BASE / "workers.json").read_text())
RF_BATCH = 5  # trees per task (fixed, taake 1/2/3 PCs par forest same rahe)

app = FastAPI(title="MiniCluster ML")
JOB = None
JOB_LOCK = threading.Lock()
HISTORY = []


class TrainRequest(BaseModel):
    dataset: str = "student_pass"
    n: int = Field(100_000, ge=100, le=2_000_000)
    target: str | None = None
    task: str = Field("auto", pattern="^(auto|classification|regression)$")
    algo: str = Field("logreg", pattern="^(linreg|logreg|rf)$")
    workers: int = Field(3, ge=1)
    epochs: int = Field(60, ge=1, le=3000)
    lr: float = Field(0.5, gt=0, le=5.0)
    work: int = Field(1, ge=1, le=2000)
    trees: int = Field(30, ge=1, le=2000)
    max_depth: int = Field(10, ge=0, le=100)
    cores: int = Field(1, ge=1, le=64)
    slow_ms: int = Field(0, ge=0, le=5000)


# ---------------------------------------------------------------- helpers
def call_worker(w, msg, timeout=120):
    with socket.create_connection((w["host"], w["port"]), timeout=5) as s:
        s.settimeout(timeout)
        send_msg(s, msg)
        return recv_msg(s)


def parallel(workers, make_msg, timeout=120):
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


# ---------------------------------------------------------------- gradient descent
def distribute_shards(job, alive, ds, algo):
    """Train data ko alive workers mein shards mein baanto. Fail hone wale hata kar dobara."""
    while True:
        if not alive:
            raise RuntimeError("Koi worker available nahi")
        rng = dict(zip([w["name"] for w in alive], shard_ranges(len(ds["ytr"]), len(alive))))

        def msg(w):
            a, b = rng[w["name"]]
            return {"type": "load", "mode": "shard", "algo": algo, "K": ds["K"],
                    "X": enc(ds["Xtr"][a:b]), "y": enc(ds["ytr"][a:b])}

        ok, failed = parallel(alive, msg)
        for w, r in ok:
            job["workers"][w["name"]].update(state="online", rows=r["rows"])
        if not failed:
            return alive
        mark_offline(job, failed)
        dead = {w["name"] for w, _ in failed}
        alive = [w for w in alive if w["name"] not in dead]
        event(job, f"Data {len(alive)} workers mein dobara baanta ja raha hai")


def run_gd(job, req, ds, alive):
    ytr = ds["ytr"]
    ds = {**ds, "ytr": ytr.astype(np.int64) if req.algo == "logreg" else ytr.astype(float)}
    job["phase"] = "loading"
    t = time.perf_counter()
    alive = distribute_shards(job, alive, ds, req.algo)
    load_time = time.perf_counter() - t
    event(job, f"Data shards load: {', '.join(w['name'] for w in alive)} ({load_time:.2f}s)")

    d1, K = ds["Xtr"].shape[1] + 1, ds["K"]
    w_vec = np.zeros(d1 * K if req.algo == "logreg" else d1)
    N = len(ytr)
    epoch_wall, epoch_compute = [], []
    job["phase"] = "training"
    t_train = time.perf_counter()
    epoch = 0
    while epoch < req.epochs:
        if job["stop"]:
            return None
        t_ep = time.perf_counter()
        ok, failed = parallel(alive, lambda w: {"type": "gradient", "algo": req.algo, "w": w_vec.tolist(), "work": req.work})
        if failed:  # fault tolerance: re-shard aur isi epoch ko dobara chalao
            mark_offline(job, failed)
            dead = {w["name"] for w, _ in failed}
            alive = [w for w in alive if w["name"] not in dead]
            event(job, f"Epoch {epoch + 1} dobara chalega, task reassignment")
            alive = distribute_shards(job, alive, ds, req.algo)
            event(job, "Re-shard done: " + ", ".join(f"{w['name']}={job['workers'][w['name']]['rows']:,} rows" for w in alive))
            continue
        grad = sum(np.array(r["grad"]) for _, r in ok)
        rows = sum(r["rows"] for _, r in ok)
        w_vec -= req.lr * grad / rows
        for w, r in ok:
            job["workers"][w["name"]]["compute_ms"] = round(r["time"] * 1000, 1)
        epoch_compute.append(max(r["time"] for _, r in ok))
        epoch += 1
        job["losses"].append(sum(r["loss"] for _, r in ok) / rows)
        if req.algo == "logreg":
            job["accs"].append(sum(r["correct"] for _, r in ok) / rows)
        job["epoch"] = epoch
        if req.slow_ms:
            time.sleep(req.slow_ms / 1000)
        epoch_wall.append(time.perf_counter() - t_ep - req.slow_ms / 1000)
    compute_time = time.perf_counter() - t_train - req.slow_ms / 1000 * req.epochs

    job["phase"] = "evaluating"
    Xte = np.hstack([ds["Xte"], np.ones((len(ds["Xte"]), 1))])
    if req.algo == "logreg":
        pred = (Xte @ w_vec.reshape(d1, K)).argmax(1)
        metrics = datasets.classification_metrics(ds["yte"], pred, K)
    else:
        pred = (Xte @ w_vec) * ds["ysd"] + ds["ymu"]
        metrics = datasets.regression_metrics(ds["yte"], pred)
    return {"metrics": metrics, "compute_time": compute_time, "load_time": load_time, "workers_used": len(alive),
            "avg_epoch_ms": float(np.mean(epoch_wall)) * 1000, "avg_compute_ms": float(np.mean(epoch_compute)) * 1000}


# ---------------------------------------------------------------- random forest
def rf_train_missing(job, req, ds, alive, tasks, owner):
    """Jo tasks owner mein nahi, unhein alive workers par chalao. Idle worker queue se agla task uthata hai."""
    lock = threading.Lock()
    while True:
        missing = [t for t in tasks if t[0] not in owner]
        if not missing or job["stop"]:
            return alive
        if not alive:
            raise RuntimeError("Koi worker available nahi")
        q = queue.Queue()
        for t in missing:
            q.put(t)
        dead = set()

        def loop(w):
            name = w["name"]
            while not job["stop"]:
                try:
                    tid, trees = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    r = call_worker(w, {"type": "rf_train", "task": tid, "trees": trees, "seed": 1000 + tid,
                                        "max_depth": req.max_depth, "cores": req.cores, "kind": ds["task"]}, timeout=900)
                    if r.get("type") == "error":
                        raise RuntimeError(r["error"])
                except Exception as e:
                    with lock:
                        dead.add(name)
                        lost = [k for k, v in owner.items() if v == name]
                        for k in lost:  # is worker par bane trees ab unreachable hain
                            del owner[k]
                        mark_offline(job, [(w, str(e) or type(e).__name__)])
                        if lost:
                            event(job, f"{name} ke {len(lost)} finished tasks dobara schedule honge")
                        sync_counts(job, owner)
                    return
                with lock:
                    owner[tid] = name
                    job["workers"][name]["compute_ms"] = round(r["time"] * 1000, 1)
                    sync_counts(job, owner)

        threads = [threading.Thread(target=loop, args=(w,)) for w in alive]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        alive = [w for w in alive if w["name"] not in dead]


def sync_counts(job, owner):
    for name, info in job["workers"].items():
        info["tasks"] = sum(1 for v in owner.values() if v == name)
    job["tasks_done"] = len(owner)


def run_rf(job, req, ds, alive):
    job["phase"] = "loading"
    t = time.perf_counter()
    full = {"type": "load", "mode": "full", "K": ds["K"], "X": enc(ds["Xtr"]),
            "y": enc(ds["ytr"].astype(np.int64) if ds["task"] == "classification" else ds["ytr"].astype(float))}
    ok, failed = parallel(alive, lambda w: full)  # har worker ko poora train data
    for w, r in ok:
        job["workers"][w["name"]].update(state="online", rows=r["rows"])
    mark_offline(job, failed)
    dead = {w["name"] for w, _ in failed}
    alive = [w for w in alive if w["name"] not in dead]
    if not alive:
        raise RuntimeError("Koi worker data load nahi kar saka")
    load_time = time.perf_counter() - t
    event(job, f"Full train data broadcast: {', '.join(w['name'] for w in alive)} ({load_time:.2f}s)")

    tasks = [(i, min(RF_BATCH, req.trees - i * RF_BATCH)) for i in range((req.trees + RF_BATCH - 1) // RF_BATCH)]
    job["tasks_total"] = len(tasks)
    owner = {}
    job["phase"] = "training"
    t_train = time.perf_counter()
    alive = rf_train_missing(job, req, ds, alive, tasks, owner)
    if job["stop"]:
        return None

    job["phase"] = "predicting"
    Xte = enc(ds["Xte"])
    while True:
        by = {}
        for tid, name in owner.items():
            by.setdefault(name, []).append(tid)
        targets = [w for w in alive if w["name"] in by]
        ok, failed = parallel(targets, lambda w: {"type": "rf_predict", "tasks": by[w["name"]], "X": Xte}, timeout=900)
        if not failed:
            break
        mark_offline(job, failed)
        for w, _ in failed:
            for k in [k for k, v in owner.items() if v == w["name"]]:
                del owner[k]
        dead = {w["name"] for w, _ in failed}
        alive = [w for w in alive if w["name"] not in dead]
        sync_counts(job, owner)
        event(job, "Prediction phase mein worker gira, trees dobara train honge")
        job["phase"] = "training"
        alive = rf_train_missing(job, req, ds, alive, tasks, owner)
        if job["stop"]:
            return None
        job["phase"] = "predicting"
    compute_time = time.perf_counter() - t_train

    job["phase"] = "evaluating"
    total = sum(dec(r["sum"]) for _, r in ok)
    trees = sum(r["trees"] for _, r in ok)
    avg = total / trees
    if ds["task"] == "classification":
        metrics = datasets.classification_metrics(ds["yte"], avg.argmax(1), ds["K"])
    else:
        metrics = datasets.regression_metrics(ds["yte"], avg * ds["ysd"] + ds["ymu"])
    return {"metrics": metrics, "compute_time": compute_time, "load_time": load_time, "workers_used": len(alive),
            "trees": trees}


# ---------------------------------------------------------------- job runner
def train_job(job, req):
    try:
        job["phase"] = "preparing"
        ds = datasets.prepare(req.dataset, req.n, req.target, req.task)
        ok_algo = {"classification": ("logreg", "rf"), "regression": ("linreg", "rf")}[ds["task"]]
        if req.algo not in ok_algo:
            raise RuntimeError(f"{req.algo} algorithm {ds['task']} dataset ke liye nahi hai (use: {', '.join(ok_algo)})")
        job["info"] = {"dataset": req.dataset, "rows": ds["rows"], "features": len(ds["names"]), "task": ds["task"],
                       "classes": ds["class_names"], "train_rows": len(ds["ytr"]), "test_rows": len(ds["yte"])}

        chosen = WORKERS[: req.workers]
        job["workers"] = {w["name"]: {"state": "pending", "rows": 0, "compute_ms": 0, "tasks": 0} for w in chosen}
        ok, failed = parallel(chosen, lambda w: {"type": "ping"}, timeout=3)
        mark_offline(job, failed)
        alive = [w for w, _ in ok]
        if not alive:
            raise RuntimeError("Koi worker online nahi hai")

        res = (run_rf if req.algo == "rf" else run_gd)(job, req, ds, alive)
        if res is None:
            job["state"] = "stopped"
            return
        res["algo"] = req.algo
        res["task"] = ds["task"]
        res["class_names"] = ds["class_names"]
        job["result"] = res
        m = res["metrics"]
        name, val = ("accuracy", m["accuracy"]) if ds["task"] == "classification" else ("r2", m["r2"])
        HISTORY.append({"id": len(HISTORY) + 1, "dataset": req.dataset, "algo": req.algo, "task": ds["task"],
                        "workers_requested": req.workers, "workers_used": res["workers_used"],
                        "metric": name, "value": round(val, 4), "compute_s": round(res["compute_time"], 3),
                        "load_s": round(res["load_time"], 3), "rows": ds["rows"]})
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
            r = call_worker(w, {"type": "ping"}, timeout=3)
            return {**w, "online": True, "cpu": r.get("cpu"), "ram": r.get("ram")}
        except Exception:
            return {**w, "online": False}

    with ThreadPoolExecutor(len(WORKERS)) as ex:
        return list(ex.map(ping, WORKERS))


@app.get("/api/datasets")
def list_datasets():
    return datasets.list_datasets()


@app.post("/api/dataset/upload")
async def upload(request: Request, name: str = "upload.csv"):
    body = await request.body()
    if len(body) > 100_000_000:
        raise HTTPException(413, "File 100MB se badi hai")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = body.decode("latin-1")
    try:
        return datasets.register_csv(name, text)
    except DatasetError as e:
        raise HTTPException(400, str(e))


@app.get("/api/dataset/preview")
def preview(name: str, n: int = 100_000, target: str | None = None, task: str = "auto"):
    try:
        return datasets.summary(name, n, target or None, task)
    except DatasetError as e:
        raise HTTPException(400, str(e))


@app.post("/api/train")
def start_train(req: TrainRequest):
    global JOB
    if req.workers > len(WORKERS):
        raise HTTPException(400, f"Sirf {len(WORKERS)} workers configured hain")
    with JOB_LOCK:
        if JOB and JOB["state"] == "running":
            raise HTTPException(409, "Ek job pehle se chal rahi hai")
        JOB = {"state": "running", "phase": "starting", "algo": req.algo, "epoch": 0, "epochs": req.epochs,
               "tasks_done": 0, "tasks_total": 0, "losses": [], "accs": [], "events": [], "workers": {},
               "info": None, "result": None, "error": None, "stop": False, "t0": time.perf_counter()}
        threading.Thread(target=train_job, args=(JOB, req), daemon=True).start()
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


@app.get("/api/history")
def history():
    return HISTORY


@app.get("/api/history.csv")
def history_csv():
    buf = io.StringIO()
    cols = ["id", "dataset", "algo", "task", "rows", "workers_requested", "workers_used", "metric", "value", "compute_s", "load_s"]
    wr = csv.DictWriter(buf, fieldnames=cols)
    wr.writeheader()
    wr.writerows(HISTORY)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=experiments.csv"})
