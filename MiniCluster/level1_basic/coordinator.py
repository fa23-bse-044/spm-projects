"""MiniCluster coordinator (FastAPI). Sirf ek PC par chalayein.
Usage: uvicorn coordinator:app --host 0.0.0.0 --port 8000
"""
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from worker import recv_msg, send_msg

BASE = Path(__file__).parent
WORKERS = json.loads((BASE / "workers.json").read_text())

app = FastAPI(title="MiniCluster")


class RunRequest(BaseModel):
    n: int = Field(10000, ge=1, le=5_000_000)
    workers: int = Field(3, ge=1)
    work: int = Field(1, ge=1, le=100_000)


def call_worker(w, msg, timeout=60):
    with socket.create_connection((w["host"], w["port"]), timeout=timeout) as s:
        send_msg(s, msg)
        return recv_msg(s)


def split_ranges(n, k):
    """1..n ko k barabar hisson mein baantna."""
    size, rem = divmod(n, k)
    ranges, start = [], 1
    for i in range(k):
        end = start + size + (1 if i < rem else 0) - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/status")
def status():
    def ping(w):
        try:
            r = call_worker(w, {"type": "ping"}, timeout=2)
            return {**w, "online": True, "host_name": r["host"]}
        except Exception:
            return {**w, "online": False}

    with ThreadPoolExecutor(len(WORKERS)) as ex:
        return list(ex.map(ping, WORKERS))


@app.post("/api/run")
def run(req: RunRequest):
    if req.workers > len(WORKERS):
        raise HTTPException(400, f"Sirf {len(WORKERS)} workers configured hain")
    chosen = WORKERS[: req.workers]
    ranges = split_ranges(req.n, req.workers)

    def job(args):
        w, (a, b) = args
        try:
            r = call_worker(w, {"type": "square", "start": a, "end": b, "work": req.work})
            return {"worker": w["name"], "start": a, "end": b, "ok": True,
                    "time": r["time"], "result": r["result"]}
        except Exception as e:
            return {"worker": w["name"], "start": a, "end": b, "ok": False, "error": str(e)}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(req.workers) as ex:
        parts = list(ex.map(job, zip(chosen, ranges)))
    total = time.perf_counter() - t0

    failed = [p for p in parts if not p["ok"]]
    if failed:
        raise HTTPException(502, "Worker fail: " + ", ".join(f"{p['worker']} ({p['error']})" for p in failed))

    combined = []
    for p in parts:  # ranges ordered hain, isliye concat sahi order deta hai
        combined.extend(p["result"])

    return {
        "n": req.n,
        "workers": req.workers,
        "total_time": total,
        "correct": combined == [x * x for x in range(1, req.n + 1)],
        "preview": combined[:10],
        "last": combined[-3:],
        "parts": [{"worker": p["worker"], "start": p["start"], "end": p["end"],
                   "count": p["end"] - p["start"] + 1, "time": p["time"]} for p in parts],
    }
