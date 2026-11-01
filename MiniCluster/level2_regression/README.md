# MiniCluster ML - Level 2 (Distributed Linear Regression)

University ke student records (matric, intermediate, entry_test, attendance, study_hours, prev_gpa) se
final score predict karna, training 3 PCs par baant kar.

```
Browser -> FastAPI Coordinator --TCP--> Worker PC-01 / PC-02 / PC-03
                  ^                          |
                  +---- gradients <----------+   (har epoch)
```

## Algorithm (data-parallel gradient descent)
Har epoch:
1. Coordinator current weights `w` sab workers ko bhejta hai.
2. Har worker apne shard par gradient nikalta hai (data apne PC par hi rehta hai).
3. Coordinator gradients ka **sum** karta hai (aggregation), `w -= lr * grad / N`.
4. Agla epoch.

Ye single PC par poore data se training ke **bilkul barabar** result deta hai. Dashboard mein learned weights
closed-form (least squares) solution se compare hoti hain. 1, 2, 3 workers par weights wahi aate hain.

> Sirf data 3 hisson mein kaat kar 3 alag models train kar dena aur average karna hamesha barabar nahi hota
> (jaise non-linear models mein). Isliye har epoch mein gradient sync zaroori hai.

## Files
- `worker.py` : har PC par (data shard load karta hai, gradient compute karta hai)
- `coordinator.py` : FastAPI, scheduling, aggregation, fault tolerance
- `data.py` : synthetic dataset (har worker apna hissa khud generate karta hai)
- `common.py` : socket messages
- `workers.json` : workers ke IP/port
- `static/index.html` : dashboard

## Setup
Har PC par poora folder copy karein, phir:
```
pip install -r requirements.txt
```

## A) Ek laptop par test (VS Code, 4 terminals)
```
python worker.py --port 5001
python worker.py --port 5002
python worker.py --port 5003
uvicorn coordinator:app --port 8000
```
Browser: http://localhost:8000

## B) Asli 3 lab PCs par
1. Teeno PCs ek network par, har worker PC par `ipconfig` se IP note karein.
2. Worker PCs par `python worker.py --port 5001` (firewall popup par Allow).
3. Coordinator PC par `workers.json` mein IPs likhein, phir `uvicorn coordinator:app --host 0.0.0.0 --port 8000`.
4. Browser: `http://<coordinator-ip>:8000`.

Note: worker default mein 1 CPU thread use karta hai (fair comparison ke liye). Pure PC ke saare cores
chahiye to chalane se pehle `set MC_THREADS=8` (Windows) ya `MC_THREADS=8 python worker.py ...`.

## Demos
**1. Speedup:** "Compare 1 / 2 / 3 PCs" dabayein. Time, Speedup, Efficiency table banti hai.
Work factor kam kar ke dekhein ke speedup kyun gir jata hai (communication overhead). Dashboard "avg network/sync overhead" bhi dikhata hai.

**2. Fault tolerance:** Slow-motion `200` ms set karein, Start Training dabayein, phir kisi worker ke terminal mein
`Ctrl+C` karein. Event log mein dikhega: worker OFFLINE, task reassignment, re-shard, aur training baqi workers par
poori ho jati hai, result wahi aata hai.

**3. Consistency:** Workers 1, 2, 3 par R² aur weights same aate hain. Kyun?

## Students ke sawal
1. 3 PCs par time 1/3 kyun nahi?
2. Epochs badhane par communication ka share kya hota hai?
3. Worker band hone par epoch dobara kyun chalaya gaya (aadha epoch kyun nahi use kiya)?
4. Learning rate 1.5 karein to kya hota hai?

## Aglay level ke ideas
Multiprocessing/OpenMP har worker ke andar, MPI (mpi4py) se gradient `Allreduce`, Random Forest, CSV upload.
