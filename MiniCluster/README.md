# MiniCluster: Web-Based Parallel & Distributed ML Platform

Teen lab PCs ko ek distributed computing platform bana kar browser se ML tasks chalana.
P&DC course ke topics: Socket Programming, Multithreading, Multiprocessing, MPI-style distributed computation.

```
Browser -> FastAPI Coordinator --TCP sockets--> Worker PC-01 / PC-02 / PC-03 -> combine
```

## Folder structure
| Folder | Level | Kya sikhata hai |
|---|---|---|
| `level1_basic/` | 1 | Basic distributed computing: 1..N ke squares 3 PCs par. Sockets, threads, speedup |
| `level2_regression/` | 2 | Distributed Linear Regression: gradient aggregation, fault tolerance, live dashboard |
| `level3_platform/` | 3 | **Final project:** dataset (built-in ya CSV) + algorithm (Logistic/Linear Regression, Random Forest) + PCs browser se chunein, load balancing, speedup/Amdahl analysis, history |

Har folder alag chalta hai aur apni `README.md` rakhta hai. Final submission/demo ke liye `level3_platform` use karein;
Level 1 aur 2 samjhane ke liye step-by-step progression hain.

## Quick start (final project, Level 3)
```
cd level3_platform
pip install -r requirements.txt

python worker.py --port 5001      # alag terminal
python worker.py --port 5002      # alag terminal
python worker.py --port 5003      # alag terminal
uvicorn coordinator:app --port 8000
```
Browser: http://localhost:8000

Asli 3 PCs par: har PC par folder copy karein, worker chalayein, coordinator PC par `workers.json` mein teeno IPs likhein aur
`uvicorn coordinator:app --host 0.0.0.0 --port 8000` chalayein (Windows Firewall mein Allow karein).

## Manual ke topics ka mapping
| Topic | Project mein kahan |
|---|---|
| Socket programming | Worker/coordinator ke darmiyan length-prefixed JSON messages (`common.py`) |
| Multithreading | Worker har connection par thread; coordinator workers ko parallel threads se call karta hai |
| Multiprocessing | RF ke liye "Cores / worker" (har PC ke andar cores) |
| Distributed memory | Har worker ka apna data shard, communication sirf messages se |
| MPI-style collective | Gradient sum = reduce/allreduce ka concept |
| Speedup, efficiency, Amdahl | "Compare 1/2/3 PCs" table |
| Fault tolerance, load balancing | Worker band hone par reassignment, `--slow` worker demo |

## Teaching scenario (bachon ke liye ek line)
"Hamare paas ek bara ML ka kaam hai. Ek computer akela karne ke bajaye hum 3 computers ko network ke through ek team bana dete hain.
Har computer apna hissa process karta hai aur akhir mein coordinator sab ke results combine karke final answer deta hai."

## Tested
Teeno levels 3 workers + coordinator ke saath test hue: results 1/2/3 PCs par same, fault tolerance (GD aur RF), load balancing,
CSV upload, aur dashboard headless Chrome mein. Test machine mein sirf 2 cores the, is liye 3 PCs par speedup wahan cap hota tha;
asli 3 PCs par behtar aana chahiye.

## Ideas for next steps
`mpi4py` se gradient `Allreduce`, GPU workers (PyTorch), worker auto-discovery, CSV preview mein feature selection.
