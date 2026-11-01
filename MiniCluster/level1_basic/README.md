# MiniCluster - Level 1 (Distributed Squares)

Web se N tak ke numbers ke squares 3 PCs par baant kar nikalna.

```
Browser -> FastAPI Coordinator -> TCP sockets -> Worker PC-01/02/03 -> combine
```

## Files
- `worker.py` : har PC par chalta hai (TCP socket server, har request alag thread mein)
- `coordinator.py` : FastAPI server, kaam baantta hai aur results combine karta hai
- `workers.json` : workers ke IP/port
- `static/index.html` : dashboard
- `requirements.txt`

## Setup (har PC par ek baar)
```
pip install -r requirements.txt
```
(Workers ko sirf Python chahiye, FastAPI sirf coordinator PC par zaroori hai.)

## A) Pehle ek hi laptop par test (VS Code terminal)
Teen alag terminals kholein:
```
python worker.py --port 5001
python worker.py --port 5002
python worker.py --port 5003
```
Chautha terminal:
```
uvicorn coordinator:app --port 8000
```
Browser mein kholein: http://localhost:8000

## B) Asli 3 lab PCs par
1. Teeno PCs ek hi network/WiFi/switch par hon.
2. Har worker PC par `ipconfig` (Windows) chala kar IPv4 address note karein.
3. Har worker PC par: `python worker.py --port 5001`
4. Windows Firewall popup aaye to **Allow** karein (ya port 5001 allow karein).
5. Coordinator PC par `workers.json` mein IP likhein:
```json
[
  {"name": "PC-01", "host": "192.168.1.11", "port": 5001},
  {"name": "PC-02", "host": "192.168.1.12", "port": 5001},
  {"name": "PC-03", "host": "192.168.1.13", "port": 5001}
]
```
6. Coordinator PC par: `uvicorn coordinator:app --host 0.0.0.0 --port 8000`
7. Browser: http://localhost:8000 (doosre PCs se `http://<coordinator-ip>:8000`)

## Use karna
- **Run**: N, workers ki tadaad aur work factor chun kar chalayein. Har worker ka range aur time dikhega.
- **Compare 1/2/3 PCs**: teen runs khud chalata hai aur Speedup/Efficiency table banata hai.
- **Work factor**: 10,000 squares itne tez hote hain ke network overhead speedup chhupa deta hai. Isliye har number par extra computation (work factor) lagti hai. Isay ghata kar dekhein ke speedup kam kyun ho jata hai (communication overhead).

## Students ke liye sawal
1. 3 PCs par time ideal 1/3 kyun nahi?
2. Work factor 1 par speedup kya hota hai? Kyun?
3. Ek worker band kar ke Run karein. Kya hota hai? (Level 2/3 mein fault tolerance add hogi)

## Manual ke topics ka connection
Socket programming (worker/coordinator), Multithreading (worker har connection par thread, coordinator parallel dispatch).
Aglay levels: multiprocessing/OpenMP (worker ke andar cores), MPI (mpi4py).
