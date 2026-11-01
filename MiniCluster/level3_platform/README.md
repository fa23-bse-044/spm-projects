# MiniCluster ML - Level 3 (Web-based Distributed ML Platform)

Browser se dataset + algorithm + PCs chunein, aur 3 lab PCs par distributed ML experiment chalayein.

```
Browser -> FastAPI Coordinator --TCP sockets--> Worker PC-01 / PC-02 / PC-03
```

## Kya kya hai
| Cheez | Details |
|---|---|
| Datasets | `student_pass` (classification) aur `student_score` (regression) synthetic, N 100 se 2,000,000 tak. Plus iris, wine, breast_cancer, digits, diabetes. Plus **apni CSV upload** |
| Algorithms | Logistic Regression (multiclass softmax), Linear Regression, Random Forest |
| Metrics | Accuracy, macro-F1, confusion matrix (classification); R², RMSE, MAE (regression), sab test set (20%) par |
| Dashboard | Live CPU/RAM, loss curve, per-worker state, event log, experiment history (CSV download) |
| Analysis | "Compare 1/2/3 PCs": time, speedup, efficiency, Amdahl serial fraction |
| Fault tolerance | Worker girne par kaam baqi workers par dobara schedule hota hai |

## Algorithms kaise distribute hote hain
**Logistic / Linear Regression (data parallel):** train data workers mein shards mein baanta jata hai. Har epoch:
coordinator weights bhejta hai -> har worker apne shard ka gradient wapas bhejta hai -> coordinator gradients ka sum
kar ke weights update karta hai. Result single-PC training ke barabar aata hai (1, 2, 3 PCs par same accuracy).
Worker girne par data baqi workers mein dobara baanta jata hai aur wahi epoch dobara chalta hai.

**Random Forest (task parallel):** trees 5-5 ke tasks mein bante hain. Har worker ke paas poora train data hota hai.
Jo worker free hota hai wo queue se agla task le leta hai, yani **dynamic load balancing**: tez PC zyada tasks karta hai.
Prediction mein har worker apne trees se test set par probabilities nikalta hai, coordinator average karta hai.
Worker girne par uske bane hue trees bhi ghayab ho jate hain, isliye uske saare tasks dobara train hote hain.
Seeds task-id se bante hain, is liye forest (aur accuracy) PCs ki tadaad se nahi badalti.

## Setup
Har PC par poora folder copy karein, phir `pip install -r requirements.txt`.

### A) Ek laptop par test (VS Code, 4 terminals)
```
python worker.py --port 5001
python worker.py --port 5002
python worker.py --port 5003
uvicorn coordinator:app --port 8000
```
Browser: http://localhost:8000

### B) Asli 3 lab PCs
1. Worker PCs par `ipconfig` se IP note karein, `python worker.py --port 5001` chalayein (firewall: Allow).
2. Coordinator PC par `workers.json` mein IPs likhein, phir `uvicorn coordinator:app --host 0.0.0.0 --port 8000`.
3. Browser: `http://<coordinator-ip>:8000`

Worker default mein 1 CPU thread use karta hai. Poore cores ke liye `set MC_THREADS=8` (Windows) ya `MC_THREADS=8 python worker.py`.
Random Forest ke liye dashboard ka **Cores / worker** bhi use ho sakta hai (har PC ke andar multiprocessing).

## Apni CSV
"Apni CSV" se file chunein. Default target aakhri column hota hai (dropdown se badal sakte hain). Target text ho ya kam values wala
integer ho to classification, warna regression (Task dropdown se override ho sakta hai). Text columns aur `id` jaise columns
feature se hata diye jate hain (preview mein dikhta hai), missing numeric values column ke average se bhar di jati hain.

## Demos
1. **Speedup:** `student_pass`, N = 300000, Random Forest, "Compare 1/2/3 PCs". Phir N ko 5000 kar ke dekhein ke speedup kyun gir jata hai.
2. **Same accuracy:** Compare table mein accuracy har PC count par barabar hai. Kyun?
3. **Load balancing:** ek worker ko `python worker.py --port 5001 --slow 4` se 4 guna slow chalayein, Random Forest run karein, aur Live table mein "Tasks done" dekhein.
4. **Fault tolerance (GD):** Logistic Regression, Slow-motion (Advanced) 200 ms, run karein, beech mein ek worker par `Ctrl+C`.
5. **Fault tolerance (RF):** 120 trees, run karein, beech mein ek worker band karein. Event log mein uske finished tasks ka dobara schedule hona dikhega.
6. **Real data:** iris/wine/breast_cancer par Logistic Regression vs Random Forest ki accuracy compare karein (chhote data par parallel faida nahi, kyun?).
7. **Lab report:** history ka CSV download karein.

## Sawal
1. Random Forest ka speedup logistic regression se behtar kyun hai? (communication per unit of work)
2. Amdahl ka serial fraction kin cheezon se banta hai?
3. RF mein data shard karne (har worker ko sirf apna hissa) aur poora data dene mein accuracy/speed ka kya trade-off hai?
4. Gradient descent mein har epoch par sync ki zarurat kyun hai?

## Files
`worker.py` `coordinator.py` `datasets.py` `data.py` `common.py` `workers.json` `static/index.html` `requirements.txt`

Note: coordinator ka dataset sirf coordinator PC se aata hai, workers ko data network se milta hai, is liye pehle run mein "data transfer" ka time alag dikhta hai.
