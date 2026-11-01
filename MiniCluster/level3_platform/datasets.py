"""Datasets: built-in (synthetic + sklearn), CSV upload, preprocessing, metrics."""
import csv
import io
import threading

import numpy as np

from data import FEATURES as SYN_FEATURES
from data import generate

SEED = 42


class DatasetError(Exception):
    pass


SYNTH = {"student_pass": "classification", "student_score": "regression"}
SKL = {  # name: (loader, task, rows, features)
    "iris": ("load_iris", "classification", 150, 4),
    "wine": ("load_wine", "classification", 178, 13),
    "breast_cancer": ("load_breast_cancer", "classification", 569, 30),
    "digits": ("load_digits", "classification", 1797, 64),
    "diabetes": ("load_diabetes", "regression", 442, 10),
}
ID_LIKE = {"id", "index", "unnamed: 0", "sr", "sr.", "s.no", "sno", "serial", "serial no", "row", "row_id"}
UPLOADED = {}
_CACHE = {}
_LOCK = threading.Lock()


# ------------------------------------------------------------------ registry
def list_datasets():
    out = []
    for name, task in SYNTH.items():
        out.append({"name": name, "source": "synthetic", "rows": None, "features": len(SYN_FEATURES),
                    "task": task, "adjustable": True})
    for name, (_, task, rows, feats) in SKL.items():
        out.append({"name": name, "source": "sklearn", "rows": rows, "features": feats,
                    "task": task, "adjustable": False})
    for name, raw in UPLOADED.items():
        out.append({"name": name, "source": "upload", "rows": raw["rows"], "features": len(raw["header"]) - 1,
                    "task": None, "adjustable": False, "columns": raw["header"]})
    return out


def register_csv(name, text):
    name = (name or "upload.csv").replace("\\", "/").split("/")[-1].strip()[:60] or "upload.csv"
    if name in SYNTH or name in SKL:
        name += "_csv"
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if len(rows) < 11:
        raise DatasetError("CSV mein kam az kam 10 rows aur 1 header chahiye")
    header = []
    for i, h in enumerate(rows[0]):
        h = h.strip() or f"col{i}"
        while h in header:
            h += "_"
        header.append(h)
    if len(header) < 2:
        raise DatasetError("CSV mein kam az kam 2 columns chahiye (features + target)")
    body = [r for r in rows[1:] if len(r) == len(header) and any(c.strip() for c in r)]
    if len(body) < 10:
        raise DatasetError("Valid rows 10 se kam hain (columns ki tadaad match nahi kar rahi)")
    cols = {h: [r[i].strip() for r in body] for i, h in enumerate(header)}
    UPLOADED[name] = {"header": header, "cols": cols, "rows": len(body)}
    with _LOCK:
        _CACHE.clear()
    return {"name": name, "rows": len(body), "columns": header}


# ------------------------------------------------------------------ loading
def _to_float(values):
    out = np.empty(len(values))
    for i, v in enumerate(values):
        if v == "":
            out[i] = np.nan
        else:
            out[i] = float(v)  # ValueError agar numeric nahi
    return out


def _build_uploaded(raw, target, task_pref):
    header = raw["header"]
    target = target or header[-1]
    if target not in header:
        raise DatasetError(f"Target column '{target}' nahi mila")
    keep = [i for i, v in enumerate(raw["cols"][target]) if v != ""]
    if len(keep) < 10:
        raise DatasetError("Target column mein bohat missing values hain")
    tvals = [raw["cols"][target][i] for i in keep]
    dropped, feats, names = [], [], []
    for h in header:
        if h == target:
            continue
        if h.lower() in ID_LIKE:  # row-id columns feature nahi hote
            dropped.append(h)
            continue
        try:
            col = _to_float([raw["cols"][h][i] for i in keep])
        except ValueError:
            dropped.append(h)
            continue
        if np.all(np.isnan(col)):
            dropped.append(h)
            continue
        col[np.isnan(col)] = np.nanmean(col)
        feats.append(col)
        names.append(h)
    if not feats:
        raise DatasetError("Koi numeric feature column nahi mila")
    X = np.column_stack(feats)

    class_names = None
    try:
        y = _to_float(tvals)
        numeric = True
    except ValueError:
        numeric = False
    if numeric:
        uniq = np.unique(y)
        is_int = bool(np.all(y == np.round(y)))
        if task_pref == "classification" or (task_pref == "auto" and len(uniq) <= 20 and is_int):
            task = "classification"
            class_names = [str(int(u)) if is_int else str(u) for u in uniq]
            y = np.searchsorted(uniq, y).astype(np.int64)
        else:
            task = "regression"
    else:
        if task_pref == "regression":
            raise DatasetError("Target text hai, regression nahi ho sakti")
        task = "classification"
        uniq = sorted(set(tvals))
        class_names = uniq
        lookup = {u: i for i, u in enumerate(uniq)}
        y = np.array([lookup[v] for v in tvals], dtype=np.int64)
    if task == "classification" and not (2 <= len(class_names) <= 50):
        raise DatasetError(f"Classes ki tadaad {len(class_names)} hai (2 se 50 chahiye). Task 'regression' try karein.")
    return {"X": X, "y": y, "names": names, "task": task, "class_names": class_names, "dropped": dropped}


def load_dataset(name, n=100_000, target=None, task="auto"):
    if name in SYNTH:
        X, y = generate(0, n)
        if name == "student_pass":
            return {"X": X, "y": (y >= 50).astype(np.int64), "names": list(SYN_FEATURES), "task": "classification",
                    "class_names": ["fail", "pass"], "dropped": []}
        return {"X": X, "y": y, "names": list(SYN_FEATURES), "task": "regression", "class_names": None, "dropped": []}
    if name in SKL:
        import importlib
        loader, tsk, _, _ = SKL[name]
        try:
            ds = getattr(importlib.import_module("sklearn.datasets"), loader)()
        except ImportError:
            raise DatasetError("scikit-learn install karein: pip install scikit-learn")
        names = [str(f) for f in ds.feature_names] if hasattr(ds, "feature_names") else [f"f{i}" for i in range(ds.data.shape[1])]
        cn = [str(c) for c in ds.target_names] if tsk == "classification" else None
        y = ds.target.astype(np.int64) if tsk == "classification" else ds.target.astype(float)
        return {"X": ds.data.astype(float), "y": y, "names": names, "task": tsk, "class_names": cn, "dropped": []}
    if name in UPLOADED:
        return _build_uploaded(UPLOADED[name], target, task)
    raise DatasetError(f"Dataset '{name}' nahi mila")


def prepare(name, n, target, task):
    """Load + train/test split (80/20) + standardize. Result cached."""
    key = (name, n if name in SYNTH else 0, target, task)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    ds = load_dataset(name, n, target, task)
    N = len(ds["y"])
    if N < 10:
        raise DatasetError("Dataset bohat chhota hai")
    idx = np.random.default_rng(SEED).permutation(N)
    n_test = max(2, int(0.2 * N))
    te, tr = idx[:n_test], idx[n_test:]
    mu = ds["X"][tr].mean(0)
    sd = ds["X"][tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (ds["X"][tr] - mu) / sd, (ds["X"][te] - mu) / sd
    ytr, yte = ds["y"][tr], ds["y"][te]
    ymu, ysd = 0.0, 1.0
    if ds["task"] == "regression":  # training target standardize; yte raw rehta hai (metrics original scale par)
        ymu, ysd = float(ytr.mean()), float(ytr.std()) or 1.0
        ytr = (ytr - ymu) / ysd
    prepared = {**ds, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "ymu": ymu, "ysd": ysd,
                "K": len(ds["class_names"]) if ds["task"] == "classification" else 1, "rows": N}
    with _LOCK:
        if len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = prepared
    return prepared


def summary(name, n, target, task):
    """Dashboard preview ke liye."""
    nn = min(n, 5000) if name in SYNTH else n
    ds = load_dataset(name, nn, target, task)
    head = [[round(float(v), 3) for v in row] for row in ds["X"][:5]]
    out = {"rows": n if name in SYNTH else len(ds["y"]), "features": len(ds["names"]), "names": ds["names"],
           "task": ds["task"], "dropped": ds["dropped"], "head": head, "class_names": ds["class_names"],
           "target_head": [float(v) if ds["task"] == "regression" else ds["class_names"][int(v)] for v in ds["y"][:5]]}
    if ds["task"] == "classification":
        counts = np.bincount(ds["y"], minlength=len(ds["class_names"]))
        out["class_counts"] = counts.tolist()
    return out


# ------------------------------------------------------------------ metrics
def classification_metrics(y, pred, K):
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y, pred), 1)
    acc = float(np.trace(cm) / len(y))
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        p = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        r = tp / cm[k, :].sum() if cm[k, :].sum() else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "confusion": cm.tolist()}


def regression_metrics(y, pred):
    err = y - pred
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {"r2": 1 - float(np.sum(err ** 2)) / sst, "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err)))}
