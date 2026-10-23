"""BiLSTM cho phan loai phan hoi sinh vien (multiclass / multilabel) + tinh chinh hyperparameter.
Du lieu CSV, cot:
  text        noi dung phan hoi (bat buoc)
  topic       nhan don vi tiep nhan. Multiclass: 1 nhan. Multilabel: nhieu nhan cach nhau boi ';'
  sentiment   (tuy chon) nhan cam xuc -> head phu (multi-task)
  priority    (tuy chon) thap/trung binh/cao -> head phu (multi-task)
  split       (tuy chon) train/val/test. Neu thieu: chia 80/10/10 phan tang theo seed
Chay:
  python train_dl.py --data data.csv --topic_col primary_topic --mode multiclass --loss weighted --tune
  python train_dl.py --data data.csv --topic_col topics --mode multilabel --loss focal
"""
import argparse, itertools, json, random, re, time, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

DEFAULT_PRED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "preds",
)


def save_predictions(frame, model, split, pred_dir):
    os.makedirs(pred_dir, exist_ok=True)
    output_path = os.path.join(pred_dir, f"{model}_{split}.csv")
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def update_latency(model, latency_ms, pred_dir):
    os.makedirs(pred_dir, exist_ok=True)
    output_path = os.path.join(pred_dir, "latency.json")
    measurements = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            measurements = json.load(file)
        if not isinstance(measurements, dict):
            raise ValueError(f"Định dạng latency không hợp lệ: {output_path}")
    measurements[model] = None if latency_ms is None else float(latency_ms)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(measurements, file, ensure_ascii=False, indent=2)


p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--topic_col", default="topic", help="cot nhan don vi (vd: primary_topic cho multiclass, topics cho multilabel)")
p.add_argument("--mode", choices=["multiclass", "multilabel"], default="multiclass")
p.add_argument("--loss", choices=["ce", "weighted", "focal"], default="weighted")
p.add_argument("--tune", action="store_true", help="tim hyperparameter tren val Macro-F1")
p.add_argument("--n_trials", type=int, default=8, help="so cau hinh lay ngau nhien tu luoi khi --tune")
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--aux_weight", type=float, default=0.5, help="trong so loss cua head phu")
p.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default=os.path.join("checkpoints", "best_bilstm"))
p.add_argument(
    "--checkpoint",
    default=os.path.join("checkpoints", "best_bilstm", "best_bilstm.pt"),
)
p.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)

# Data
df = pd.read_csv(args.data).rename(columns={args.topic_col: "topic"}).dropna(subset=["text", "topic"])
if "split" in df:
    unknown_splits = set(df["split"].astype(str)).difference({"train", "val", "test", "ood"})
    if unknown_splits:
        raise ValueError(f"Giá trị split không hợp lệ: {sorted(unknown_splits)}")
df = df[~df.text.str.lower().str.strip().duplicated()].reset_index(drop=True)  # khu trung lap sau chuan hoa
df["topic"] = df.topic.astype(str).str.replace(r"\s*;\s*", ";", regex=True).str.strip()
if "split" not in df:  # chia sau khi khu trung lap de tranh ro ri du lieu
    first = df.topic.str.split(";").str[0]; first = first.where(first.map(first.value_counts()) >= 10, "_rare")  # gop lop qua hiem de stratify
    tr, rest = train_test_split(df.index, test_size=0.2, stratify=first, random_state=args.seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=first[rest], random_state=args.seed)
    df["split"] = "train"; df.loc[va, "split"] = "val"; df.loc[te, "split"] = "test"
for split in ("train", "val", "test"):
    if not (df["split"] == split).any():
        raise ValueError(f"Split {split!r} rỗng; không thể huấn luyện/đánh giá.")
tok = lambda s: re.findall(r"\w+", str(s).lower())  # muc am tiet
vocab = {"<pad>": 0, "<unk>": 1}
for w, c in Counter(w for s in df[df.split == "train"].text for w in tok(s)).items():
    if c >= 2: vocab[w] = len(vocab)
MAXLEN = 64
def encode(texts):
    x = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tok(s)][:MAXLEN]; x[i, :len(ids)] = ids
    return torch.tensor(x)
topics = sorted({t for s in df.topic.astype(str) for t in s.split(";")})
tix = {t: i for i, t in enumerate(topics)}
def topic_y(col):
    if args.mode == "multiclass": return torch.tensor([tix[str(s)] for s in col])
    y = np.zeros((len(col), len(topics)), dtype=np.float32)
    for i, s in enumerate(col):
        for t in str(s).split(";"): y[i, tix[t]] = 1
    return torch.tensor(y)
aux = {}  # head phu: ten -> danh sach lop
for name in ("sentiment", "priority"):
    if name in df: aux[name] = sorted(df[name].dropna().astype(str).unique())

def build(split):
    d = df[df.split == split]
    y = {"topic": topic_y(d.topic)}
    for n, cls in aux.items():
        y[n] = torch.tensor([cls.index(str(v)) if pd.notna(v) else -100 for v in d[n]])
    return encode(d.text.tolist()), y, d
data = {s: build(s) for s in ("train", "val", "test", "ood")}

# Model
class Net(nn.Module):
    def __init__(self, emb, hid, drop):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb, padding_idx=0)
        self.enc = nn.LSTM(emb, hid, batch_first=True, bidirectional=True); d = 4 * hid  # max + mean pool
        self.drop = nn.Dropout(drop)
        self.heads = nn.ModuleDict({"topic": nn.Linear(d, len(topics)), **{n: nn.Linear(d, len(c)) for n, c in aux.items()}})
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1); e = self.drop(self.emb(x))
        h, _ = self.enc(e)
        mx = h.masked_fill(~mask, -1e9).max(1).values
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        z = self.drop(torch.cat([mx, mean], -1))
        return {n: h(z) for n, h in self.heads.items()}

def class_weights(y, n):  # nghich dao tan suat, chuan hoa trung binh = 1
    cnt = torch.bincount(y[y >= 0], minlength=n).float().clamp(min=1)
    return (cnt.sum() / (n * cnt)).to(dev)
def focal(logits, y, w, gamma):  # FL = -alpha_t (1 - p_t)^gamma log p_t
    valid = y >= 0
    if not valid.any():
        return logits.sum() * 0
    y = y[valid]
    logp = F.log_softmax(logits[valid], -1).gather(1, y[:, None]).squeeze(1)
    a = w[y] if w is not None else 1.0
    return (-a * (1 - logp.exp()) ** gamma * logp).mean()
def make_loss(ytr):
    w = {"topic": None}; pos = None
    if args.mode == "multiclass":
        if args.loss != "ce": w["topic"] = class_weights(ytr["topic"], len(topics))
    else:  # multilabel: pos_weight = neg / pos
        pos_cnt = ytr["topic"].sum(0).clamp(min=1); pos = ((len(ytr["topic"]) - pos_cnt) / pos_cnt).to(dev) if args.loss != "ce" else None
    for n, cls in aux.items():
        w[n] = class_weights(ytr[n], len(cls)) if args.loss != "ce" else None
    def loss(out, y):
        if args.mode == "multiclass":
            lt = focal(out["topic"], y["topic"], w["topic"], args.gamma) if args.loss == "focal" else F.cross_entropy(out["topic"], y["topic"], weight=w["topic"])
        else:
            bce = F.binary_cross_entropy_with_logits(out["topic"], y["topic"], pos_weight=pos, reduction="none")
            if args.loss == "focal":
                pt = torch.exp(-F.binary_cross_entropy_with_logits(out["topic"], y["topic"], reduction="none")); bce = (1 - pt) ** args.gamma * bce
            lt = bce.mean()
        la = sum((focal(out[n], y[n], w[n], args.gamma) if args.loss == "focal" else F.cross_entropy(out[n], y[n], weight=w[n], ignore_index=-100)) for n in aux if (y[n] >= 0).any())
        return lt + args.aux_weight * la
    return loss

# Train / danh gia
def to_dev(y): return {k: v.to(dev) for k, v in y.items()}
@torch.no_grad()
def predict(model, x, bs=256):
    model.eval(); outs = {}
    for i in range(0, len(x), bs):
        for k, v in model(x[i:i + bs].to(dev)).items(): outs.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v) for k, v in outs.items()}
def topic_pred(logits, thr):
    return logits.argmax(1).numpy() if args.mode == "multiclass" else (torch.sigmoid(logits) > thr).int().numpy()
def best_thr(logits, y):  # multilabel: chon nguong toan cuc tren val
    return max(np.arange(0.2, 0.75, 0.05), key=lambda t: f1_score(y.numpy(), topic_pred(logits, t), average="macro", zero_division=0))
def val_score(model, thr=0.5):
    x, y, _ = data["val"]; lg = predict(model, x)["topic"]
    t = best_thr(lg, y["topic"]) if args.mode == "multilabel" else thr
    return f1_score(y["topic"].numpy(), topic_pred(lg, t), average="macro", zero_division=0), t
def train(cfg):
    torch.manual_seed(args.seed)
    xtr, ytr, _ = data["train"]; model = Net(cfg["emb"], cfg["hid"], cfg["drop"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    loss_fn = make_loss(ytr); best = (-1, None, 0.5); bad = 0
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            loss = loss_fn(model(xtr[idx].to(dev)), to_dev({k: v[idx] for k, v in ytr.items()}))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        f1, thr = val_score(model)
        if f1 > best[0]: best = (f1, {k: v.clone() for k, v in model.state_dict().items()}, thr); bad = 0
        else:
            bad += 1
            if bad >= args.patience: break
    model.load_state_dict(best[1]); return model, best[0], best[2]

base = {"emb": 128, "hid": 128, "drop": 0.4, "lr": 2e-3, "bs": 32}
grid = {"emb": [100, 200], "hid": [64, 128], "drop": [0.3, 0.5], "lr": [1e-3, 3e-3]} if args.tune else {}
cfgs = [dict(base, **dict(zip(grid, v))) for v in itertools.product(*grid.values())] if grid else [base]
cfgs = random.Random(args.seed).sample(cfgs, min(args.n_trials, len(cfgs)))  # random search tren luoi 16 cau hinh
trials, best = [], None
for cfg in cfgs:
    t0 = time.time(); m, f1, thr = train(cfg); trials.append({**cfg, "val_macro_f1": round(f1, 4), "sec": round(time.time() - t0, 1)})
    print(trials[-1])
    if best is None or f1 > best[1]: best = (m, f1, thr, cfg)
model, val_f1, thr, cfg = best

# Danh gia test va ood su dung du lieu thu thap tay, ngoai phan phoi
def evaluate(split):
    x, y, _ = data[split]
    if len(x) == 0: return None, None
    out = predict(model, x); yt = y["topic"].numpy(); pt = topic_pred(out["topic"], thr)
    r = {"n": len(x), **{f"{a}_f1": f1_score(yt, pt, average=a, zero_division=0) for a in ("macro", "micro", "weighted")},
         "per_class_f1": dict(zip(topics, f1_score(yt, pt, average=None, zero_division=0).round(4).tolist()))}
    for n, cls in aux.items():
        ya, pa = y[n].numpy(), out[n].argmax(1).numpy(); ok = ya >= 0
        r[f"{n}_macro_f1"] = f1_score(ya[ok], pa[ok], average="macro", zero_division=0)
        if n == "priority" and "cao" in cls:  # precision/recall lop khan cap
            pr, rc, _, _ = precision_recall_fscore_support(ya[ok] == cls.index("cao"), pa[ok] == cls.index("cao"), average="binary", zero_division=0)
            r["urgent_precision"], r["urgent_recall"] = pr, rc
    return r, (yt, pt)
res = {"arch": "bilstm", "mode": args.mode, "loss": args.loss, "best_cfg": cfg, "val_macro_f1": val_f1}
res["test"], test_predictions = evaluate("test")
if test_predictions is None:
    raise ValueError("Split test rỗng; không thể tạo kết quả đánh giá.")
yt, pt = test_predictions
res["ood"], _ = evaluate("ood")
if args.mode == "multiclass":
    cm = confusion_matrix(yt, pt, labels=range(len(topics))); res["confusion_matrix"] = {"labels": topics, "matrix": cm.tolist()}
model.eval(); lat = []; texts = data["test"][2].text.tolist()
with torch.no_grad():  # do tre tung mau: token hoa + ma hoa + suy luan
    for t in texts[:200]:
        t0 = time.perf_counter(); model(encode([t]).to(dev)); lat.append((time.perf_counter() - t0) * 1000)
res["latency_ms_mean"] = float(np.mean(lat)) if lat else None
res["n_params"] = sum(p.numel() for p in model.parameters()); res["trials"] = trials
tag = f"bilstm_{args.mode}_{args.loss}"
with open(f"{args.out}/{tag}.json", "w", encoding="utf-8") as file:
    json.dump(res, file, ensure_ascii=False, indent=2)
checkpoint = {
    "state": model.state_dict(), "vocab": vocab, "topics": topics, "aux": aux,
    "cfg": cfg, "thr": thr, "arch": "bilstm", "mode": args.mode,
}
os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
torch.save(checkpoint, args.checkpoint)

metadata_columns = [
    column for column in ("id", "text", "source", "split", "group", "has_accent",
                          "n_words", "is_short", "is_urgent")
    if column in df.columns
]
for split, (_, targets, split_frame) in data.items():
    prediction_frame = split_frame[metadata_columns].copy()
    if len(split_frame):
        outputs = predict(model, data[split][0])
        topic_predictions = topic_pred(outputs["topic"], thr)
        if args.mode == "multiclass":
            prediction_frame["y_true_topic"] = split_frame["topic"].astype(str).values
            prediction_frame["y_pred_topic"] = [topics[index] for index in topic_predictions]
        else:
            true_labels = [str(value).split(";") for value in split_frame["topic"]]
            predicted_labels = [
                [topics[index] for index in np.flatnonzero(row)]
                for row in topic_predictions
            ]
            prediction_frame["y_true_ml"] = [";".join(labels) for labels in true_labels]
            prediction_frame["y_pred_ml"] = [";".join(labels) for labels in predicted_labels]
            if "primary_topic" in split_frame.columns:
                prediction_frame["y_true_topic"] = split_frame["primary_topic"].astype(str).values
                top_indices = outputs["topic"].argmax(1).numpy()
                prediction_frame["y_pred_topic"] = [topics[index] for index in top_indices]
        for name, classes in aux.items():
            true_values = split_frame[name]
            valid = true_values.notna().to_numpy()
            if valid.any():
                true_column = "y_true_sent" if name == "sentiment" else "y_true_prio"
                pred_column = "y_pred_sent" if name == "sentiment" else "y_pred_prio"

                prediction_frame[true_column] = true_values.fillna("").astype(str).values
                predicted_indices = outputs[name].argmax(1).numpy()
                prediction_frame[pred_column] = [
                    classes[index] for index in predicted_indices
                ]
    save_predictions(prediction_frame, "bilstm", split, args.pred_dir)
update_latency("bilstm", res["latency_ms_mean"], args.pred_dir)
if args.mode == "multiclass":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(topics)), topics, rotation=45, ha="right"); ax.set_yticks(range(len(topics)), topics)
    for i, j in itertools.product(range(len(topics)), repeat=2): ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); plt.tight_layout(); plt.savefig(f"{args.out}/{tag}_cm.png", dpi=150)
print(json.dumps({k: v for k, v in res.items() if k not in ("trials", "confusion_matrix")}, ensure_ascii=False, indent=2, default=float))
