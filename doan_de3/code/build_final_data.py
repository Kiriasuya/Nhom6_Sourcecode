"""Tao data/clean.csv cho baseline.py va train_dl.py tu cleanDATA.csv (chia theo nhom cau goc) + tap OOD.

Chay: python build_final_data.py --data cleanDATA.csv --manual new_data.csv --out data
- Giu nguyen cot split (train/val/test) cua cleanDATA.csv (chia theo `group`, khong ro ri cau goc).
- text := text_clean (ban sach); ban cu luu o text_raw.
- Cac mau manual (new_data.csv) duoc lam sach cung quy tac roi gan split = ood; kiem tra khong trung train/val/test.
"""
import argparse, os, re, sys, unicodedata
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--manual", default=None, help="new_data.csv (mau tu thu thap tay) -> tap OOD")
p.add_argument("--out", default="data")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

key = lambda s: re.sub(r"[^a-z0-9]+", " ", "".join(c for c in unicodedata.normalize("NFD", s.lower().replace("đ", "d")) if unicodedata.category(c) != "Mn")).strip()
df = pd.read_csv(args.data, encoding="utf-8-sig", keep_default_na=False)
df["text_raw"] = df["text"]; df["text"] = df["text_clean"].str.replace(r"\s+", " ", regex=True).str.strip()
df["topics"] = df.topics.str.replace(r"\s*;\s*", ";", regex=True)

if args.manual:  # lam sach cung quy tac co ban: NFC, chu thuong, bo khoang trang thua
    m = pd.read_csv(args.manual, encoding="utf-8-sig", keep_default_na=False)
    m["text_raw"] = m.text
    m["text"] = m.text.map(lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip())
    m.loc[m.topics.str.strip() == "", ["topics", "primary_topic"]] = "không xác định"
    m["topics"] = m.topics.str.replace(r"\s*;\s*", ";", regex=True)
    m["is_urgent"] = (m.priority == "cao").astype(int); m["split"] = "ood"
    seen = set(df.text.map(key)); dup = m.text.map(key).isin(seen)
    if dup.any(): print(f"Bo {int(dup.sum())} mau manual trung voi train/val/test"); m = m[~dup]
    df = pd.concat([df, m], ignore_index=True)

keep = ["id", "text", "text_raw", "topics", "primary_topic", "sentiment", "priority", "is_urgent", "source", "split"]
df = df[keep + [c for c in ("is_short", "no_diacritics", "group", "core") if c in df]]
df["id"] = range(1, len(df) + 1)
df.to_csv(f"{args.out}/clean.csv", index=False, encoding="utf-8-sig")
print(df.split.value_counts().to_dict())
print(pd.crosstab(df.primary_topic, df.split).to_string())
