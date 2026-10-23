"""
evaluate.py: Đánh giá kết quả các mô hình cho đồ án
Chạy file: python evalute.py --pred_dir preds --out results/eval
"""


import argparse
import glob
import json
import os
import re
import time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

#Thiết lập các hằng số 
#Chỉ định 3 tập dữ liệu cần đánh giá
SPLITS = ("val", "test", "ood")

#Định nghĩa cột thực tế - dự đoán cho 3 bài toán đơn nhãn: topic, sentiment, priority
TASKS = {"topic": ("y_true_topic", "y_pred_topic"), 
         "sentiment": ("y_true_sent", "y_pred_sent"),
         "priority": ("y_true_prio", "y_pred_prio")}

ML = ("y_true_ml", "y_pred_ml")     #Khai báo cột cho bài toán đa nhãn

DIAC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]")

URGENT = "cao"  #Đặt mốc cao nhất để tính điểm khẩn cấp

#Nạp và căn chỉnh dữ liệu nạp vào
def load_preds (pred_dir):
    P = {}
    list_file = glob.glob(pred_dir + "/*.csv")
    for f in (list_file):
        file_name = os.path.basename(f)
        file_name = file_name.replace(".csv", "")

        #Bỏ qua nếu tên file không có dấu gạch dưới
        if "_" not in file_name:
            continue
        
        parts = file_name.split("_")
        split = parts[-1]
        model = "_".join(parts[:-1])

        if split not in SPLITS:
            continue

        df = pd.read_csv(f, encoding="utf-8-sig", keep_default_na=False)
        
        if "id" not in df.columns:
            print(f"[Warning] {f} thiếu cột id")
            continue
        df["id"] = df["id"].astype(int)
        if not df.id.is_unique:
            print(f"[Warning] {f} có id trùng lặp")
        P.setdefault(model, {})[split] = df
    return P

def has_cols(df, cols):
    return all(c in df.columns for c in cols) and (df[cols[0]].astype(str).str.strip() != "").any

def to_sets(col):
    result = []
    for s in col:
        chuoi = str(s)
        arr = chuoi.split(";")
        tap_hop = set()
        for x in arr:
            x_strip = x.strip()
            if x_strip != "":
                tap_hop.add(x_strip)
        result.append(tap_hop)
    return result

def align(P, split, intersect=True):
    models_split = []
    for m in P.keys():
        if split in P[m]:
            models_split.append(m)
    
    if len(models_split) == 0:
        return {}
    
    ids_chung = set(P[models_split[0]][split]["id"])
    for m in models_split:
        tap_id = set(P[m][split]["id"])
        ids_chung = ids_chung.intersection(tap_id)
    
    out = {}
    for m in models_split:
        df = P[m][split]
        df_new = df[df["id"].isin(ids_chung)]
        df_new = df_new.sort_values(by="id")
        df_new = df_new.reset_index(drop = True)
        out[m] = df_new
    
    print("Xử lý xong tập " + split + ", số dòng chung là " + str(len(ids_chung)))
    return out


#Các thước đo
def mc_metrics(yt, yp):
    yt_list = list(yt)
    yp_list = list(yp)

    nhan_thuc_te = []
    for label in yt_list:
        if str(label) not in nhan_thuc_te:
            nhan_thuc_te.append(str(label))
    nhan_thuc_te.sort()

    macro = f1_score(yt_list, yp_list, labels = nhan_thuc_te, average="macro", zero_division=0)
    micro = f1_score(yt_list, yp_list, average="micro", zero_division=0)
    weighted = f1_score(yt_list, yp_list, average="weighted", zero_division=0)

    return {
        "n": len(yt_list),
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted
    }

def urgent_metrics(yt, yp):
    tp = 0
    fp = 0
    fn = 0
    yt_list = list(yt)
    yp_list = list(yp)
    for i in range(len(yt_list)):
        thuc_te = str(yt_list[i])
        du_doan = str(yp_list[i])
        if thuc_te == URGENT and du_doan == URGENT:
            tp = tp + 1
        elif thuc_te != URGENT and du_doan == URGENT:
            fp = fp + 1
        elif thuc_te == URGENT and du_doan != URGENT:
            fn = fn + 1
    #Tính precision
    if (tp+fp) == 0:
        P = 0.0
    else:
        P = tp / (tp + fp)
    
    #Tính recall
    if (tp + fn) == 0:
        R = 0.0
    else:
        R = tp / (tp + fn)
    
    #Tính F1
    if (P + R) == 0:
        F = 0.0
    else:
        F = (2 * P * R) / (P + R)
    
    return {
        "urgent_precision": P,
        "urgent_recall": R,
        "urgent_f1": F,
        "urgent_tp": tp,
        "urgent_fp": fp,
        "urgent_fn": fn,
        "urgent_support": tp + fn
    }

def ml_metrics(yt_sets, yp_sets):
    all_label = set()
    for s in yt_sets:
        all_label.update(s)
    for s in yp_sets:
        all_label.update(s)
    
    labels = list(all_label)
    labels.sort()

    mlb = MultiLabelBinarizer(classes=labels)
    Yt = mlb.fit_transform(yt_sets)
    Yp = mlb.transform(yp_sets)

    macro = f1_score(Yt, Yp, average="macro", zero_division=0)
    micro = f1_score(Yt, Yp, average="micro", zero_division=0)
    weighted = f1_score(Yt, Yp, average="weighted", zero_division=0)

    return {
        "n": len(Yt),
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted
    }


#Hàm phân tích lỗi
def error_analysis(test_predictions, best_topic_model, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confusion_rows = []
    group_rows = []
    urgent_rows = []

    for model, df in test_predictions.items():
        for task, (true_col, pred_col) in TASKS.items():
            if true_col not in df.columns or pred_col not in df.columns:
                continue

            valid = (
                df[true_col].astype(str).str.strip().ne("")
                & df[pred_col].astype(str).str.strip().ne("")
            )
            task_df = df.loc[valid].copy()

            if task_df.empty:
                continue

            wrong = (
                task_df[true_col].astype(str)
                != task_df[pred_col].astype(str)
            )

            # Top cặp nhãn bị nhầm, kèm một câu ví dụ
            errors = task_df.loc[wrong]
            counts = Counter(zip(
                errors[true_col].astype(str),
                errors[pred_col].astype(str),
            ))

            for (actual, predicted), count in counts.most_common(5):
                example = errors[
                    errors[true_col].astype(str).eq(actual)
                    & errors[pred_col].astype(str).eq(predicted)
                ].iloc[0]

                confusion_rows.append({
                    "model": model,
                    "task": task,
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "example_id": example.get("id", ""),
                    "example_text": example.get("text", ""),
                })

            # Các nhóm lỗi: câu ngắn, không dấu, đa chủ đề, khẩn cấp
            text = (
                task_df["text"].fillna("").astype(str)
                if "text" in task_df.columns
                else pd.Series("", index=task_df.index)
            )

            if "n_words" in task_df.columns:
                word_count = pd.to_numeric(
                    task_df["n_words"], errors="coerce"
                ).fillna(text.str.split().str.len())
            else:
                word_count = text.str.split().str.len()

            group_masks = {
                "short_le_3_words": word_count <= 3,
            }

            if "has_accent" in task_df.columns:
                has_accent = task_df["has_accent"].astype(str).str.lower()
                group_masks["no_diacritics"] = has_accent.isin(
                    {"false", "0", "no"}
                )
            else:
                group_masks["no_diacritics"] = ~text.str.contains(
                    DIAC, na=False
                )

            if "y_true_ml" in task_df.columns:
                group_masks["multiple_topics"] = task_df["y_true_ml"].map(
                    lambda value: len(to_sets([value])[0]) > 1
                )

            if task == "priority":
                group_masks["urgent_priority"] = (
                    task_df[true_col].astype(str).str.lower().eq("cao")
                )

                # Các mẫu khẩn cấp bị bỏ sót
                missed = task_df[
                    task_df[true_col].astype(str).str.lower().eq("cao")
                    & task_df[pred_col].astype(str).str.lower().ne("cao")
                ]
                for _, row in missed.iterrows():
                    urgent_rows.append({
                        "model": model,
                        "id": row.get("id", ""),
                        "text": row.get("text", ""),
                        "actual_priority": row[true_col],
                        "predicted_priority": row[pred_col],
                    })

            for group_name, mask in group_masks.items():
                count = int(mask.sum())
                if count == 0:
                    continue

                group_wrong = wrong.loc[mask]
                group_rows.append({
                    "model": model,
                    "task": task,
                    "group": group_name,
                    "n": count,
                    "errors": int(group_wrong.sum()),
                    "error_rate": float(group_wrong.mean()),
                })

    pd.DataFrame(confusion_rows, columns=[
        "model", "task", "actual", "predicted", "count",
        "example_id", "example_text",
    ]).to_csv(
        out_dir / "confusions_top.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(group_rows, columns=[
        "model", "task", "group", "n", "errors", "error_rate",
    ]).to_csv(
        out_dir / "error_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(urgent_rows, columns=[
        "model", "id", "text", "actual_priority", "predicted_priority",
    ]).to_csv(
        out_dir / "urgent_missed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Confusion matrix topic của model tốt nhất trên validation
    if best_topic_model is not None:
        df = test_predictions[best_topic_model]
        true_col, pred_col = TASKS["topic"]

        if true_col in df.columns and pred_col in df.columns:
            valid = (
                df[true_col].astype(str).str.strip().ne("")
                & df[pred_col].astype(str).str.strip().ne("")
            )
            topic_df = df.loc[valid]

            if not topic_df.empty:
                labels = sorted(
                    set(topic_df[true_col].astype(str))
                    | set(topic_df[pred_col].astype(str))
                )
                matrix = confusion_matrix(
                    topic_df[true_col].astype(str),
                    topic_df[pred_col].astype(str),
                    labels=labels,
                )

                fig, ax = plt.subplots(
                    figsize=(max(7, len(labels) * 0.7), max(6, len(labels) * 0.6))
                )
                image = ax.imshow(matrix, cmap="Blues")
                fig.colorbar(image, ax=ax)
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)), labels)
                ax.set_xlabel("Dự đoán")
                ax.set_ylabel("Thực tế")
                ax.set_title(f"Confusion topic - {best_topic_model}")
                fig.tight_layout()
                fig.savefig(
                    out_dir / "confusion_topic_best.png",
                    dpi=150,
                )
                plt.close(fig)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["baseline", "bilstm", "phobert", "visobert"])
    parser.add_argument("--data", default="data/clean.csv")
    parser.add_argument("--pred_dir", default="preds")
    parser.add_argument("--out", default="results/eval_m4")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    P = load_preds(args.pred_dir)
    if len(P) == 0:
        print("Không tìm thấy file kết quả nào")
        return
    
    A = {}
    for s in SPLITS:
        A[s] = align(P,s)

    rows = []
    for s in SPLITS:
        data_table = A[s]
        for model in data_table.keys():
            df = data_table[model]

            for t, (tc, pc) in TASKS.items():
                if tc in df.columns and pc in df.columns:
                    score = mc_metrics(df[tc], df[pc])
                    score["model"] = s
                    score["task"] = t

                    if t == "priority":
                        score_khan_cap = urgent_metrics(df[tc], df[pc])
                        score.update(score_khan_cap)

                    rows.append(score)
            
            if ML[0] in df.columns and ML[1] in df.columns:
                score_ml = ml_metrics(to_sets(df[ML[0]]), to_sets(df[ML[1]]))
                score_ml["model"] = model
                score_ml["split"] = s
                score_ml["task"] = "multilabel"
                rows.append(score_ml)
            
    res = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res.to_csv(
        out_dir / "results_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Đã lưu kết quả chung vào " + args.out + "/results_eval.csv")

    best_topic_model = None
    best_topic_score = -1.0

    for model, frame in A["val"].items():
        true_col, pred_col = TASKS["topic"]

        if true_col not in frame.columns or pred_col not in frame.columns:
            continue

        score = mc_metrics(frame[true_col], frame[pred_col])["macro_f1"]
        if score > best_topic_score:
            best_topic_score = score
            best_topic_model = model

    error_analysis(A["test"], best_topic_model, out_dir)

if __name__ == "__main__":
    main()