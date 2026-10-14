"""Fine-tune PhoBERT (hoặc ViSoBERT) cho sentiment + priority.

Mặc định script chạy PhoBERT.  Nó luôn tôn trọng split đã có trong
``data/clean.csv``: train dùng để fit, val dùng để chọn checkpoint, test chỉ
được đánh giá một lần, còn ood chỉ dùng để tham khảo.  Checkpoint chứa đầy đủ
metadata để ``inference_predict.py`` có thể nạp lại mô hình và giải mã nhãn.

Ví dụ:
    python phobert_multitask_train.py
    python phobert_multitask_train.py --loss weighted --epochs 6
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_DIR / "data" / "clean.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "pretrained"
MODEL_PRESETS = {
    "phobert": {
        "model_name": "vinai/phobert-base",
        # PhoBERT pre-train trên văn bản đã word-segmented.
        "use_pyvi_segmenter": True,
    },
    "visobert": {
        "model_name": "uitnlp/visobert",
        # ViSoBERT dùng SentencePiece; giữ văn bản mạng xã hội ở dạng thô.
        "use_pyvi_segmenter": False,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_text_preprocessor(use_pyvi_segmenter: bool):
    """Trả về hàm tiền xử lý nhất quán giữa train và inference."""
    if not use_pyvi_segmenter:
        return lambda text: " ".join(str(text).split())

    try:
        from pyvi import ViTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PhoBERT cần word segmentation. Cài pyvi bằng `pip install pyvi`."
        ) from exc

    def segment(text: str) -> str:
        return ViTokenizer.tokenize(" ".join(str(text).split()))

    return segment


def read_splits(data_path: str):
    df = pd.read_csv(data_path, encoding="utf-8-sig", keep_default_na=False)
    required = {"id", "text", "sentiment", "priority", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    if (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("Có phản hồi rỗng trong cột text.")

    splits = {name: df[df["split"] == name].copy().reset_index(drop=True)
              for name in ("train", "val", "test", "ood")}
    for name in ("train", "val", "test"):
        if splits[name].empty:
            raise ValueError(f"Split {name!r} rỗng; không thể huấn luyện công bằng.")
    return splits


def build_label_maps(train_df: pd.DataFrame):
    """Chỉ fit mapping trên train để checkpoint có contract nhãn rõ ràng."""
    maps = {}
    for column in ("sentiment", "priority"):
        labels = sorted(train_df[column].astype(str).str.strip().unique().tolist())
        if len(labels) < 2:
            raise ValueError(f"{column} phải có ít nhất hai nhãn trong train.")
        maps[column] = {label: index for index, label in enumerate(labels)}
    return maps


def validate_labels(frame: pd.DataFrame, label_maps: dict, split_name: str) -> None:
    for column, mapping in label_maps.items():
        values = set(frame[column].astype(str).str.strip())
        unknown = sorted(values.difference(mapping))
        if unknown:
            raise ValueError(
                f"Split {split_name} có nhãn {column} chưa thấy trong train: {unknown}"
            )


class MultiTaskTextDataset(Dataset):
    def __init__(self, frame, tokenizer, text_preprocessor, max_len, label_maps=None):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_preprocessor = text_preprocessor
        self.max_len = max_len
        self.label_maps = label_maps

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        text = self.text_preprocessor(self.frame.at[index, "text"])
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.label_maps is not None:
            item["sentiment_label"] = torch.tensor(
                self.label_maps["sentiment"][str(self.frame.at[index, "sentiment"]).strip()],
                dtype=torch.long,
            )
            item["priority_label"] = torch.tensor(
                self.label_maps["priority"][str(self.frame.at[index, "priority"]).strip()],
                dtype=torch.long,
            )
        return item


class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name, n_sentiment, n_priority, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, n_sentiment)
        self.priority_head = nn.Linear(hidden_size, n_priority)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = self.dropout(output.last_hidden_state[:, 0, :])
        return {
            "sentiment": self.sentiment_head(sentence_embedding),
            "priority": self.priority_head(sentence_embedding),
        }


def class_weights(train_df, column, mapping, device):
    ids = train_df[column].astype(str).str.strip().map(mapping).to_numpy()
    counts = np.bincount(ids, minlength=len(mapping)).astype(np.float32)
    # Cân bằng nghịch đảo tần suất, trung bình trọng số bằng 1.
    weights = counts.sum() / (len(mapping) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_losses(args, train_df, label_maps, device):
    if args.loss == "ce":
        return nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    return (
        nn.CrossEntropyLoss(class_weights(train_df, "sentiment", label_maps["sentiment"], device)),
        nn.CrossEntropyLoss(class_weights(train_df, "priority", label_maps["priority"], device)),
    )


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    output = {"sentiment_pred": [], "priority_pred": [],
              "sentiment_prob": [], "priority_prob": [],
              "sentiment_true": [], "priority_true": []}
    has_labels = None
    for batch in loader:
        has_labels = "sentiment_label" in batch
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        for task in ("sentiment", "priority"):
            probabilities = torch.softmax(logits[task], dim=1)
            output[f"{task}_pred"].extend(probabilities.argmax(dim=1).cpu().tolist())
            output[f"{task}_prob"].extend(probabilities.max(dim=1).values.cpu().tolist())
            if has_labels:
                output[f"{task}_true"].extend(batch[f"{task}_label"].tolist())
    output["has_labels"] = bool(has_labels)
    return output


def task_metrics(y_true, y_pred, labels, include_urgent=False):
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        },
    }
    if include_urgent and "cao" in labels:
        urgent_id = labels.index("cao")
        precision, recall, _, _ = precision_recall_fscore_support(
            np.asarray(y_true) == urgent_id,
            np.asarray(y_pred) == urgent_id,
            average="binary",
            zero_division=0,
        )
        metrics["urgent_precision"] = float(precision)
        metrics["urgent_recall"] = float(recall)
    return metrics


def evaluate_predictions(predictions, label_maps):
    sentiment_labels = list(label_maps["sentiment"])
    priority_labels = list(label_maps["priority"])
    return {
        "n": len(predictions["sentiment_pred"]),
        "sentiment": task_metrics(
            predictions["sentiment_true"], predictions["sentiment_pred"], sentiment_labels
        ),
        "priority": task_metrics(
            predictions["priority_true"], predictions["priority_pred"], priority_labels,
            include_urgent=True,
        ),
    }


def prediction_frame(frame, predictions, label_maps):
    result_columns = [column for column in ("id", "text", "source", "split") if column in frame]
    result = frame[result_columns].copy()
    sentiment_labels = list(label_maps["sentiment"])
    priority_labels = list(label_maps["priority"])
    if predictions["has_labels"]:
        result["sentiment_true"] = frame["sentiment"].astype(str).values
        result["priority_true"] = frame["priority"].astype(str).values
    result["sentiment_pred"] = [sentiment_labels[i] for i in predictions["sentiment_pred"]]
    result["priority_pred"] = [priority_labels[i] for i in predictions["priority_pred"]]
    result["sentiment_confidence"] = predictions["sentiment_prob"]
    result["priority_confidence"] = predictions["priority_prob"]
    return result


@torch.no_grad()
def mean_latency_ms(model, tokenizer, text_preprocessor, texts, max_len, device, limit=200):
    if not texts:
        return None
    model.eval()
    timings = []
    for text in texts[:limit]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        encoded = tokenizer(
            text_preprocessor(text), truncation=True, max_length=max_len,
            padding="max_length", return_attention_mask=True, return_tensors="pt",
        )
        model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.mean(timings))


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    preset = MODEL_PRESETS[args.model_kind]
    model_name = args.model_name or preset["model_name"]
    use_segmenter = preset["use_pyvi_segmenter"] and not args.disable_word_segmentation

    splits = read_splits(args.data)
    label_maps = build_label_maps(splits["train"])
    for split_name, frame in splits.items():
        validate_labels(frame, label_maps, split_name)

    text_preprocessor = make_text_preprocessor(use_segmenter)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    datasets = {
        name: MultiTaskTextDataset(frame, tokenizer, text_preprocessor, args.max_len, label_maps)
        for name, frame in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    model = MultiTaskTransformer(
        model_name, len(label_maps["sentiment"]), len(label_maps["priority"]), args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sentiment_loss, priority_loss = make_losses(args, splits["train"], label_maps, device)

    best_score, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss_sentiment = sentiment_loss(logits["sentiment"], batch["sentiment_label"].to(device))
            loss_priority = priority_loss(logits["priority"], batch["priority_label"].to(device))
            loss = loss_sentiment + args.priority_loss_weight * loss_priority
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        val_predictions = collect_predictions(model, loaders["val"], device)
        val_metrics = evaluate_predictions(val_predictions, label_maps)
        val_score = (val_metrics["sentiment"]["macro_f1"] + val_metrics["priority"]["macro_f1"]) / 2
        record = {
            "epoch": epoch,
            "train_loss": round(total_loss / len(loaders["train"]), 6),
            "val_sentiment_macro_f1": round(val_metrics["sentiment"]["macro_f1"], 6),
            "val_priority_macro_f1": round(val_metrics["priority"]["macro_f1"], 6),
            "val_joint_macro_f1": round(val_score, 6),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_score > best_score:
            best_score, best_epoch, bad_epochs = val_score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping tại epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    test_predictions = collect_predictions(model, loaders["test"], device)
    ood_predictions = collect_predictions(model, loaders["ood"], device)
    test_metrics = evaluate_predictions(test_predictions, label_maps)
    ood_metrics = evaluate_predictions(ood_predictions, label_maps)
    test_metrics["latency_ms_mean"] = mean_latency_ms(
        model, tokenizer, text_preprocessor, splits["test"]["text"].tolist(), args.max_len, device
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_kind}_multitask_{args.loss}"
    checkpoint_path = output_dir / f"{tag}_best.pt"
    checkpoint = {
        "state": model.state_dict(),
        "model_kind": args.model_kind,
        "model_name": model_name,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "use_pyvi_segmenter": use_segmenter,
        "sentiment_labels": list(label_maps["sentiment"]),
        "priority_labels": list(label_maps["priority"]),
        "loss": args.loss,
        "priority_loss_weight": args.priority_loss_weight,
        "best_epoch": best_epoch,
        "best_val_joint_macro_f1": best_score,
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "architecture": "pretrained_transformer_multitask",
        "model_kind": args.model_kind,
        "model_name": model_name,
        "loss": args.loss,
        "train_n": len(splits["train"]),
        "val_n": len(splits["val"]),
        "best_epoch": best_epoch,
        "best_val_joint_macro_f1": best_score,
        "history": history,
        "test": test_metrics,
        "ood": ood_metrics,
    }
    (output_dir / f"{tag}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_frame(splits["test"], test_predictions, label_maps).to_csv(
        output_dir / f"{tag}_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    prediction_frame(splits["ood"], ood_predictions, label_maps).to_csv(
        output_dir / f"{tag}_ood_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Macro-F1 (sentiment/priority): "
          f"{test_metrics['sentiment']['macro_f1']:.4f} / {test_metrics['priority']['macro_f1']:.4f}")
    return metrics


def build_parser(default_model_kind="phobert"):
    parser = argparse.ArgumentParser(description="Fine-tune Transformer đa tác vụ sentiment + priority")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-kind", choices=MODEL_PRESETS, default=default_model_kind)
    parser.add_argument("--model-name", default=None, help="Override Hugging Face model id.")
    parser.add_argument("--disable-word-segmentation", action="store_true")
    parser.add_argument("--loss", choices=("ce", "weighted"), default="weighted")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--priority-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(default_model_kind="phobert"):
    args = build_parser(default_model_kind).parse_args()
    train(args)


if __name__ == "__main__":
    main()
