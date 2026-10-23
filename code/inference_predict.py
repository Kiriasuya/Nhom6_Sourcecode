"""Nạp checkpoint PhoBERT/ViSoBERT đa tác vụ và xuất dự đoán có nhãn chữ.

Ví dụ:
    python inference_predict.py --checkpoint ../results/pretrained/phobert_multitask_weighted_best.pt
    python inference_predict.py --checkpoint ../results/pretrained/visobert_multitask_weighted_best.pt --split ood
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from phobert_multitask_train import (
    DEFAULT_DATA,
    MultiTaskTextDataset,
    MultiTaskTransformer,
    collect_predictions,
    evaluate_predictions,
    make_text_preprocessor,
    mean_latency_ms,
    prediction_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference checkpoint Transformer đa tác vụ")
    parser.add_argument("--checkpoint", required=True, help="File *_best.pt do script train tạo.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV có cột text; nhãn là tùy chọn.")
    parser.add_argument("--split", choices=("test", "val", "ood", "train", "all"), default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required = {"state", "model_name", "max_len", "dropout", "sentiment_labels", "priority_labels"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint không đúng định dạng multitask: thiếu {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    frame = pd.read_csv(args.data, encoding="utf-8-sig", keep_default_na=False)
    if "text" not in frame:
        raise ValueError("CSV inference cần cột text.")
    if args.split != "all":
        if "split" not in frame:
            raise ValueError("CSV không có cột split; dùng --split all để dự đoán toàn bộ.")
        frame = frame[frame["split"] == args.split].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("Không có mẫu nào thỏa split được chọn.")

    label_maps = {
        "sentiment": {label: index for index, label in enumerate(checkpoint["sentiment_labels"])},
        "priority": {label: index for index, label in enumerate(checkpoint["priority_labels"])},
    }
    labeled = {"sentiment", "priority"}.issubset(frame.columns)
    if labeled:
        for task, mapping in label_maps.items():
            unknown = set(frame[task].astype(str).str.strip()).difference(mapping)
            if unknown:
                raise ValueError(f"Nhãn {task} không có trong checkpoint: {sorted(unknown)}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"], use_fast=False)
    text_preprocessor = make_text_preprocessor(checkpoint.get("use_pyvi_segmenter", False))
    dataset = MultiTaskTextDataset(
        frame, tokenizer, text_preprocessor, checkpoint["max_len"], label_maps if labeled else None
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = MultiTaskTransformer(
        checkpoint["model_name"],
        len(checkpoint["sentiment_labels"]),
        len(checkpoint["priority_labels"]),
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state"])

    predictions = collect_predictions(model, loader, device)
    output_frame = prediction_frame(frame, predictions, label_maps)
    output_path = Path(args.output) if args.output else checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{args.split}_predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    result = {
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint["model_name"],
        "split": args.split,
        "n": len(frame),
        "predictions_file": str(output_path),
        "latency_ms_mean": mean_latency_ms(
            model, tokenizer, text_preprocessor, frame["text"].tolist(), checkpoint["max_len"], device
        ),
    }
    if labeled:
        result["metrics"] = evaluate_predictions(predictions, label_maps)
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
