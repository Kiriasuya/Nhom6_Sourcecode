"""Entry point fine-tune ViSoBERT cho sentiment + priority.

Logic huấn luyện nằm trong phobert_multitask_train.py để hai pretrained model
dùng hoàn toàn cùng protocol dữ liệu, metric và định dạng artifact.
"""
from phobert_multitask_train import main


if __name__ == "__main__":
    main(default_model_kind="visobert")
