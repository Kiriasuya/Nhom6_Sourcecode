#Tiền xử lý dữ liệu
python code/build_final_data.py --data data/cleanDATA.csv --out data

#Baseline
python code/baseline.py --data data/clean.csv --out results/baseline

#BiLSTM
python code/train_dl.py --data data/clean.csv --topic_col primary_topic --out checkpoints/best_bilstm --checkpoint checkpoints/best_bilstm/best_bilstm.pt

#PhoBERT
python code/phobert_multitask_train.py --data data/clean.csv --output checkpoints

#VisoBERT
python code/visobert_multitask_train.py

#Đánh giá mô hình
python code/evaluate.py --pred_dir preds --out results/eval

#Dự đoán câu mới
#PhoBERT
python code/inference_predict.py --checkpoint checkpoints/best_phobert/phobert_multitask_weighted_best.pt --split test
#VisoBERT
python code/inference_predict.py --checkpoint checkpoints/best_visobert/visobert_multitask_weighted_best.pt --split test
