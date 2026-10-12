# Đề tài 3 – Phân loại & định tuyến phản hồi sinh viên (phần của Noah: mô hình BiLSTM + paper)

## Cấu trúc
- code/     build_final_data.py, train_dl.py (BiLSTM), baseline.py (TF-IDF + LR/SVM, chỉ để so sánh), requirements.txt
- data/     cleanDATA.csv (nhóm làm sạch + chia theo nhóm câu gốc), new_data.csv (14 mẫu thu thập tay -> OOD), clean.csv (đầu vào cho mô hình)
- results/  results_summary.csv (baseline + BiLSTM), deep/ (JSON + confusion matrix BiLSTM), baseline/
- paper/    Introduction_va_Kien_truc_DL.docx (Introduction + Methodology 3.2), fig_architecture.png, paper_sections.md

## Chạy lại (từ thư mục code/)
    pip install -r requirements.txt
    python build_final_data.py --data ../data/cleanDATA.csv --manual ../data/new_data.csv --out ../data
    python baseline.py --data ../data/clean.csv --out ../results/baseline
    python train_dl.py --data ../data/clean.csv --topic_col primary_topic --mode multiclass --loss weighted --tune
    python train_dl.py --data ../data/clean.csv --topic_col topics        --mode multilabel --loss weighted --tune
(--loss ce | weighted | focal để so sánh xử lý mất cân bằng; seed 42; kết quả vào runs/)

## Số liệu cần khớp
cleanDATA.csv: 2106 dòng (train 1502 / val 309 / test 295; 220 nhóm câu gốc không nằm ở hai tập) + 14 mẫu manual = OOD.

## Lưu ý khi viết kết quả
- Test không có mẫu đa nhãn (140 ở train, 1 ở val) -> kết quả multilabel chỉ phản ánh dự đoán đúng 1 nhãn.
- Lớp "không xác định" không có trong test; OOD chỉ 14 mẫu; một seed -> chênh lệch nhỏ giữa các mô hình nằm trong sai số.
