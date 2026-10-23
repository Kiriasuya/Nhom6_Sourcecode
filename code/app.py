
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from scipy.sparse import hstack

st.set_page_config(
    page_title="Feedback Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard phân tích phản hồi")
st.caption("Theo dõi phản hồi, dự đoán AI và cảnh báo khẩn cấp")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

MODEL_PATH = PROJECT_DIR / "code" / "outputs_baseline" / "dashboard_models.joblib"
CORRECTION_PATH = DATA_DIR / "corrected_feedback.csv"

DATA_PATHS = [
    DATA_DIR / "clean.csv",
    DATA_DIR / "cleanDATA.csv",
]

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


# 1. Đọc dữ liệu và giữ các nhãn đã chỉnh sửa
if CORRECTION_PATH.exists():
    df = load_data(str(CORRECTION_PATH))
    st.info("Đang đọc dữ liệu đã hiệu chỉnh: corrected_feedback.csv")
else:
    df = None
    for path in DATA_PATHS:
        if path.exists():
            try:
                df = load_data(str(path))
                st.info(f"Đang đọc dữ liệu từ: {path.name}")
                break
            except Exception as e:
                st.error(f"Không đọc được {path.name}: {e}")

if df is None:
    st.error("Không tìm thấy file dữ liệu CSV.")
    st.stop()

if df.empty:
    st.warning("File dữ liệu hiện không có phản hồi.")
    st.stop()


# 2. Nạp mô hình
artifact = None

if MODEL_PATH.exists():
    try:
        artifact = load_artifact(str(MODEL_PATH))
    except Exception as e:
        st.error(f"Không tải được mô hình: {e}")
else:
    st.warning(
        "Chưa tìm thấy outputs_baseline/dashboard_models.joblib. "
        "Dashboard vẫn có thể hiển thị dữ liệu nhưng chưa dự đoán AI."
    )


# 3. Chọn cột văn bản đầu vào
text_col = artifact.get("text_column", "text_clean") if artifact else "text_clean"

if text_col not in df.columns:
    candidates = [
        c for c in df.columns
        if any(word in c.lower() for word in
               ["text_clean", "text", "feedback", "content", "comment"])
    ]
    if candidates:
        text_col = candidates[0]

# 4. Dự đoán bằng mô hình đã huấn luyện
if artifact is not None and text_col in df.columns:
    try:
        texts = df[text_col].fillna("").astype(str)

        X = hstack([
            artifact["word_vec"].transform(texts),
            artifact["char_vec"].transform(texts)
        ]).tocsr()

        predicted = pd.DataFrame(index=df.index)

        for task, info in artifact["models"].items():
            predicted[f"predicted_{task}"] = info["model"].predict(X)

        # Không ghi đè nhãn gốc hoặc nhãn cán bộ đã sửa
        for col in predicted.columns:
            df[col] = predicted[col].values

        st.success(f"Đã dự đoán {len(df):,} phản hồi bằng mô hình AI.")

    except Exception as e:
        st.error(f"Lỗi khi dự đoán phản hồi: {e}")
        predicted = pd.DataFrame(index=df.index)
else:
    predicted = pd.DataFrame(index=df.index)
    if artifact is not None:
        st.error(f"Không tìm thấy cột văn bản '{text_col}' trong CSV.")


# 5. Thống kê tổng quan
c1, c2, c3 = st.columns(3)
c1.metric("Tổng phản hồi", f"{len(df):,}")
c2.metric("Số trường dữ liệu", len(df.columns))

prediction_cols = [
    c for c in df.columns if c.startswith("predicted_")
]
c3.metric("Nhãn dự đoán AI", len(prediction_cols))

st.divider()


# 6. Thống kê nhãn dự đoán
st.subheader("📈 Thống kê dự đoán AI")

if prediction_cols:
    selected_pred = st.selectbox(
        "Chọn loại dự đoán",
        prediction_cols,
        format_func=lambda c: c.replace("predicted_", "").title()
    )

    pred_counts = df[selected_pred].fillna(
        "Chưa có dự đoán"
    ).astype(str).value_counts()

    st.bar_chart(pred_counts)
else:
    st.info("Chưa có kết quả dự đoán để thống kê.")


# 7. Tìm kiếm và danh sách phản hồi
st.divider()
st.subheader("📋 Danh sách phản hồi")

search = st.text_input("Tìm kiếm nội dung phản hồi")

text_candidates = [
    c for c in df.columns
    if any(word in c.lower() for word in
           ["text", "feedback", "content", "comment"])
]

filtered = df.copy()

if search and text_candidates:
    mask = filtered[text_candidates].fillna("").astype(str).apply(
        lambda col: col.str.contains(
            search, case=False, regex=False
        )
    ).any(axis=1)
    filtered = filtered[mask]

st.write(f"Tìm thấy **{len(filtered):,}** phản hồi")
st.dataframe(filtered, use_container_width=True, hide_index=True)


# 8. Cảnh báo khẩn cấp theo dự đoán priority
st.divider()
st.subheader("🚨 Cảnh báo khẩn cấp")

priority_col = "predicted_priority"

if priority_col in df.columns:
    urgent_df = df[
        df[priority_col].astype(str).str.strip().str.lower() == "cao"
    ].copy()

    st.metric("AI dự đoán mức ưu tiên cao", len(urgent_df))

    if not urgent_df.empty:
        st.warning(
            "Cần cán bộ kiểm tra các phản hồi được AI dự đoán ưu tiên cao."
        )
        st.dataframe(urgent_df, use_container_width=True, hide_index=True)
    else:
        st.success("AI chưa phát hiện phản hồi nào có mức ưu tiên cao.")
elif "is_urgent" in df.columns:
    urgent_values = pd.to_numeric(
        df["is_urgent"], errors="coerce"
    ).fillna(0)

    urgent_df = df[urgent_values == 1]

    st.metric("Phản hồi có cờ khẩn cấp trong dữ liệu", len(urgent_df))
    if not urgent_df.empty:
        st.dataframe(urgent_df, use_container_width=True, hide_index=True)
else:
    st.info("Chưa có nhãn ưu tiên dự đoán hoặc cột is_urgent.")


# 9. Cán bộ hiệu chỉnh nhãn
st.divider()
st.subheader("✏️ Hiệu chỉnh nhãn phản hồi")

# Chỉ cho phép sửa nhãn gốc, không sửa cột dự đoán AI
label_candidates = [
    c for c in df.columns
    if c.lower() in [
        "sentiment", "label", "priority", "primary_topic", "topic"
    ]
]

if not label_candidates:
    st.info("Không tìm thấy cột nhãn gốc để hiệu chỉnh.")
else:
    text_display_col = (
        text_candidates[0] if text_candidates else None
    )

    row_index = st.selectbox(
        "Chọn phản hồi cần sửa",
        df.index.tolist(),
        format_func=lambda i: (
            str(df.loc[i, text_display_col])[:100]
            if text_display_col else f"Phản hồi dòng {i}"
        )
    )

    label_col = st.selectbox(
        "Chọn nhãn gốc cần chỉnh sửa",
        label_candidates,
        key="edit_label_col"
    )

    current_label = str(df.loc[row_index, label_col])

    new_label = st.text_input(
        "Nhãn mới",
        value=current_label,
        key=f"new_label_{row_index}_{label_col}"
    )

    if st.button("Lưu nhãn đã sửa"):
        if new_label.strip():
            # Dữ liệu gốc được giữ riêng; chỉ lưu bản hiệu chỉnh
            if CORRECTION_PATH.exists():
                edited = load_data(str(CORRECTION_PATH))
            else:
                edited = df[
                    [c for c in df.columns if not c.startswith("predicted_")]
                ].copy()

            edited.loc[row_index, label_col] = new_label.strip()

            edited.to_csv(
                CORRECTION_PATH,
                index=False,
                encoding="utf-8-sig"
            )

            st.cache_data.clear()
            st.success(
                "Đã lưu nhãn cán bộ chỉnh sửa vào corrected_feedback.csv. "
                "Nhấn Rerun nếu kết quả chưa cập nhật."
            )
        else:
            st.warning("Nhãn mới không được để trống.")
