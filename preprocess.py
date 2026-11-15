import pandas as pd
from google_play_scraper import reviews, app, Sort
from vncorenlp import VnCoreNLP
import os

# ===================== KHỞI TẠO VNCORENLP =====================

print("Đang khởi tạo VnCoreNLP...")

@st.cache_resource
def load_vncorenlp():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(base_dir, "VnCoreNLP", "VnCoreNLP-1.1.1.jar")
    
    annotator = VnCoreNLP(
        jar_path,
        annotators="wseg",
        max_heap_size="-Xmx512m"
    )
    return annotator

rdrsegmenter = load_vncorenlp()

print("Đã khởi tạo VnCoreNLP.")

# ===================== XỬ LÝ KÝ TỰ ĐẶC BIỆT =====================

speacial_chars = {
    ":)": " tích_cực ",
    ":D": " vui_vẻ ",
    ":>": " hạnh_phúc ",
    "^^": " cười_tươi ",
    ":- )": " tích_cực ",
    "(:": " tích_cực ",
    ":(": " tiêu_cực ",
    ":<": " buồn_bã ",
    ":'(": " khóc ",
    ":p": " trêu_chọc ",
    ":P": " trêu_chọc ",
    ":o": " ngạc_nhiên ",
    ":O": " ngạc_nhiên ",
    "-_-": " khó_chịu ",
    ":-|": " nghiêm_túc "
}

# ===================== HÀM TIỀN XỬ LÝ =====================

def preprocess_vietnamese_text(text):
    if not isinstance(text, str):
        return ""

    # Thay thế biểu tượng cảm xúc
    for char, replacement in speacial_chars.items():
        text = text.replace(char, replacement)

    try:
        sentences = rdrsegmenter.tokenize(text)

        flat_text = " ".join(
            [" ".join(sentence) for sentence in sentences]
        )

        return flat_text

    except Exception as e:
        print(f"Lỗi tách từ: {e}")
        return text


# ===================== 1. TIỀN XỬ LÝ TRAIN =====================

app_ids = [
    'com.facebook.katana',
    'com.google.android.youtube',
    'com.shopee.vn',
    'com.ss.android.ugc.trill',
    'com.haidilao.intl',
    'com.garena.game.kgvn'
]

all_review_data = []

for app_id in app_ids:
    print(f"Đang thu thập dữ liệu từ ứng dụng {app_id}...")

    try:
        app_info = app(app_id, lang='vi', country='vn')

        raw_data, _ = reviews(
            app_id,
            lang='vi',
            country='vn',
            sort=Sort.NEWEST,
            count=10000,
            filter_score_with=None
        )

        temp_df = pd.DataFrame(raw_data)[['content', 'score']]

        temp_df.rename(
            columns={
                'content': 'text',
                'score': 'rating'
            },
            inplace=True
        )

        all_review_data.append(temp_df)

    except Exception as e:
        print(f"Lỗi khi thu thập dữ liệu từ ứng dụng {app_id}: {e}")
        continue


if all_review_data:

    df = pd.concat(all_review_data, ignore_index=True)

    print(f"Tổng số mẫu thu thập được: {len(df)}")

    print("Đang tiền xử lý dữ liệu Train...")

    df = df[df['rating'] != 3]

    df['label'] = df['rating'].apply(
        lambda x: 1 if x > 3 else 0
    )

    clean_df = (
        df[['text', 'label']]
        .dropna()
        .drop_duplicates(subset=['text'])
        .reset_index(drop=True)
    )

    print("Đang chạy tách từ Train bằng VnCoreNLP...")

    clean_df['text'] = clean_df['text'].apply(
        preprocess_vietnamese_text
    )

    clean_df.to_csv(
        "clean_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Đã hoàn thành tiền xử lý Train.")
    print(f"Số mẫu Train: {len(clean_df)}")


# ===================== 2. TIỀN XỬ LÝ TEST =====================

print("\nĐang đọc dữ liệu Test.csv...")

if os.path.exists("Test.csv"):

    test_df = pd.read_csv("Test.csv")

    print(f"Số mẫu Test ban đầu: {len(test_df)}")

    # Kiểm tra cột dữ liệu
    if not {'text', 'label'}.issubset(test_df.columns):
        raise ValueError(
            "Test.csv phải có hai cột: text và label"
        )

    # Loại bỏ dữ liệu rỗng
    test_df = test_df.dropna(
        subset=['text', 'label']
    ).copy()

    test_df['label'] = test_df['label'].astype(int)

    # Loại bỏ trùng lặp trong Test
    test_df = test_df.drop_duplicates(
        subset=['text']
    ).reset_index(drop=True)

    print("Đang chạy tách từ Test bằng VnCoreNLP...")

    test_df['text'] = test_df['text'].apply(
        preprocess_vietnamese_text
    )

    # Lưu Test đã tách từ
    test_df.to_csv(
        "test_preprocessed.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Đã hoàn thành tiền xử lý Test.")
    print(f"Số mẫu Test sau xử lý: {len(test_df)}")

else:
    print("Không tìm thấy Test.csv. Bỏ qua tiền xử lý Test.")


# ===================== 3. ĐÓNG VNCORENLP =====================

rdrsegmenter.close()

print("\nHoàn tất tiền xử lý dữ liệu!")
print("Các file đầu ra:")
print("- clean_dataset.csv")
print("- test_preprocessed.csv")
