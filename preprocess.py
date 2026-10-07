
import pandas as pd
from google_play_scraper import Sort, reviews
from vncorenlp import VnCoreNLP

# Khởi tạo công cụ tách từ tiếng Việt
print("Đang khởi tạo VnCoreNLP...")
rdrsegmenter = VnCoreNLP("/content/drive/MyDrive/Python /Sentiment Analysis/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size="-Xmx512m")
print("Đã khởi tạo VnCoreNLP.")
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
    "😭": " khóc ",      
    ":p": " trêu_chọc ",
    ":P": " trêu_chọc ",
    ":o": " ngạc_nhiên ",
    ":O": " ngạc_nhiên ",
    "-_-": " khó_chịu ",
    ":-|": " nghiêm_túc "
}
def preprocess_vietnamese_text(text):
    if not isinstance(text, str):
        return ""
    for char, replacement in speacial_chars.items(): text = text.replace(char, replacement)
    try:
        sentences = rdrsegmenter.tokenize(text)
        flat_text = " ".join([" ".join(sentence) for sentence in sentences])
        return flat_text
    except Exception as e:
        print(e)
        return text

def run_pipeline(app_id="com.shopee.vn", count=10000):
    print("Đang cào dữ liệu từ Google Play...")
    raw_data, _ = reviews(
        app_id,
        lang='vi',
        country='vn',
        sort=Sort.NEWEST,
        count=count,
        filter_score_with=None
    )
    print("Đã cào dữ liệu xong.")

    df = pd.DataFrame(raw_data)[['content', 'score']]
    df.rename(columns={'content':'text', 'score':'rating'}, inplace=True)
    print("Đang tiền xử lý dữ liệu...")
    df = df[df['rating']!=3]
    df['label'] = df['rating'].apply(lambda x: 1 if x>3 else 0)

    clean_df = df[['text', 'label']].drop_duplicates(subset=['text']).dropna()
    print("Đang chạy tách từ bằng VnCoreNLP...")
    clean_df['text'] = clean_df['text'].apply(preprocess_vietnamese_text)
    print("Đã tách từ xong.")
    clean_df.to_csv("clean_dataset.csv", index=False, encoding="utf-8-sig")
    print("Data set đã được làm sạch. Đã sẵn sàng để huấn luyện mô hình!")

if __name__ == "__main__":
    run_pipeline()
