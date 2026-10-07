import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from models import PhoBERT_BILSTM_Attention

# ============================================================
# CẤU HÌNH
# ============================================================
SEED = 42
BATCH_SIZE = 8
MAX_LEN = 64
LEARNING_RATE = 2e-5
EPOCHS = 5

# Tỷ lệ dữ liệu: Train 70% - Dev 15% - Test 15%
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# Đảm bảo kết quả chia dữ liệu có thể tái lập

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


tokenizer = AutoTokenizer.from_pretrained(
    "vinai/phobert-base",
    use_fast=False
)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = tokenizer(
            str(self.texts[item]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[item], dtype=torch.long)
        }


def evaluate(model, loader, device):
    """Đánh giá mô hình trên một tập dữ liệu."""
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }


def train():
    # 1. Đọc dữ liệu đã tiền xử lý
    df = pd.read_csv("clean_dataset.csv").dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Chuẩn hóa kiểu nhãn
    df["label"] = df["label"].astype(int)

    # 2. Chia Train 70% - Dev 15% - Test 15%
    #    Chia 2 bước và stratify theo label để giữ tỷ lệ lớp.
    train_df, temp_df = train_test_split(
        df,
        test_size=(DEV_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=df["label"]
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (DEV_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=temp_df["label"]
    )

    # Kiểm tra tỷ lệ thực tế
    total = len(df)
    print("=" * 60)
    print("PHÂN CHIA DỮ LIỆU: TRAIN 70% - DEV 15% - TEST 15%")
    print("=" * 60)
    print(f"Tổng số mẫu : {total}")
    print(f"Train       : {len(train_df)} ({len(train_df)/total:.2%})")
    print(f"Dev         : {len(dev_df)} ({len(dev_df)/total:.2%})")
    print(f"Test        : {len(test_df)} ({len(test_df)/total:.2%})")

    print("\nPhân bố nhãn:")
    for name, part in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        counts = part["label"].value_counts().sort_index().to_dict()
        print(f"{name}: {counts}")

    # Lưu các tập dữ liệu để có thể kiểm tra/tái lập thực nghiệm
    train_df.to_csv("train_dataset.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv("dev_dataset.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv("test_dataset.csv", index=False, encoding="utf-8-sig")

    # 3. DataLoader
    train_loader = DataLoader(
        SentimentDataset(train_df["text"], train_df["label"]),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_loader = DataLoader(
        SentimentDataset(dev_df["text"], dev_df["label"]),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        SentimentDataset(test_df["text"], test_df["label"]),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 4. Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    print("Train batches:", len(train_loader))
    print("Dev batches:", len(dev_loader))
    print("Test batches:", len(test_loader))

    model = PhoBERT_BILSTM_Attention().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Huấn luyện: chỉ dùng Dev để chọn best_model.pt
    best_dev_acc = -1.0
    history = []

    print("\nĐang bắt đầu huấn luyện mô hình...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        dev_result = evaluate(model, dev_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_result["loss"],
            "dev_accuracy": dev_result["accuracy"],
            "dev_precision": dev_result["precision"],
            "dev_recall": dev_result["recall"],
            "dev_f1": dev_result["f1"]
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Dev Loss: {dev_result['loss']:.4f} - "
            f"Dev Acc: {dev_result['accuracy']:.4f} - "
            f"Dev F1: {dev_result['f1']:.4f}"
        )

        # Chỉ dùng DEV để chọn checkpoint tốt nhất.
        if dev_result["accuracy"] > best_dev_acc:
            best_dev_acc = dev_result["accuracy"]
            torch.save(model.state_dict(), "best_model.pt")
            print("💾 Đã lưu best_model.pt theo Dev Accuracy!")

    # Lưu lịch sử huấn luyện 
    pd.DataFrame(history).to_csv(
        "training_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 6. Đánh giá TEST sau khi đã chọn best_model.pt
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST")
    print("=" * 60)

    model.load_state_dict(
        torch.load("best_model.pt", map_location=device)
    )

    test_result = evaluate(model, test_loader, device)

    print(f"Test Loss      : {test_result['loss']:.4f}")
    print(f"Test Accuracy  : {test_result['accuracy']:.4f}")
    print(f"Test Precision : {test_result['precision']:.4f}")
    print(f"Test Recall    : {test_result['recall']:.4f}")
    print(f"Test F1-score  : {test_result['f1']:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_result["y_true"],
            test_result["y_pred"],
            target_names=["Tiêu cực", "Tích cực"],
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        test_result["y_true"],
        test_result["y_pred"]
    )
    cm_df = pd.DataFrame(
        cm,
        index=["Thực tế: Tiêu cực", "Thực tế: Tích cực"],
        columns=["Dự đoán: Tiêu cực", "Dự đoán: Tích cực"]
    )
    print("Confusion Matrix:")
    print(cm_df)

    # Lưu kết quả test để đưa vào báo cáo
    test_metrics = pd.DataFrame([{
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_precision": test_result["precision"],
        "test_recall": test_result["recall"],
        "test_f1": test_result["f1"]
    }])
    test_metrics.to_csv(
        "test_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(cm,index=["actual_negative", "actual_positive"], columns=["pred_negative", "pred_positive"]).to_csv("test_confusion_matrix.csv", encoding="utf-8-sig")
# LƯU KẾT QUẢ DỰ ĐOÁN TỪNG MẪU TEST
    prediction_df = test_df.copy()

    # Nhãn thực tế và nhãn mô hình dự đoán
    prediction_df["actual"] = test_result["y_true"]
    prediction_df["predicted"] = test_result["y_pred"]

    # Đổi nhãn số sang tên cảm xúc
    prediction_df["actual_label"] = prediction_df["actual"].map({
        0: "Tiêu cực",
        1: "Tích cực"
    })

    prediction_df["predicted_label"] = prediction_df["predicted"].map({
        0: "Tiêu cực",
        1: "Tích cực"
    })

    # Kiểm tra mô hình dự đoán đúng hay sai
    prediction_df["correct"] = (
        prediction_df["actual"] == prediction_df["predicted"]
    )

    # Lưu toàn bộ kết quả dự đoán
    prediction_df.to_csv(
        "test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Lưu riêng những mẫu dự đoán sai
    wrong_predictions = prediction_df[
        prediction_df["correct"] == False
    ]

    wrong_predictions.to_csv(
        "test_wrong_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nHoàn tất. Các file kết quả đã được tạo:")
    print("- train_dataset.csv")
    print("- dev_dataset.csv")
    print("- test_dataset.csv")
    print("- best_model.pt")
    print("- training_history.csv")
    print("- test_metrics.csv")
    print("- test_confusion_matrix.csv")
    print("- test_predictions.csv")
    print("- test_wrong_predictions.csv")



if __name__ == "__main__":
    train()
