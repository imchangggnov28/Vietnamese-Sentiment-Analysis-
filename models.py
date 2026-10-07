import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERT_BILSTM_Attention(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super().__init__()
        # Tải mô hình PhoBERT pre-trained
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # Đóng băng (freeze) toàn bộ trọng số của PhoBERT pre-trained
        for param in self.phobert.parameters():
            param.requires_grad = False

        # Mở khóa đóng băng 2 tầng Encoder cuối cùng (tầng 10 & 11) + tầng Pooler
        for name, param in self.phobert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

        self.bilstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        lstm_out, _ = self.bilstm(embeddings)

        # Tính toán xem từ nào trong câu quan trọng nhất bằng attention
        attn_logits = self.attention(lstm_out)  # Tính điểm số quan trọng thô cho từng từ

        attn_logits = attn_logits.squeeze(-1).masked_fill( attention_mask == 0,-1e9)  # Xử lý attention masking

        attn_weights = torch.softmax( attn_logits, dim=1)  # Chuẩn hóa attention score giữa các token; tổng trọng số = 1

        attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len] -> [batch, seq_len, 1]

        context = torch.sum(lstm_out * attn_weights, dim=1)  # Tạo vector biểu diễn bằng tổng có trọng số của các hidden state
        logits = self.fc(context)
        return logits, attn_weights
