import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class PhoBERT_BILSTM_Attention(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_classes=2,
        emb_dropout=0.2,          # dropout trên embedding PhoBERT trước khi vào BiLSTM
        fc_dropout=0.5,           # dropout trên vector ngữ cảnh trước lớp phân loại
        unfreeze_layers=(10, 11)  # các tầng encoder của PhoBERT được fine-tune
    ):
        super().__init__()
        # Tải mô hình PhoBERT pre-trained
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # Đóng băng toàn bộ PhoBERT
        for param in self.phobert.parameters():
            param.requires_grad = False

        # Chỉ mở khóa các tầng encoder cuối (pooler không dùng trong forward nên bỏ qua)
        prefixes = tuple(f"encoder.layer.{i}." for i in unfreeze_layers)
        for name, param in self.phobert.named_parameters():
            if any(p in name for p in prefixes):
                param.requires_grad = True

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.bilstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.emb_dropout(outputs.last_hidden_state)
        seq_len = embeddings.size(1)

        # Pack để BiLSTM không đọc token padding (quan trọng cho chiều ngược)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len
        )

        # Attention: che token padding rồi chuẩn hóa
        attn_logits = self.attention(lstm_out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, L, 1]

        # Vector ngữ cảnh = tổng có trọng số các hidden state
        context = torch.sum(lstm_out * attn_weights, dim=1)
        logits = self.fc(self.fc_dropout(context))
        return logits, attn_weights
