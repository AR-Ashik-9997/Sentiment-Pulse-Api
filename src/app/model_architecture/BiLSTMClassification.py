import torch
import torch.nn as nn
class BiLSTMClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, (hidden, cell) = self.lstm(embedded)

        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_out = self.dropout(hidden_cat)
        output = self.fc(dense_out)
        return output
