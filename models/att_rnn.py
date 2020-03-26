import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(query, key, value, mask):
    hd = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hd)
    scores.masked_fill_(mask, 1e-9)
    att_vector = F.softmax(scores, dim=-1)
    return torch.matmul(att_vector, value), att_vector

class Att_RNN(nn.Module):
    
    def __init__(self, embedding, pad_idx, freeze, hidden_dim=256,
                 n_layers=2, output_dim=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding, padding_idx=pad_idx, freeze=freeze
        )
        embed_dim = embedding.shape[1]
        self.rnn = nn.LSTM(embed_dim, hidden_dim,
                           num_layers=n_layers, 
                           bidirectional=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_vector = None
        self.pad_idx = pad_idx

    def create_mask(self, text):
        mask = (text==self.pad_idx).to(text.device)
        return mask.unsqueeze(1)

    def forward(self, text, length, output_attention=False):
        # [ batch_size, sent_len ]
        embed = self.embedding(text.T)
        # [ sent_len, batch_size ]

        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, length)
        packed_output, (_, _) = self.rnn(packed_embed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output: [ sent_len, batch_size, num_dir * hidden_dim ]
        outputs = outputs.transpose_(0, 1)
        mask = self.create_mask(text)
        attentive, self.att_vector = attention(outputs, outputs, outputs, mask)
        res = self.dropout(torch.mean(attentive, dim=1))
        return self.fc(res)

