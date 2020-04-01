import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    def __init__(self, embedding, pad_idx, freeze, hidden_dim=512,
                 n_layers=1, output_dim=2, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
        	embedding, padding_idx=pad_idx, freeze=freeze
        )
        embed_dim = embedding.shape[-1]
        
        self.rnn = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout)
       	self.fc = nn.Linear(hidden_dim*2+embed_dim, output_dim)

    def forward(self, text, length):
    	# [ batch_size, sent_len ]
        embed = self.embedding(text.T)
        # [ sent_len, batch_size, embed_dim ]
        packed_embed =  nn.utils.rnn.pack_padded_sequence(embed, length)
        packed_output, (_, _) = self.rnn(packed_embed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        # [ sent_len, batch_size, hidden_dim * num_direction ]
        feat = F.relu(torch.cat((outputs, embed), dim=2)).permute(1, 2, 0)
        # [ sent_len, batch_size, hidden_dim * num_direction + embed_dim]
        out = F.max_pool1d(feat, feat.shape[2]).squeeze(2)
        return self.fc(out)
