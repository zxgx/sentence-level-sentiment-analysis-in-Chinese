import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding, pad_idx, freeze, hidden_dim=64,
                 n_layers=4, output_dim=2, dropout=0.75):
        
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            embedding, padding_idx=pad_idx, freeze=freeze
        )
        
        embed_dim = embedding.shape[1]
        self.rnn = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [ batch size, sent len]
        
        embedded = self.embedding(text.T)
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        #packed_output, (hidden, cell) = self.rnn(packed_embedded)
        _, (hidden, _) = self.rnn(packed_embedded)
        
        #unpack sequence
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)

