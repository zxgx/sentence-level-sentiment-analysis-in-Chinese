import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    
    def __init__(self, embedding, pad_idx, freeze, num_filters=16,
                 filter_sizes=[2, 3], output_dim=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            embedding, padding_idx=pad_idx, freeze=freeze
        )
        
        embed_dim = embedding.shape[1]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=num_filters,
                                              kernel_size=(filter_size, embed_dim))
                                    for filter_size in filter_sizes])    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters*len(filter_sizes), output_dim)
    
    def forward(self, x):
        # [ batch_size, seq_len ]
        
        x = self.embedding(x).unsqueeze(1)

        # [ batch_size, 1, seq_len, embed_dim ]
        
        conved = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        
        # conved[i] = [ batch_size, num_filters, conved_height ]
        # conved_width equals to 1, and is squeezed
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # pooled[i] = [ batch_size, num_filters ]
        # pooled_height equals to 1, and is squeezed
        
        x = self.dropout(torch.cat(pooled, 1))
        
        # [ batch_size, num_filters*len(filter_sizes) ]
        
        x = self.fc(x)
        
        return x

