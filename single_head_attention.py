import torch 
import torch.nn as nn
from torch.nn.functional import softmax
import math


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model,seq_len,pad_mask:bool = False):
        super().__init__()

        self.query = nn.Linear(d_model,d_model,bias=False)
        self.key = nn.Linear(d_model,d_model,bias=False)
        self.value = nn.Linear(d_model,d_model,bias=False)
        self.output_linear = nn.Linear(d_model, d_model,bias=True)

        self.d_model = d_model
        self.seq_len = seq_len
        self.pad_mask = pad_mask

    def causal_memory_mask(self)->torch.Tensor:
        """
        create causal mask

        Returns
        -------
        torch.Tensor
            shape(1,seq_len,seq_len)
        """

        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0)
        return mask == 0
    
    def padding_mask(self,x:torch.tensor)->torch.Tensor:
        """
        create padding mask

        Returns
        -------
        torch.Tensor
        """

        mask = x 

        return mask == 0

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        Compute scaled dot product attention for a signle head
    
        Parameters
        ----------
        x : torch.Tensor 
            Shape (batch_size,seq_len,d_model)
        
        Internals
        ----------
        q: torch.Tensor
            Shape (batch_size,seq_len,d_model)
        k: torch.Tensor
            Shape (batch_size,seq_len,d_model)
        v: torch.Tensor
            Shape (batch_size,seq_len,d_model)
        scores: torch.Tensor
            Shape (batch_size,seq_len,seq_len)
        
        Returns
        -------
        self_attention: torch.Tensor
            Shape (batch_size,seq_len,d_model) 
        
        output : torch.Tensor
            Shape (batch_size,seq_len,d_model)
        """

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
    
        causal_mask = self.causal_memory_mask().to(x.device)

        masked_scores = scores.masked_fill_(causal_mask, float('-inf'))

        if self.pad_mask:
            padding_mask = self.padding_mask(masked_scores)

            masked_scores = masked_scores.masked_fill_(padding_mask, float('-inf'))

        attention_weights = softmax(masked_scores, dim=-1)

        self_attention = torch.matmul(attention_weights, v)

        output = self.output_linear(self_attention)

        return output
    


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    indx = [[1,2,3,4,5],[1,2,3,6,0],[1,2,3,0,0]]

    idx = torch.tensor(indx)

    d_model = 12
    vocab_size = 100
    seq_len = 5
    emb = nn.Embedding(vocab_size, 
                    d_model,
                    padding_idx=0)

    x = emb(idx).to(device)

    attention = SingleHeadAttention(d_model,seq_len,pad_mask=True).to(device)

    output = attention(x)
    