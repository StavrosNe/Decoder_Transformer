import torch 
import torch.nn as nn
from torch.nn.functional import softmax
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 seq_len: int, pad_mask: bool = False, 
                 dropout:float = 0.1):
        
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.seq_len = seq_len
        self.pad_mask = pad_mask
        
    
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        self.output_linear = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(dropout)
        self.linear_dropout = nn.Dropout(dropout)

    def causal_memory_mask(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor 
        shape(1, 1, seq_len, seq_len) 
        """
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0)
        return mask == 0
    

    def padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor 
        """
        mask = x
        return mask == 0
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, seq_len, d_model)
        
        Returns
        -------
        torch.Tensor (batch_size, num_heads, seq_len, dk)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.num_heads, self.dk)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, num_heads, seq_len, dk)
        
        Returns
        -------
        torch.Tensor (batch_size, seq_len, d_model)
        """
        batch_size = x.shape[0]
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, self.seq_len, self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, d_model)
        
        Internals
        ----------
        q: torch.Tensor
            Shape (batch_size, num_heads, seq_len, dk)
        
        k: torch.Tensor
            Shape (batch_size, num_heads, seq_len, dk)
        
        v: torch.Tensor
            Shape (batch_size, num_heads, seq_len, dk)
        
        scores: torch.Tensor
            Shape (batch_size, num_heads, seq_len, seq_len)
        
        attention_output: torch.Tensor
            Shape (batch_size, num_heads, seq_len, dk)
        
        Returns
        -------
        output_projection: torch.Tensor
            Shape (batch_size, seq_len, d_model)
        """
        
        q = self.split_heads(self.query(x))  
        k = self.split_heads(self.key(x))    
        v = self.split_heads(self.value(x))  
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        
        causal_mask = self.causal_memory_mask().to(x.device)
        masked_scores = scores.masked_fill(causal_mask, float('-inf'))
        
        if self.pad_mask:
            padding_mask = self.padding_mask(masked_scores).to(x.device)
            masked_scores = masked_scores.masked_fill(padding_mask, float('-inf'))

        attention_weights = self.attention_dropout(softmax(masked_scores, dim=-1))
        attention_output = torch.matmul(attention_weights, v)

        output = self.combine_heads(attention_output)
        output_projection = self.linear_dropout(self.output_linear(output))

        return output_projection

if __name__ == "__main__":
    d_model = 16
    h = 2
    seq_len = 4
    vocab_size = 400
    emb = nn.Embedding(vocab_size, 
                    d_model,
                    padding_idx=0
                    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transformer = MultiHeadAttention(d_model,h,seq_len,pad_mask=True).to(device)

    indx = [[5,2,3,7],[1,2,3,0],[1,2,3,0]]
    idx = torch.tensor(indx)

    x = emb(idx).to(device)
    print(f'x: {x.shape}')

    split = transformer.split_heads(x)
    print(f'split heads: {split.shape}')

    output = transformer.forward(x)
    print(f'output : {output.shape}')


