import torch 
import torch.nn as nn
from multi_head_attention import MultiAttention

class Decoder_Block(nn.Module):
    def __init__(self, d_model:int, num_heads: int, 
                 seq_len: int, pad_mask: bool = False, 
                 dropout:float = 0.1):
        
        super().__init__()


        self.attention = MultiAttention(d_model,num_heads,
                                        seq_len,pad_mask=pad_mask,
                                        dropout=dropout)
        
        self.fcff = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                          nn.ReLU(),
                                          nn.Linear(4 * d_model, d_model),
                                          nn.Dropout(dropout))

        self.norm_att = nn.LayerNorm(d_model)
        self.norm_fcff = nn.LayerNorm(d_model)

    def forward(self, x) -> torch.Tensor:
        """
        Compute the output of the Decoder_Block.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, d_model)

        Returns
        -------
        output : torch.Tensor
            Shape (batch_size, seq_len, d_model)
        """
        x_norm = self.norm_att(x)
        x = x + self.attention(x_norm)
        
        x_norm = self.norm_fcff(x)
        x = x + self.fcff(x_norm)
        return x