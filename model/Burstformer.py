import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = hidden_dim // num_heads
        self.scale = self.attn_dim ** -0.5

        self.linear_Q = nn.Linear(hidden_dim, num_heads * self.attn_dim)
        self.linear_K = nn.Linear(hidden_dim, num_heads * self.attn_dim)
        self.linear_V = nn.Linear(hidden_dim, num_heads * self.attn_dim)
        self.linear_out = nn.Linear(num_heads * self.attn_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

    def forward(self, q, k, v, attn_bias=None, attn_mask=None):
        batch_size = q.size(0)

        q = self.linear_Q(q).view(batch_size, -1, self.num_heads, self.attn_dim)
        k = self.linear_K(k).view(batch_size, -1, self.num_heads, self.attn_dim)
        v = self.linear_V(v).view(batch_size, -1, self.num_heads, self.attn_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            attn_bias_expand = attn_bias.unsqueeze(1).unsqueeze(2)
            attn_bias_expand = attn_bias_expand.expand(-1, attn_weights.size(1), attn_weights.size(2), -1)
            attn_weights = attn_weights + attn_bias_expand

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.attn_dim)
        output = self.linear_out(output)

        return output, attn_weights

class BurstformerEnc(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, dropout_rate, attn_dropout_rate, activation):
        super(BurstformerEnc, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout_rate)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, src, f_burst=None, src_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_bias=f_burst, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class BurstformerDec(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, dropout_rate, attn_dropout_rate, activation):
        super(BurstformerDec, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout_rate)
        self.cross_attn = MultiHeadAttention(d_model, nhead, attn_dropout_rate)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, tgt, memory, f_burst_src=None, f_burst_fur=None, tgt_mask=None, memory_mask=None):
        # Self Attention with bias
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_bias=f_burst_fur, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross Attention with bias
        tgt2, cross_attn_weights = self.cross_attn(tgt, memory, memory, attn_bias=f_burst_src, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward layer
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, self_attn_weights, cross_attn_weights
