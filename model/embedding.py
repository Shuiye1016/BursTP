import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from torch.nn.utils import weight_norm
import builtins

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 500000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation, padding=(kernel_size - 1) * dilation, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Attention(nn.Module):

    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        weighted_rs = attn_weights * x
        return weighted_rs, attn_weights

class TokenEncoding(nn.Module):

    def __init__(self, c_in, d_model, hidden_dim=64, num_layers=1, dropout=0.2):
        super(TokenEncoding, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=c_in, num_channels=[d_model] * 3)

        self.bigru = nn.GRU(input_size=d_model, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.attention = Attention(hidden_dim * 2)

    def forward(self, x):
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        gru_out, _ = self.bigru(x)
        weighted_rs, attn_weights = self.attention(gru_out)
        return weighted_rs


class TemporalEncoding(nn.Module):
    """
    day，hour，minute， second (based on time_diff)
    return time_embedding and time_diff_embedding
    """
    def __init__(self, d_model):
        super(TemporalEncoding, self).__init__()
        day=32; hour = 24; minute=60; second=60; time_diff = 86400

        Embed = nn.Embedding

        self.day_embed = Embed(day, d_model)
        self.hour_embed = Embed(hour, d_model)
        self.minute_embed = Embed(minute, d_model)
        self.second_embed = Embed(second, d_model)
        self.diff_embed = Embed(time_diff, d_model)

        self.attn_fc = nn.Linear(d_model, 1)

    def forward(self, x, embed_type='pop'):
        x = x.long()
        if embed_type == 'pop':
            day_x = self.day_embed(x[:, :, 0])
            hour_x = self.hour_embed(x[:, :, 1])
            minute_x = self.minute_embed(x[:, :, 2])
            second_x = self.second_embed(x[:, :, 3])
            return day_x + hour_x + minute_x + second_x

        else:
            day_x = self.day_embed(x[:, 0])
            hour_x = self.hour_embed(x[:, 1])
            minute_x = self.minute_embed(x[:, 2])
            second_x = self.second_embed(x[:, 3])
            time_diff_x = self.diff_embed(torch.clamp(x[:, 4], 0, 86399))

            combined = torch.stack((day_x, hour_x, minute_x, second_x), dim=0)

            attn_weights = torch.softmax(-self.attn_fc(time_diff_x), dim=0)

            weighted_combined = attn_weights * combined
            output = torch.sum(weighted_combined, dim=0)

            return output, time_diff_x

class InfluenceEncoding(nn.Module):
    def __init__(self, d_model):
        super(InfluenceEncoding, self).__init__()
        Embed = nn.Embedding
        deg_count = 2500
        self.inf_embed = Embed(deg_count, d_model)

        # none-linear
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.long()
        inf_x = self.inf_embed(x)

        # none-linear layer
        inf_x = self.activation(self.fc1(inf_x))
        inf_x = self.dropout(inf_x)
        inf_x = self.activation(self.fc2(inf_x))
        inf_x = self.dropout(inf_x)

        return inf_x

class InitalEmbedding(nn.Module):
    def __init__(self, d_model, N, max_len=500000):
        super(InitalEmbedding, self).__init__()
        self.embed = nn.Embedding(N, d_model)

    def forward(self, x):
        x = x.long()
        return self.embed(x)

class NodeEmbedding(nn.Module):
    def __init__(self, d_model, N, dropout):
        super(NodeEmbedding, self).__init__()

        self.init_embedding = InitalEmbedding(d_model, N, 500000)
        self.inf_embedding = InfluenceEncoding(d_model)
        self.temp_embedding = TemporalEncoding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ids, deg, time):
        time_embedding, time_diff_embedding = self.temp_embedding(time, embed_type='node')
        x = self.init_embedding(ids) + self.inf_embedding(deg) + time_embedding
        return x, time_embedding, time_diff_embedding

class PopEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super(PopEmbedding, self).__init__()

        self.token_embedding = TokenEncoding(c_in, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.temp_embedding = TemporalEncoding(d_model)

    def forward(self, x, t_pos):
        x = x.unsqueeze(1)
        x = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        temp_emb = self.temp_embedding(t_pos, embed_type='pop')
        min_len = builtins.min(pos_emb.size(1), x.size(1), temp_emb.size(1))
        x = x[:, :min_len, :]
        pos_emb = pos_emb[:, :min_len, :]
        temp_emb = temp_emb[:, :min_len, :]
        y = x + pos_emb + temp_emb
        return y