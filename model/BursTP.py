import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch import GlobalAttentionPooling
from torch.nn import Transformer
from model.Burstformer import BurstformerEnc, BurstformerDec

from model.embedding import NodeEmbedding, PopEmbedding
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score, mean_squared_error, precision
from torch.nn.functional import relu

from torchmetrics import F1Score, ConfusionMatrix, MeanAbsoluteError
from torchmetrics.classification import MulticlassAccuracy

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], dropout_rate=0.2):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            # layers.append(nn.ReLU())
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dropout(x)
        return self.layers(x)

class CoAttentionFusion(nn.Module):
    """
    CoAttentionFusion for POP_Embedding and Graph_Embedding
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(CoAttentionFusion, self).__init__()
        self.attn1 = nn.Linear(input_dim1, hidden_dim)
        self.attn2 = nn.Linear(input_dim2, hidden_dim)
        self.v1 = nn.Linear(hidden_dim, 1, bias=False)
        self.v2 = nn.Linear(hidden_dim, 1, bias=False)
        self.fc = nn.Linear(input_dim1 + input_dim2, output_dim)

    def forward(self, x1, x2):
        attn_weights12 = torch.softmax(self.v1(torch.tanh(self.attn1(x1) + self.attn2(x2))), dim=1)
        attn_weights21 = torch.softmax(self.v2(torch.tanh(self.attn1(x1) + self.attn2(x2))), dim=1)
        x1_attn = attn_weights12 * x1
        x2_attn = attn_weights21 * x2
        combined = torch.cat((x1_attn, x2_attn), dim=-1)
        output = self.fc(combined)
        return output

class GCN(nn.Module):
    """
    DAG-GNN
    """
    def __init__(self, net_params, dropout_rate):
        super(GCN, self).__init__()
        in_size, out_size, gcn_layers = net_params

        self.apply_func = MLP(in_size, out_size)
        self.gate_nn = MLP(in_size, 1)

        # GIN
        self.dgn = nn.ModuleList([GINConv(apply_func=self.apply_func, aggregator_type='max', activation=relu)
                                  for _ in range(gcn_layers)])

        self.pooling = GlobalAttentionPooling(gate_nn=self.gate_nn)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, graph, node_feat):
        h = node_feat
        for dgn in self.dgn:
            h = dgn(graph, node_feat)
            h_new = self.dropout(dgn(graph, h))
            h = h + self.residual_scale * h_new

        graph.ndata['h'] = h
        hg = self.pooling(graph, graph.ndata['h'])
        return hg

class GraphEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim, gcn_layers, N, dropout_rate, time_loss_weight):
        super(GraphEmbedding, self).__init__()
        self.gcn = GCN([d_model, hidden_dim, gcn_layers], dropout_rate)
        self.NE = NodeEmbedding(d_model, N, dropout_rate)
        self.hidden_dim = hidden_dim
        self.time_loss_weight = time_loss_weight

    def compute_time_loss(self, time_embedding, time_diff_embedding):
        time_loss = torch.zeros(1, device=time_embedding.device)

        adjacency_time_cs = 1 - F.cosine_similarity(time_embedding[:-1, :], time_embedding[1:, :], dim=-1)
        time_diff_cs = 1 - F.cosine_similarity(time_diff_embedding[:-1, :], time_diff_embedding[1:, :], dim=-1)
        weighted_loss = adjacency_time_cs * time_diff_cs

        time_loss += torch.sum(weighted_loss)
        time_loss = self.time_loss_weight * time_loss.mean()
        return time_loss

    def forward(self, batch_graphs):
        self.outs = torch.zeros(batch_graphs[0].batch_size, len(batch_graphs), self.hidden_dim, device=batch_graphs[0].device)
        total_time_loss = 0.0

        for i, graph in enumerate(batch_graphs):
            node_input, time_embedding, time_diff_embedding = self.NE(graph.ndata['nid'], graph.ndata['deg'], graph.ndata['time'])
            h_prev = self.gcn(graph, node_input)

            # GNN
            self.outs[:, i, :] = h_prev

            # compute time_loss
            time_loss = self.compute_time_loss(time_embedding=time_embedding, time_diff_embedding=time_diff_embedding)
            total_time_loss += time_loss

        total_time_loss /= len(batch_graphs)
        return self.outs, total_time_loss

class PopEncoderDecoder(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, dropout_rate, act, num_layers, label_len):
        super(PopEncoderDecoder, self).__init__()
        self.label_len = label_len

        if act == 'relu':
            activation = F.relu
        elif act == 'gelu':
            activation = F.gelu
        elif act == 'leaky_relu':
            activation = F.leaky_relu
        elif act == 'elu':
            activation = F.elu

        self.encoder_layer = BurstformerEnc(d_model=d_model, nhead=nhead, ffn_dim=ffn_dim,
                                                     dropout_rate=dropout_rate, attn_dropout_rate=dropout_rate, activation=activation)
        self.encoder = nn.ModuleList([self.encoder_layer for _ in range(num_layers)])

        self.decoder_layer = BurstformerDec(d_model=d_model, nhead=nhead, ffn_dim=ffn_dim,
                                                     dropout_rate=dropout_rate, attn_dropout_rate=dropout_rate, activation=activation)
        self.decoder = nn.ModuleList([self.decoder_layer for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_rate)

        # Conv
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.gelu = nn.GELU()

    def forward(self, x, tgt, f_burst_src=None, f_burst_fur=None, masks=None, memory_mask=None):
        if masks is None:
            masks = Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Enc
        x = self.dropout(x)
        for layer in self.encoder:
            x, attn_weights = layer(x, f_burst=f_burst_src)

        # Dec
        tgt = self.dropout(tgt)
        for layer in self.decoder:
            tgt, self_attn_weights, cross_attn_weights = layer(tgt, x, f_burst_src=f_burst_src, f_burst_fur=f_burst_fur, tgt_mask=masks, memory_mask=memory_mask)

        # De-Conv
        de_out = tgt.transpose(1, 2)
        de_out = self.conv(de_out)
        de_out = de_out.transpose(1, 2)
        de_out = self.gelu(de_out)
        de_out = self.dropout(de_out)

        return x, de_out[:, self.label_len:, :]

class BursTP(pl.LightningModule):
    def __init__(self,d_model, hidden_dim, gcn_layers, N, dropout_rate, seq_len, nhead, ffn_dim, act,
                 num_layers, classes, label_len, lr, weight_decay, time_loss_weight):
        super(BursTP, self).__init__()
        self.label_len = label_len
        self.hidden_dim = hidden_dim
        self.classes = classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.POPE = PopEmbedding(1, d_model, dropout_rate)
        self.PED = PopEncoderDecoder(d_model, nhead, ffn_dim, dropout_rate, act,
                                     num_layers, label_len)
        self.GE = GraphEmbedding(d_model=d_model,hidden_dim=hidden_dim, gcn_layers=gcn_layers,
                                     N=N, dropout_rate=dropout_rate, time_loss_weight=time_loss_weight)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layernorm = nn.LayerNorm(d_model)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse = nn.MSELoss(reduction='sum')
        self.fusion = CoAttentionFusion(d_model, d_model, hidden_dim, d_model)
        self.classifier = MLP(1536, classes)
        self.regressier = MLP(hidden_dim, 1, hidden_dims=[hidden_dim, hidden_dim], dropout_rate=0.2)
        self.max_pool = nn.AdaptiveMaxPool1d(1536)
        self.extra_fc = nn.Linear(hidden_dim, hidden_dim)
        self.extra_activation = nn.ReLU()

        self.save_hyperparameters()
        self.f1_score = F1Score(task='multiclass', num_classes=classes, average='weighted')
        self.mae = MeanAbsoluteError()
        self.conf_matrix = ConfusionMatrix(task='multiclass', num_classes=classes)

        self.multi_acc = MulticlassAccuracy(num_classes=self.classes, average=None)

        self.alpha_1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.alpha_3 = nn.Parameter(torch.randn(1), requires_grad=True)

    def compute_f_burst(self, pop_tr, fd_tr, sd_tr):
        # Compute learnable bias for each window
        f_burst = torch.tanh(
            self.alpha_1 * pop_tr +
            self.alpha_2 * fd_tr +
            self.alpha_3 * sd_tr
        )
        return f_burst

    def forward(self, graphs, pops, t_pos, tgt_pops, tgt_pos, src_fd, src_sd, fur_fd, fur_sd):
        pope = self.POPE(pops, t_pos)
        tgt_pope = self.POPE(tgt_pops, tgt_pos)
        ge, time_loss = self.GE(graphs)
        out = self.fusion(pope, ge)
        f_burst_src = self.compute_f_burst(pops, src_fd, src_sd)
        f_burst_fur = self.compute_f_burst(tgt_pops, fur_fd, fur_sd)
        en_x, de_out = self.PED(out, tgt_pope, f_burst_src=f_burst_src, f_burst_fur=f_burst_fur)
        # processed_deout = self.extra_fc(de_out.sum(1))
        # processed_deout = self.extra_activation(processed_deout)
        en_x = en_x.view(en_x.size(0), -1)
        en_x = self.max_pool(en_x)
        class_out = self.classifier(en_x)
        regress_out = self.regressier(de_out).squeeze(-1)
        return class_out, regress_out, time_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=3)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [cosine_annealing]


    def training_step(self, batch_data, batch_idx):
        self.train()
        batch_graphs, src_pop_tr, src_pos_t, labels, pop_labels_tr, tgt_label, tgt_pos_t, cids, pln_tr, fdn_tr, sdn_tr, src_fd_tr, src_sd_tr = batch_data
        class_out, regress_out, time_loss = self(batch_graphs, src_pop_tr, src_pos_t, tgt_label, tgt_pos_t, src_fd_tr, src_sd_tr, fdn_tr, sdn_tr)
        loss = self.ce(class_out, labels) + 0.05 * self.mse(regress_out, tgt_label[:, self.label_len:]) + time_loss
        acc = accuracy(class_out.softmax(-1), labels, top_k=1, task="multiclass", num_classes=self.classes)
        self.log_dict({'loss': loss, 'acc':acc, 'train_te': time_loss})
        return loss

    def validation_step(self, batch_data, batch_idx):
        self.eval()
        batch_graphs, src_pop_tr, src_pos_t, labels, pop_labels_tr, tgt_label, tgt_pos_t, cids, pln_tr, fdn_tr, sdn_tr, src_fd_tr, src_sd_tr = batch_data
        class_out, regress_out, time_loss = self(batch_graphs, src_pop_tr, src_pos_t, tgt_label, tgt_pos_t, src_fd_tr, src_sd_tr, fdn_tr, sdn_tr)
        loss = self.ce(class_out, labels) + 0.05 * self.mse(regress_out, tgt_label[:, self.label_len:]) + time_loss
        return {'loss': loss, 'class_pre': class_out, 'classes': labels, 'regress_pre': regress_out, 'regresses': tgt_label[:, self.label_len:], 'te': time_loss}

    def validation_epoch_end(self, outputs):
        avg_te = torch.stack([x['te'] for x in outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        class_pre = torch.cat([o['class_pre'] for o in outputs])
        classes = torch.cat([o['classes'] for o in outputs])
        regress_pre = torch.cat([o['regress_pre'] for o in outputs])
        regresses = torch.cat([o['regresses'] for o in outputs])
        acc = accuracy(preds=class_pre.softmax(-1), target=classes, top_k=1, task="multiclass", num_classes=self.classes)
        f1 = f1_score(preds=class_pre.softmax(-1), target=classes, task="multiclass", num_classes=self.classes,
                               average="weighted")
        mse = mean_squared_error(preds=regress_pre, target=regresses)
        pc = precision(preds=class_pre, target=classes, task="multiclass", average='weighted', num_classes=self.classes)

        self.log_dict({'val_loss': avg_loss})


    def test_step(self, batch_data, batch_idx):
        self.eval()
        batch_graphs, src_pop_tr, src_pos_t, labels, pop_labels_tr, tgt_label, tgt_pos_t, cids, pln_tr, fdn_tr, sdn_tr, src_fd_tr, src_sd_tr = batch_data
        class_out, regress_out, time_loss = self(batch_graphs, src_pop_tr, src_pos_t, tgt_label, tgt_pos_t, src_fd_tr,
                                                 src_sd_tr, fdn_tr, sdn_tr)
        return {'class_pre': class_out, 'classes': labels, 'regress_pre': regress_out, 'regresses': tgt_label[:, self.label_len:]}

    def test_epoch_end(self, outputs):
        class_pre = torch.cat([o['class_pre'] for o in outputs])
        classes = torch.cat([o['classes'] for o in outputs])
        regress_pre = torch.cat([o['regress_pre'] for o in outputs])
        regresses = torch.cat([o['regresses'] for o in outputs])
        acc = accuracy(preds=class_pre.softmax(-1), target=classes, top_k=1, task="multiclass", num_classes=self.classes)
        f1 = f1_score(preds=class_pre.softmax(-1), target=classes, task="multiclass", num_classes=self.classes, average="weighted")
        pc = precision(preds=class_pre, target=classes, task="multiclass", average='weighted', num_classes=self.classes)
        self.log_dict({'acc': acc, 'f1': f1, 'precision': pc})

    @staticmethod
    def setting_model_args(parent_parser):
        parser = parent_parser.add_argument_group("BursTP")
        parser.add_argument('--data_name', type=str, default="twitter")
        parser.add_argument('--seq_len', type=int, default=2, help="input length.")
        parser.add_argument('--label_len', type=int, default=1, help="start_len.")
        parser.add_argument('--classes', type=int, default=46, help="Num of classes.")
        parser.add_argument('--d_model', type=int, default=128, help="The input dimension of models.")
        parser.add_argument('--hidden_dim', type=int, default=128, help="The hidden dimension.")
        parser.add_argument('--gcn_layers', type=int, default=2, help="Num of graph encoder layers.")
        parser.add_argument('--aggregators', type=str, default='max', help="aggregators.")
        parser.add_argument('--act', type=str, default='gelu', help="Active function.")
        parser.add_argument('--nhead', type=int, default=4, help='The num heads of attention.')
        parser.add_argument('--ffn_dim', type=int, default=128, help='FFN dimension.')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout prob.')
        parser.add_argument('--attn_dropout_rate', type=float, default=0.3, help='Attention dropout prob.')
        parser.add_argument('--num_layers', type=int, default=4, help="Num of Transformer encoder layers.")
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay.')
        parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
        parser.add_argument('--clip_val', type=float, default=0, help="Gradient clipping values. ")
        parser.add_argument('--total_epochs', type=int, default=500, help="Max epochs of model training.")
        parser.add_argument('--lr_decay_step', type=int, default=5, help="Learning rate decay step size.")
        parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help="Learning rate decay rate.")
        parser.add_argument('--gpu_lst', nargs='+', type=int, help="Which gpu to use.")
        parser.add_argument('--observation', type=str, default="")
        parser.add_argument('--is_ge', action='store_false', default="add cascade graph embedding.")
        parser.add_argument('--is_vae', action='store_false', default="add vae module.")
        parser.add_argument('--is_reg', action='store_false', default="add regression task loss.")
        parser.add_argument('--time_loss_weight', type=float, default=1e-5, help='Time loss weight')
        return parent_parser