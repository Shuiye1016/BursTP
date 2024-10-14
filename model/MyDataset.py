import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import networkx as nx
import numpy as np
import pickle
from collections import Counter
import dgl
import pprint
from model.utils import StandardScaler

class CasData(Data.Dataset):
    def __init__(self, path, seq_len, label_len):
        super(CasData, self).__init__()
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        self.cid = dataset['cid']
        self.graph_lst = dataset['graph_lst']
        self.pop_lst = dataset['pop_lst']
        self.t_pos = np.array(dataset['t_pos'])
        self.classes = [c for c in dataset['classes']]
        self.pop_label = dataset['pop_label']

        self.pln = dataset['pop_label_norm']
        self.fdn = dataset['first_derivative_norm']
        self.sdn = dataset['second_derivative_norm']

        self.fd_lst = dataset['fd_lst']
        self.sd_lst = dataset['sd_lst']

        scaler = StandardScaler()
        data = np.hstack((self.pop_lst, self.pop_label))
        scaler.fit(data)
        self.pop_lst = scaler.transform(self.pop_lst)
        self.pop_label = scaler.transform(self.pop_label)
        self.fd_lst = scaler.transform(self.fd_lst)
        self.sd_lst = scaler.transform(self.sd_lst)

        self.tgt_label = np.hstack((self.pop_lst[:,-label_len:], np.zeros_like(self.pop_label)))
        self.src_pos_t = self.t_pos[:,:seq_len,:]
        self.tgt_pos_t = self.t_pos[:,seq_len-label_len:,:]


    def __getitem__(self, item):
        return self.cid[item], self.graph_lst[item], self.pop_lst[item], \
            self.src_pos_t[item], self.classes[item], self.pop_label[item], \
            self.tgt_label[item], self.tgt_pos_t[item], self.pln[item], self.fdn[item], self.sdn[item], self.fd_lst[item], self.sd_lst[item]

    def __len__(self):
        return len(self.cid)

def collen_fn(batch):
    """
    """
    cid, graph_lst, pop_lst, src_pos_t, classes, pop_label, tgt_label, tgt_pos_t, pln, fdn, sdn, fd_lst, sd_lst = zip(*batch)
    batch_graphs = []
    for i in range(len(graph_lst[0])):
        batch_g = [gs[i] for gs in graph_lst]
        dgl_graph = [dgl.from_networkx(g, node_attrs=['time', 'deg', 'nid'], edge_attrs=['weight']) for g in batch_g]
        batch_graphs.append(dgl.batch(dgl_graph))
    src_pop_tr = torch.FloatTensor(np.array(pop_lst)) # src
    src_pos_t = torch.LongTensor(np.array(src_pos_t))
    labels = torch.LongTensor(classes)
    pop_labels_tr = torch.FloatTensor(np.array(pop_label))
    tgt_label = torch.FloatTensor(np.array(tgt_label))
    tgt_pos_t = torch.LongTensor(np.array(tgt_pos_t))
    cids = torch.LongTensor([int(c) for c in cid])

    pln_tr = torch.FloatTensor(np.array(pln))
    fdn_tr = torch.FloatTensor(np.array(fdn))
    fdn_tr = F.pad(fdn_tr, (2, 0), 'constant', 0)
    sdn_tr = torch.FloatTensor(np.array(sdn))
    sdn_tr = F.pad(sdn_tr, (3, 0), 'constant', 0)

    src_fd_tr = torch.FloatTensor(np.array(fd_lst))
    src_sd_tr = torch.FloatTensor(np.array(sd_lst))

    return batch_graphs, src_pop_tr, src_pos_t, labels, pop_labels_tr, tgt_label, tgt_pos_t, cids, pln_tr, fdn_tr, sdn_tr, src_fd_tr, src_sd_tr

def get_batch_data(path, batch_size, seq_len, label_len, type='train'):
    dataset = CasData(path, seq_len, label_len)
    if type == 'train':
        return Data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collen_fn, shuffle=True)
    elif type == 'valid':
        return Data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collen_fn, shuffle=False)
    else:
        return Data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collen_fn, shuffle=False)