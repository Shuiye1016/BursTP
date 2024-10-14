import pickle
import numpy as np
from global_params import get_params
import random
from collections import Counter
from params import cur_data

seed = 2024
random.seed(seed)

def write_lines(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)

dataname = cur_data
observation, _, _, _, _, _, _, wpath, _ = get_params(dataname)
train_txt_path = wpath + 'train.txt'
val_txt_path = wpath + 'val.txt'
test_txt_path = wpath + 'test.txt'
nodes_path = wpath + 'nodes.pkl'
data_path = wpath + '{}_sample.txt'.format(dataname)

val_test_ratio = 0.3

with open(data_path, 'r') as f1:
    lines = f1.readlines()
    samples = len(lines)
    val_samples = []
    test_samples = []
    train_samples = []
    nodes = []

    sort_lines = sorted(lines, key=lambda x:int(x.split('\t')[6]))

    all_lines = sort_lines
    random.shuffle(all_lines)
    split_n = len(all_lines) * val_test_ratio

    i = 0
    k = 0
    for line in all_lines:
        cascades = line.split('\t')[4].split(' ')
        for c in cascades:
            cnode = c.split(':')[0].split('/')
            nodes.extend(cnode)

        if k < split_n // 2:
            test_samples.append(line)
        elif k < split_n:
            val_samples.append(line)
        else:
            train_samples.append(line)
        k += 1

    write_lines(train_txt_path, train_samples)
    write_lines(val_txt_path, val_samples)
    write_lines(test_txt_path, test_samples)
    with open(wpath + "nodes.pkl", 'wb') as f:
        pickle.dump(set(nodes), f)
    node2ind = dict()
    for i, node in enumerate(set(nodes)):
        node2ind[node] = i+1
    with open(wpath + "node2ind.pkl", 'wb') as f:
        pickle.dump(node2ind, f)