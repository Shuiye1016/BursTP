import networkx as nx
import numpy as np
import scipy as sp
import pickle
from global_params import get_params
from itertools import groupby
import time
from collections import Counter
from params import cur_data

dataname = cur_data
observation, unit_time, span, window_size, unit_size, _, _, wpath, _ = get_params(dataname)
with open(wpath + 'node2ind.pkl', 'rb') as f:
    node2ind = pickle.load(f)
with open(wpath + 'node2time.pkl', 'rb') as f:
    node2t = pickle.load(f)
with open(wpath + 'degrees.pkl', 'rb') as f:
    node2deg = pickle.load(f)

def hash_deg():
    degs = sorted(set(node2deg.values()))
    hash_degs = {k:int(np.log2(v+1)) for v,k in enumerate(degs)}
    return hash_degs
hash_degs = hash_deg()

def transform(type = 'train'):
    transformed = {"cid":[], "pubt":[], "graph_lst":[], "pop_lst":[], "t_pos":[], "classes":[], "pop_label":[], "pop_label_norm":[], "first_derivative_norm":[], "second_derivative_norm":[], "fd_lst":[], "sd_lst":[]}
    labels = []
    with open(wpath + type + '.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            cid, rid, pub_time, pop, cascade, pops, classes, pop_label, pop_label_norm, first_derivative_norm, second_derivative_norm = line.split('\t')
            transformed["cid"].append(cid)
            transformed["pubt"].append(pub_time)
            transformed["classes"].append(int(classes))
            labels.append(int(classes))
            transformed["pop_label"].append([float(p) for p in pop_label.split(' ')])
            transformed["pop_label_norm"].append([float(p) for p in pop_label_norm.split(' ')])
            transformed["first_derivative_norm"].append([float(p) for p in first_derivative_norm.split(" ")])
            transformed["second_derivative_norm"].append([float(p) for p in second_derivative_norm.split(" ")])
            pops_arr = np.array([float(p) for p in pops.split(' ')])
            pops_arr = pops_arr.reshape((-1, window_size // unit_size))
            transformed["pop_lst"].append(pops_arr.sum(1).tolist())
            np_pop = np.array(pops_arr.sum(1).tolist())
            np_fd = np.diff(np_pop)
            np_sd = np.diff(np_fd)
            np_fd = np.insert(np_fd, 0, 0)
            np_sd = np.insert(np_sd, 0, [0, 0])
            transformed["fd_lst"].append(np_fd.tolist())
            transformed["sd_lst"].append(np_sd.tolist())
            timestamps = [int(pub_time) + window_size * (i + 1) for i in range(int(span * unit_time / window_size))]
            t_positions = []

            previous_timestamp = int(pub_time)

            for stamp in timestamps:
                realtime = time.localtime(stamp)
                d, h, m, s = realtime.tm_mday - 1, realtime.tm_hour, realtime.tm_min, realtime.tm_sec
                time_diff = stamp - previous_timestamp
                t_positions.append([d, h, m, s])
                previous_timestamp = stamp
            transformed['t_pos'].append(t_positions)

            g = nx.DiGraph()
            edges = []
            g_list = []
            previous_time = int(pub_time)
            for i, path in enumerate(cascade.split(" ")):
                nodes = path.split(":")[0].split("/")
                for j, node in enumerate(nodes):

                    if j == 0:
                        spant = 0
                    else:
                        spant = int(node2t[cid][node])
                    current_time = int(pub_time) + spant
                    realtime = time.localtime(current_time)
                    d, h, m, s = realtime.tm_mday-1, realtime.tm_hour, realtime.tm_min, realtime.tm_sec
                    timed = current_time - previous_time
                    previous_time = current_time

                    g.add_node(node2ind[node], time=[d, h, m, s, timed], deg=hash_degs[node2deg[node]], nid=node2ind[node])

                if i == 0:

                    edges.append((node2ind[nodes[0]],node2ind[nodes[0]],0))
                else:
                    for j in range(1, len(nodes)):
                        edges.append((node2ind[nodes[j-1]],node2ind[nodes[j]],node2t[cid][nodes[j]]))
            edges = sorted(edges, key=lambda x:int(x[-1]))
            edges_dict = dict()

            for k, v in groupby(edges, lambda x:int(x[-1]) // window_size):
                edges_dict[k] = list(v)
            for i in range(int(observation // window_size)):
                if i in edges_dict.keys():
                    tempg = g.copy()
                    wt_edges = [(edge[0], edge[1], float(edge[2])/observation) for edge in edges_dict[i]]
                    tempg.add_weighted_edges_from(wt_edges)
                    g = tempg.copy()
                else:
                    tempg = g.copy()
                g_list.append(tempg)
            transformed["graph_lst"].append(g_list)
    with open(wpath + type + '.pkl', 'wb') as f:
        pickle.dump(transformed, f)

if __name__ == '__main__':
    startTime = time.time()
    transform(type = 'train')
    transform(type = 'val')
    transform(type = 'test')
    print("done")
    endTime = time.time()
    useTime = (endTime - startTime) / 60
    print("pkl trans%s" % useTime)