import networkx as nx
import numpy as np
from global_params import get_params
import pickle
from params import cur_data
import time
def get_graph(dataname):
    observation, _, _, _, _, _, rppath, wpath, _ = get_params(dataname)

    node2time = dict()
    G = nx.DiGraph()
    with open(rppath, 'r') as f:
        for line in f:
            cascade = line.split('\t')[4].split(' ')
            n2t = dict()
            cid = line.split('\t')[0]
            for i, c in enumerate(cascade):
                edge = c.split(':')[0].split('/')

                t = c.split(':')[-1]
                if edge[-1] not in n2t:
                    n2t[edge[-1]] = t
                if i == 0:
                    if cur_data.split('_')[0] == 'weibo' or cur_data == 'repost' or cur_data == 'topic':
                        G.add_node(edge[0])
                    else:
                        G.add_edge(edge[0], edge[1])
                else:
                    G.add_edge(edge[-2], edge[-1])
            node2time[cid] = n2t
    with open(wpath + 'global_g.pkl', 'wb') as f:
        pickle.dump(G, f)
    degrees = {dg[0]:dg[1] for dg in G.out_degree}
    with open(wpath + 'degrees.pkl', 'wb') as f:
        pickle.dump(degrees, f)
    with open(wpath + 'node2time.pkl', 'wb') as f:
        pickle.dump(node2time, f)

if __name__ == '__main__':
    startTime = time.time()
    get_graph(dataname=cur_data)
    endTime = time.time()
    useTime = (endTime - startTime) / 60
    print("generate global %s" % useTime)