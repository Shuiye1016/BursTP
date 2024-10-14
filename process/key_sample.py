import networkx as nx
import numpy as np
from global_params import get_params
import pickle
import random
from params import cur_data
import time

seed = 712
random.seed(seed)
np.random.seed(seed)

def sample(path, degrees):
    path_deg = []

    for p in path:
        nodes = p.split(':')[0].split('/')
        deg = 0
        for node in nodes:
            deg += degrees[node] + 1
        path_deg.append(deg)

    index = np.argsort(path_deg)[::-1][:512]
    new_path = np.array(path)[index]

    return list(sorted(new_path, key=lambda x:int(x.split(':')[-1])))

def sample_cascade(dataname):
    observation, _, _, _, _, _, rppath, wpath,_ = get_params(dataname)
    count = 0
    with open(rppath, 'r') as f1, open(wpath + 'degrees.pkl', 'rb') as f2, open(wpath + '{}_sample.txt'.format(dataname), 'w') as f3:
        degrees = pickle.load(f2)
        for line in f1:
            opop = 0
            cid, rid, pub_time, pop, cascade, pops, label, pop_label, pop_label_norm, first_derivative_norm, second_derivative_norm = line.split('\t')
            observation_path = []
            all_nodes = []
            for c in cascade.split(' '):
                flag2 = 0
                time = int(c.split(':')[-1])
                nodes = c.split(':')[0].split('/')
                if time < observation:

                    flag1 = sum(map(lambda x: nodes[-1] + ":" in x, observation_path))
                    if flag1 > 0: continue

                    if len(nodes) >= 3:
                        for node in nodes[1:-1]:
                            if node not in all_nodes:
                                flag2 = 1
                                break
                    if flag2: continue
                    observation_path.append(c)
                    opop += 1
                else:
                    break
                all_nodes.extend(nodes)

            if opop < 10:
                continue


            elif opop > 512:
                new_path = sample(observation_path, degrees)
                # new_path = observation_path[:512]
                f3.write(cid + '\t' + rid + '\t' + pub_time + '\t' + pop + '\t' + ' '.join(new_path) +
                         '\t' + pops + '\t' + str(label) + '\t' + pop_label)
                count += 1

            else:
                f3.write(cid + '\t' + rid + '\t' + pub_time + '\t' + pop + '\t' + ' '.join(observation_path) +
                     '\t' + pops + '\t' + str(label) + '\t' + pop_label + '\t' + pop_label_norm + '\t' + first_derivative_norm + '\t' + second_derivative_norm)
                count += 1

if __name__ == '__main__':
    startTime = time.time()
    sample_cascade(dataname=cur_data)
    endTime = time.time()
    useTime = (endTime - startTime) / 60
    print("sample %s" % useTime)