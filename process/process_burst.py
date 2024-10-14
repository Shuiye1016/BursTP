import math

import matplotlib.pyplot as plt
import numpy as np
from global_params import get_params
from itertools import groupby
import pickle as pkl
import time
from collections import Counter
from params import cur_data
from sklearn.preprocessing import StandardScaler

theta = 2
alpha = 0.6
beta = 0.3
gamma = 0.1

scaler = StandardScaler()

def process_data(dataname):
    observation, unit_time, span, window_size, unit_size, rpath,_, wpath, data_type = get_params(dataname)

    windows = int((unit_time / window_size) * span)

    wu = int(window_size // unit_size)

    ow = int(observation // window_size)
    print("Dataset: {}, theta: {}".format(dataname, theta))
    print("All windows: {}, observation windows: {}".format(windows, ow))
    print("Label param:alpha={}, beta={}, gamma={}".format(alpha, beta, gamma))
    count = 0
    retweet_count = 0
    pop_labels = []
    pop_label_norms = []
    first_derivative_norms = []
    second_derivative_norms = []

    with open(rpath, 'r') as f1, open(wpath + '{}.txt'.format(dataname), 'w') as f2:
        for line in f1:

            cid, rid, pub_time, pop, cascade = line.strip().split('\t')

            cascade_l = cascade.split(' ')

            if dataname.split('_')[0] == 'twitter':
                cascade_l = cascade_l[1:]

            sort_cascade_l = sorted(cascade_l, key=lambda x:int(x.split(':')[-1]))

            rep_time_list = [float(cas.split(':')[-1]) for cas in sort_cascade_l]

            pops = [0] * (wu * windows)

            for k, g in groupby(rep_time_list, key=lambda x:x//unit_size):
                if k < wu * windows:
                    pops[int(k)] = len(list(g))
                else:
                    break

            pop_arr = np.array(pops).reshape((-1, wu))

            each_wd_pop = np.sum(pop_arr, -1)


            if ow >= np.argmax(each_wd_pop) >= windows:
                continue

            pop_list = list(map(str, pops))[:ow * wu]
            pop_label = each_wd_pop[ow: ]

            label, maxp = np.argmax(pop_label), np.max(pop_label)

            if maxp < each_wd_pop[0] + theta:
                continue

            first_derivative = np.diff(pop_label)
            second_derivative = np.diff(first_derivative)
            pop_label_norm = scaler.fit_transform(pop_label.reshape(-1, 1)).flatten()
            first_derivative_norm = scaler.fit_transform(first_derivative.reshape(-1, 1)).flatten()
            second_derivative_norm = scaler.fit_transform(second_derivative.reshape(-1, 1)).flatten()

            combined_metric = alpha * pop_label_norm[:-2] + beta * first_derivative_norm[:-1] + gamma * second_derivative_norm
            label, max_metric = np.argmax(combined_metric), np.max(combined_metric)

            pop_label = list(map(str, pop_label))

            pop_label_norm = list(map(str, pop_label_norm))
            first_derivative_norm = list(map(str, first_derivative_norm))
            second_derivative_norm = list(map(str, second_derivative_norm))

            count += 1
            retweet_count += int(pop)
            pop_labels.append(pop_label)
            # add norms
            pop_label_norms.append(pop_label_norm)
            first_derivative_norms.append(first_derivative_norm)
            second_derivative_norms.append(second_derivative_norm)

            f2.write(cid + '\t' + rid + '\t' + pub_time + '\t' + pop + '\t' + ' '.join(sort_cascade_l) +
                     '\t' + ' '.join(pop_list) + '\t' + str(label) + '\t' + ' '.join(pop_label) +
                     '\t' + ' '.join(pop_label_norm) + '\t' + ' '.join(first_derivative_norm) + '\t' + ' '.join(second_derivative_norm) + '\n')


    return pop_labels, window_size

if __name__ == '__main__':
    startTime = time.time()
    pops, window_size = process_data(dataname=cur_data)
    endTime = time.time()
    useTime = (endTime - startTime) / 60
    print("process %s" % useTime)