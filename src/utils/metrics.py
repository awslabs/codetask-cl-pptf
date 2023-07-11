# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        li.append(results[i-1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)

import numpy as np

def get_forgetting_metric(acc_arr, bwt=False, return_mean=False):
    num_tasks = acc_arr.shape[0]
    max_accs = np.max(acc_arr[:-1,:-1], axis=0)
    last_accs = acc_arr[-1, :-1]
    if bwt:
        task_forgetting = last_accs - max_accs
    else:
        task_forgetting = max_accs - last_accs
    if return_mean:
        return np.array(task_forgetting).mean()
    return np.array(task_forgetting)

def get_forward_transfer(acc_arr, random_accs, return_mean=False):
    fwt = []
    for i in range(1, len(acc_arr)):
        fwt.append(acc_arr[i-1, i] - random_accs[i])
    if return_mean:
        return np.array(fwt).mean()
    return np.array(fwt)