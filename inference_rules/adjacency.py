import numpy as np

def calc_adjacency(data, e_k, r_k):
    adj = np.zeros((r_k, e_k, e_k), dtype=np.float32)
    for d in data:
        adj[d[1],d[0],d[2]] = 1
    return adj