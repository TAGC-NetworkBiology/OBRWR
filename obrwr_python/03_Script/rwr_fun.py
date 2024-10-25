import networkx as nx
import numpy as np
import copy as cp


def unbiased_RWR(G,beta,sensitivity):
    # Computes Random Walk with Restart distribution on nodes
    # Using iterative method
    # beta (float) : restart probability
    # sensitivity (float) : threshold on convergence
    tmp = cp.deepcopy(G.pr)
    tmp1 = np.ones(G.N)
    
    while np.linalg.norm(tmp1-tmp) > sensitivity:
        tmp1 = cp.deepcopy(tmp)
        tmp = beta*G.pr + (1-beta)*G.W_sparse@tmp
    return tmp

def biased_RWR(G,eps,beta,sensitivity):
    if G.optimized :
        tmp = cp.deepcopy(G.pr)
        tmp1 = np.ones(G.N)
        M = ((1-eps)*(1-beta)*G.W_sparse + eps*(1-beta)*G.B)
        while np.linalg.norm(tmp1-tmp) > sensitivity :
            tmp1 = cp.deepcopy(tmp)
            tmp = beta*G.pr + M@tmp
        return tmp
    else:
        raise Exception("Optimization not run yet")

