import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx 
import pickle 

from heapq import *

def path_weight(G, path,weight,tot=1e4,beta=0.7):
    cost=1
    if not nx.is_path(G, path):
        raise nx.NetworkXNoPath("path does not exist")
    for node, nbr in nx.utils.pairwise(path):
        d = G.get_edge_data(node,nbr)
        cost *= (1-beta)*d[weight]
    return beta*tot*cost

def to_probas(distances):
    for node1 in distances.keys():
        for node2 in distances[node1].keys():
            distances[node1][node2] = np.exp(-distances[node1][node2])

def maxprob_to_targets(node,targets_left,sp_distances,Ts):
    maxprob = 0
    for target in targets_left:
        d = sp_distances[node][target]
        if d > maxprob:
            maxprob = d
    return maxprob
    
def enumerate_path_heap(G,targets,experiment,source='SRC_HUMAN',beta=0.7,r=1e-1,Tot=1e4,verbose=False,mode='total'):
    weight_sp = lambda u,v,d: -np.log((1-beta)*d['weights'+experiment])
    Ts = {t:G.nodes[t]['Pi'+experiment] for t in targets}
    pathstotarget = {t:[[],0] for t in targets}
    sp_distances = dict(nx.all_pairs_dijkstra_path_length(G,weight = weight_sp))
    to_probas(sp_distances)
    if verbose :
        print('Ts :',Ts)
    weight = lambda d: d['weights'+experiment]
    
    initial_item = [-beta*Tot,beta*Tot,[source]]
    h = []
    heappush(h,initial_item)
    
    targets_left = set(targets)
    while len(targets_left) and len(h):
        dist2target, path_prob, path = heappop(h)
        if path[-1] in targets_left:
            tar = path[-1]
            pathstotarget[tar][0].append(path)
            pathstotarget[tar][1] += path_prob
            tmp_b = False
            if mode == 'percent' and path_prob/Ts[tar] < r:
                targets_left.remove(tar)
                tmp_b = True
            elif mode == 'total' and pathstotarget[tar][1] > (1-r)*Ts[tar]:
                targets_left.remove(tar)
                tmp_b = True
            if tmp_b and verbose:
                print('Target found :',tar)
                print('Probas found :',{t:pathstotarget[t][1] for t in targets})
                print('targets_left :', targets_left)
        for neigh in nx.neighbors(G,path[-1]):
            wedge = (1-beta)*weight(G.edges[(path[-1],neigh)])
            maxp = maxprob_to_targets(neigh,targets_left,sp_distances,Ts)
            heappush(h,[-path_prob*wedge*maxp,path_prob*wedge,path+[neigh]])
    return pathstotarget

