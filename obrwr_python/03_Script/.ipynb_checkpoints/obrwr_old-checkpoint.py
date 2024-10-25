import scipy as sp
import scipy.sparse.linalg as spl
import scipy.optimize as spo
import scipy.sparse

import pickle

import gurobipy as gp
from gurobipy import GRB

import networkx as nx
import pandas as pd
import numpy as np


import copy as cp

import matplotlib.pyplot as plt

import seaborn as sns
import plotly

sns.set_theme(style="darkgrid")

##########################
## CONSTANTS DEFINITION ##
##########################

## OBJECTIVES
OBJ_VAR_FROM_STABLE = 'var_from_stable'
OBJ_ABSOLUTE = 'absolute'
OBJ_WEIGHTED_ABSOLUTE = 'weighted_absolute'
OBJ_QUADRATIC_FROM_STABLE = 'quadratic_from_stable'


def get_list_from_components(components):
    #returns list of nodes of each components 
    #in a list
    l = []
    for el in components:
        l += list(el)
    return l

def map_list(liste,mapping):
    # The signaature should speak for itself :
    # (a list * (a->b) dict ) -> b  list
    return [mapping[el] for el in liste]

class MyGraph:
    # This is the class which defines an object which encapsulates all information relevant
    # for both running the optimization and the plots.
    # It allows easy usage of the method

    def __init__(self,G,self_loop=False):
        if not G.is_directed():
            self.sorted_components = sorted(nx.connected_components(G),key=len,reverse=True)
        else:
            self.sorted_components = sorted(nx.weakly_connected_components(G),key=len,reverse=True)
        # self.sorted_components stores the components of G sorted by their size (n° of nodes)
        
        self.subG = G.subgraph(self.sorted_components[0]).copy()
        # self.subG is the subgraph of the biggest component of the graph

        if self_loop :
            for node in self.subG.nodes:
                self.subG.add_edge(node,node)
        

        self.forgotten_proteins = {"Connectivity" : get_list_from_components(self.sorted_components[1:])}
        # self.forgotten_proteins stores the proteins (nodes) left out at each step

        self.mapping = {el:i for i,el in enumerate(self.subG.nodes)}
        # self.mapping stores the correspondence between node label and integers

        self.inverse_mapping = {v:k for k,v in self.mapping.items()}
        # inverse of self.mapping

        self.subGdirected_with_annot = nx.DiGraph(self.subG.copy())
        # self.subGdirected_with_annot directed version of subG and will
        # also store different type of annotation

        self.e_l = list(nx.DiGraph(self.subGdirected_with_annot).edges)
        # self.e_l is the list of edges in subGdirected_with_annot 
       
        self.is_int = False

        self.map_elements_to_int()
        
        self.gen_instance()



    def gen_instance(self) :
        # Creates all graph theoretical related variables

        self.N = len(self.subGdirected_with_annot.nodes)
        # self.N is the size of the network

        self.pr = np.zeros(self.N)
        self.pr[0] = 1
        # self.pr vector of initial weights (set to 1 for the first elt and zero otherwise)
        self.sum_nodes = sum(self.pr)
        
        self.tar = np.zeros(self.N)
        # self.tar vector of target values 
        
        self.A_sparse = sp.sparse.csr_matrix(nx.linalg.graphmatrix.adjacency_matrix(self.subGdirected_with_annot,
                                                                                    nodelist=range(self.N)))
        self.A = self.A_sparse.todense()
        #  self.A is the ADJACENCY matrix of the graph subGdirected_with_annot

        self.D = np.asmatrix(np.diag([int(el[0]) for el in np.sum(self.A, axis=1)]))
        self.D_inv = np.asmatrix(np.diag([1/int(el[0]) for el in np.sum(self.A, axis=1)]))
        self.D_sparse = sp.sparse.csr_matrix(self.D)
        self.D_inv_sparse = spl.inv(self.D_sparse)
        # D is the diagonal matrix of degrees

        self.W_sparse = self.A_sparse*self.D_inv_sparse
        self.W = self.W_sparse.todense()
        # W is the walk matrix in the case of uniform RWR

        self.I = sp.sparse.eye(self.N,format='csc')
        # Identitiy matrix in sparse format
        
        self.m = len(self.e_l)
        # self.m is the number of edges in the (directed) graph

        self.outM = sp.sparse.csc_matrix((np.ones(self.m),([el[0] for el in self.e_l],list(range(self.m)))),shape=(self.N,self.m))
        self.inM = sp.sparse.csc_matrix((np.ones(self.m),([el[1] for el in self.e_l],list(range(self.m)))),shape=(self.N,self.m))
        # self.outM is a Nxm matrix on the kth row there is a one at line e_l[k][0] (source of kth edge)
        # self.inM is a Nxm matrix on the kth row there is a one at line e_l[k][1] (target of kth edge)
        
        self.optimized = False
        #Boolean to know wether optimization has taken place or not
        self.is_set_stable = False
        #Boolean to know wether optimization has taken place or not

        self.dinv_array = np.array([self.D_inv[i[0],i[0]] for i in self.e_l])
        self.Dinv_forpi = sp.sparse.csc_matrix((self.dinv_array,(list(range(self.m)),[el[0] for el in self.e_l])))

    def map_elements_to_int(self):
        #maps node labels to int based on self.mapping 
        if not self.is_int:
            nx.relabel_nodes(self.subG,self.mapping,copy=False)
            nx.relabel_nodes(self.subGdirected_with_annot,self.mapping,copy=False)
            self.e_l = [(self.mapping[e[0]],self.mapping[e[1]]) for e in self.e_l]
            self.is_int = True

    def map_elements_to_names(self):
        #maps back node labels from int to what they were originally
        if self.is_int:
            nx.relabel_nodes(self.subG,self.inverse_mapping,copy=False)
            nx.relabel_nodes(self.subGdirected_with_annot,self.inverse_mapping,copy=False)
            self.e_l = [(self.inverse_mapping[e[0]],self.inverse_mapping[e[1]]) for e in self.e_l]
            self.is_int = False

    def unbiased_RWR(self,beta,sensitivity):
        tmp = cp.deepcopy(self.pr)
        tmp1 = np.ones(self.N)
        while np.linalg.norm(tmp1-tmp) > sensitivity:
            tmp1 = cp.deepcopy(tmp)
            tmp = beta*self.pr + (1-beta)*self.W_sparse@tmp
        return tmp

    def set_stable(self,method):
        if method[0] == 'unbiased':
            self.stable = self.unbiased_RWR(method[1],1e-10)
            nx.set_edge_attributes(self.subGdirected_with_annot,{edge:1/self.subG.degree[edge[0]] for edge in self.subGdirected_with_annot.edges},"WeightsStable")
        nx.set_node_attributes(self.subGdirected_with_annot,{key:val for key,val in enumerate(self.stable)},"Stable")
        self.is_set_stable = True

    def biased_RWR(self,eps,beta,sensitivity):
        if self.optimized :
            tmp = cp.deepcopy(self.pr)
            tmp1 = np.ones(self.N)
            M = ((1-eps)*(1-beta)*self.W_sparse + eps*(1-beta)*self.B)
            while np.linalg.norm(tmp1-tmp) > sensitivity :
                tmp1 = cp.deepcopy(tmp)
                tmp = beta*self.pr + M@tmp
            return tmp
        else:
            raise Exception("Optimization not run yet")

    def check_protein_list(self,protein_list,category,is_mapped=False):
        tmp = []
        self.forgotten_proteins[category] = []
        if not is_mapped :
            for el in protein_list:
                if el in self.mapping.keys():
                    tmp.append(self.mapping[el])
                else:
                    self.forgotten_proteins[category].append(el)
        else:
            for el in protein_list:
                if el in self.inverse_mapping.keys():
                    tmp.append(el)
                else:
                    self.forgotten_proteins[category].append(el)
        return tmp

    def remove_unreachable_nodes(self,sources_list,is_mapped=False):
        reachable = set(sources_list)
        for el in sources_list:
            reachable = reachable | nx.descendants(self.subGdirected_with_annot,el)
        removed = set(self.subGdirected_with_annot.nodes) - reachable
        self.forgotten_proteins["Unreachable from sources"] = [self.inverse_mapping[iprot] for iprot in removed]
        self.subGdirected_with_annot.remove_nodes_from(removed)
        self.subG.remove_nodes_from(removed)

    def remap(self):
        tmp_mapping = {el:i for i,el in enumerate(self.subGdirected_with_annot.nodes)} 
        self.subGdirected_with_annot = nx.relabel_nodes(self.subGdirected_with_annot,tmp_mapping,copy=True)
        self.subG = nx.relabel_nodes(self.subG,tmp_mapping,copy=True)
        self.mapping = {self.inverse_mapping[el]:i for el,i in tmp_mapping.items()}
        self.inverse_mapping = { i:prot for prot,i in self.mapping.items()}

    def set_sources(self,sources_list,sum_nodes,is_mapped=False):
        sources_list = self.check_protein_list(sources_list,"sources not in",is_mapped)
        self.remove_unreachable_nodes(sources_list,is_mapped=True)
        self.remap()
        self.gen_instance()
        nsources = len(sources_list)
        self.pr = np.zeros(self.N)
        for i in sources_list:
            self.pr[i] = sum_nodes/nsources
        self.sum_nodes = sum_nodes
        nx.set_node_attributes(self.subGdirected_with_annot,{i:(i in sources_list) for i in self.subGdirected_with_annot.nodes},"Sources")
        return sources_list

    def set_targets(self,dict_of_values,is_mapped=False):
        targets = self.check_protein_list(list(dict_of_values.keys()),"targets not in",is_mapped)
        dict_of_values = {target:dict_of_values[self.inverse_mapping[target]] for target in targets}
        nx.set_node_attributes(self.subGdirected_with_annot,{i:(i in dict_of_values.keys()) for i in self.subGdirected_with_annot.nodes},"Targets")
        for k,v in dict_of_values.items():
            self.tar[k] = v
            print(self.inverse_mapping[k],v)

    def add_optimization_variables(self,eps,beta):
        piX = self.pi.X
        EX = self.E.X
        self.b = EX/np.array([piX[i[0]] for i in self.e_l])
        count = {i:0 for i in range(self.N)}
        for z,el in enumerate(self.e_l):
            count[el[0]] += self.b[z]
        self.b = [(self.b[i]/count[el[0]] if self.b[i] else 0) for i,el in enumerate(self.e_l)]
        self.PR = sp.sparse.csc_matrix(np.tensordot(self.pr,[1 for _ in range(self.N)],axes=0))
        self.B = sp.sparse.csc_matrix((self.b,([e[1] for e in self.e_l],[e[0] for e in self.e_l])))
        self.W_hat = (1-eps)*self.W + eps*self.B

    def add_meta_after_optim(self,eps,beta,name):
        edge_weights = (1-eps)*self.W_sparse + eps*self.B
        nx.set_node_attributes(self.subGdirected_with_annot,{key:val for key,val in enumerate(self.pi.X)},"Pi"+name)
        nx.set_node_attributes(self.subGdirected_with_annot,{key:np.log2(val/self.stable[key]) for key,val in enumerate(self.pi.X)},"Proba_logratio"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                {(i,j):edge_weights[j,i] for (i,j) in self.subGdirected_with_annot.edges},
                                "weights"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                {(i,j):self.B[j,i] for (i,j) in self.subGdirected_with_annot.edges},"B"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                 {(i,j):edge_weights[j,i]*self.pi.X[i]/self.sum_nodes for (i,j) in self.subGdirected_with_annot.edges},
                                "Proba_edge_optim"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                 {(i,j):self.W_sparse[j,i]*self.stable[i]/self.sum_nodes for (i,j) in self.subGdirected_with_annot.edges},
                                "Proba_edge_stable"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                 {(i,j):np.log2(edge_weights[j,i]*self.pi.X[i]/(self.W_sparse[j,i]*self.stable[i])) for (i,j) in self.subGdirected_with_annot.edges},
                                "Proba_edge_logratio")

    def control_optimization(self,eps,beta):
        if self.optimized:
            tmp = self.biased_RWR(eps,beta,1e-10)
            m = min(min(np.log(tmp)),min(np.log(self.pi.X))) - 2
            M = max(max(np.log(tmp)),max(np.log(self.pi.X))) + 2
            plt.figure()
            sns.lineplot(x=[m,M],y=[m,M],ls='--',color='r',alpha=0.9)
            sns.scatterplot(x=np.log(tmp),y=np.log(self.pi.X),alpha=.5, s=60)
            #sns.lineplot(x=[plt.xlim()[0],plt.xlim()[1]],y=[plt.xlim()[0],plt.xlim()[1]],ls='--',color='r')
            ax = plt.gca()
            ax.set_ylim(m,M)
            ax.set_xlim(m,M)
            plt.ylabel("Optimization Probabilities")
            plt.xlabel("Recomputed Probabilities from edge weights")
            plt.title("Control for correction of computed probabilities")

    def get_directed_subgraph(self):
        if self.optimized:
            dig = self.subGdirected_with_annot.copy()
            edges_to_keep = [(self.e_l[i][0],self.e_l[i][1]) for i in range(len(self.E.X)) if self.b[i] > 0]
            dig.remove_edges_from([el for el in dig.edges if el not in edges_to_keep])
        else:
            raise Exeption("Optimization not run")
        return dig

    def get_directed_subgraph_from_sources(self):
        dig = self.get_directed_subgraph()
        tmp = set()
        is_source = nx.get_node_attributes(self.subGdirected_with_annot, "Sources")
        for node in self.subGdirected_with_annot.nodes:
            if is_source[node]:
                print(nx.descendants(dig,node))
                tmp = tmp | nx.descendants(dig,node) | set([node])
        return dig.subgraph(tmp)

    def get_higher_edge_proba_subgraph(self):
        dig = self.subGdirected_with_annot.copy()
        edges_to_remove = [(s,e) for s,e,v in dig.edges(data=True) if (v["Proba_edge_logratio"] < 1 and v["Proba_edge_logratio"] > -1)]
        dig.remove_edges_from(edges_to_remove)
        return dig

    def plot_objective_eps(self,beta,n=20):
        Eps = np.arange(0,n)/n
        plt.figure()
        obj = [self.optimize_biased_walk_normalized(e,beta)[4].getObjective().getValue() for e in Eps]
        sns.lineplot(x=Eps,y=obj)
        plt.ylabel("Objective optimum")
        plt.xlabel("Bias strength (epsilon)")
        plt.title("Bias strength controls objective optimum")


    def find_eps_critic(self,beta,sensitivity):
        a,b = 1e-4,1-1e-4
        d = self.optimize_biased_walk_normalized(b,beta)[4].getObjective().getValue()
        while b-a > sensitivity:
            m = self.optimize_biased_walk_normalized((a+b)/2,beta)[4].getObjective().getValue()
            if m == d:
                b = (a+b)/2
            else:
                a = (a+b)/2
        return (a+b)/2

    def set_objective(self,objective):
        if objective == OBJ_VAR_FROM_STABLE:
            c = self.tar
            self.mo.setObjective(np.ones(self.ntarget)@self.X,GRB.MINIMIZE)
            xi = 0
            for i,val in enumerate(self.tar): 
                if val != 0:
                    self.mo.addConstr(self.X[xi] >= self.pi[i]*self.norm[i][i]/val - 1)
                    self.mo.addConstr(self.X[xi] >= 1 - self.pi[i]*self.norm[i][i]/val)
                    xi+=1
        elif objective == OBJ_ABSOLUTE:
            c = np.zeros(len(self.tar))
            for i,el in enumerate(self.tar):
                if el > 0: 
                    c[i] = 1
                elif el < 0 :
                    c[i] = -1
            self.mo.setObjective(-c@(self.norm@self.pi),GRB.MINIMIZE)
        elif objective == OBJ_WEIGHTED_ABSOLUTE:
            self.mo.setObjective(-self.tar@(self.norm@self.pi),GRB.MINIMIZE)
        elif objective == OBJ_QUADRATIC_FROM_STABLE:
            c = np.zeros(len(self.tar))
            N = self.norm.copy()
            for i,el in enumerate(self.tar):
                if el != 0: 
                    c[i] = 1
                else:
                    N[i][i] = 0
            self.mo.setObjective(self.tar@(np.diag(c)@self.tar) + self.pi@((N@N)@self.pi) - ((2*self.tar)@N)@self.pi,GRB.MINIMIZE)


    def optimize_biased_walk_normalized(self,eps,beta,OF=0,objective=OBJ_VAR_FROM_STABLE):
        if not self.is_set_stable:
            self.set_stable(["unbiased",beta])

        self.norm = np.linalg.inv(np.diag(self.stable))
        
        self.mo = gp.Model('Test')

        self.mo.Params.OutputFlag = OF
        self.mo.Params.Method = 2
        #mo.Params.Crossover = 0
        self.mo.Params.NumericFocus = 3

        
        self.pi = self.mo.addMVar(shape=self.N,vtype=GRB.CONTINUOUS,name='pi',lb=0,ub=self.sum_nodes)
        self.E = self.mo.addMVar(shape=self.m,vtype=GRB.CONTINUOUS,name='E',lb=0,ub=self.sum_nodes)

        self.ntarget = int(sum(self.tar != 0))
        self.X = self.mo.addMVar(shape = self.ntarget,vtype=GRB.CONTINUOUS,name='X',lb = 0)
        
        #c = D_inv @ tar
        self.set_objective(objective)


        self.mo.addConstr(self.pi - beta*self.pr - (1-eps)*(1-beta)*self.W_sparse@self.pi - eps*(1-beta)*self.inM@self.E == 0, name='c4')
        self.mo.addConstr(self.outM@self.E - self.pi == 0, name='c5')
        self.mo.addConstr(np.ones(self.N)@self.pi == self.sum_nodes,name='c6')

        self.mo.optimize()

        self.optimized = True

        self.add_optimization_variables(eps,beta)

        return self.norm,self.pi,self.E,self.X,self.mo