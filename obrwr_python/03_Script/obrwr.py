
import scipy as sp

import scipy.sparse.linalg as spl
import scipy.optimize as spo
import scipy.sparse

import pickle
import os

import pyomo.environ as pyo
import highspy as hp

import solution_parser as sparse

import networkx as nx
import pandas as pd
import numpy as np


import copy as cp
import matplotlib.pyplot as plt

import seaborn as sns
import plotly

import textalloc as ta

import random as rd

import sys
import time

sns.set_theme(style="darkgrid")

##########################
## CONSTANTS DEFINITION ##
##########################

## OBJECTIVES
OBJ_VAR_FROM_STABLE = 'var_from_stable'
OBJ_ABSOLUTE = 'absolute'
OBJ_WEIGHTED_ABSOLUTE = 'weighted_absolute'
OBJ_QUADRATIC_FROM_STABLE = 'quadratic_from_stable'

ALPHANUM =  [chr(i) for i in range(ord('a'),ord('z')+1)] +\
            [chr(i) for i in range(ord('A'),ord('Z')+1)] +\
            [chr(i) for i in range(ord('1'),ord('9')+1)]


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
        if not G.is_directed(): # Store the cc of G sorted by their size (n° of nodes)
            self.sorted_components = sorted(nx.connected_components(G),key=len,reverse=True)
        else:
            self.sorted_components = sorted(nx.weakly_connected_components(G),key=len,reverse=True)
        
        
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


    
    def gen_instance(self,reset_targets=True,reset_pr=True) :
        # Creates all graph theoretical related variables
        self.e_l = list(nx.DiGraph(self.subGdirected_with_annot).edges)
        self.N = len(self.subGdirected_with_annot.nodes)
        # self.N is the size of the network

        if reset_pr:
            self.pr = np.zeros(self.N)
            self.pr[0] = 1
            # self.pr vector of initial weights (set to 1 for the first elt and zero otherwise)
            self.sum_nodes = sum(self.pr)
            self.is_set_stable = False
        
        if reset_targets :
            self.tar = np.zeros(self.N)
        # self.tar vector of target values 
        
        self.A_sparse = sp.sparse.csr_matrix(nx.linalg.graphmatrix.adjacency_matrix(self.subGdirected_with_annot,
                                                                                    nodelist=range(self.N))).T
        self.A = self.A_sparse.todense()
        #  self.A is the ADJACENCY matrix of the graph subGdirected_with_annot
        self.D = np.asmatrix(np.diag([int(el) for el in np.sum(np.array(self.A), axis=0)]))
        self.D_inv = np.asmatrix(np.diag([1/int(el) for el in np.sum(np.array(self.A), axis=0)]))
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
        '''
        tmpGraph = self.subGdirected_with_annot.copy()
        for node in tmpGraph.nodes():
            if (node,node) in tmpGraph.edges():
                tmpGraph.remove_edge(node,node)
        A = sp.sparse.csr_matrix(nx.linalg.graphmatrix.adjacency_matrix(tmpGraph,
                                                                                    nodelist=range(self.N)))
        D = np.asmatrix(np.diag([int(el[0]) for el in np.sum(A, axis=1)]))
        D_sparse = sp.sparse.csr_matrix(D)
        D_inv_sparse = spl.inv(D_sparse)
        W_sparse = self.A_sparse*self.D_inv_sparse
        '''
        while np.linalg.norm(tmp1-tmp) > sensitivity:
            tmp1 = cp.deepcopy(tmp)
            tmp = beta*self.pr + (1-beta)*self.W_sparse@tmp
        return tmp

    def set_stable(self,method):
        if method[0] == 'unbiased':
            self.stable = self.unbiased_RWR(method[1],1e-10)
            tmp_directed = self.subG.to_directed()
            nx.set_edge_attributes(self.subGdirected_with_annot,{edge:1/tmp_directed.out_degree[edge[0]] for edge in self.subGdirected_with_annot.edges},"WeightsStable")
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
        sources_list = [self.inverse_mapping[el] for el in sources_list]
        self.remap()
        self.gen_instance()
        sources_list = [self.mapping[el] for el in sources_list]
        nsources = len(sources_list)
        self.pr = np.zeros(self.N)
        for i in sources_list:
            self.pr[i] = sum_nodes/nsources
        self.sum_nodes = sum_nodes
        nx.set_node_attributes(self.subGdirected_with_annot,{i:(i in sources_list) for i in self.subGdirected_with_annot.nodes},"Sources")
        return sources_list

    def set_targets(self,dict_of_values,is_mapped=False,self_loops=False,OF=0):
        targets = self.check_protein_list(list(dict_of_values.keys()),"targets not in",is_mapped)
        dict_of_values = {target:dict_of_values[self.inverse_mapping[target]] for target in targets}
        nx.set_node_attributes(self.subGdirected_with_annot,{i:(i in dict_of_values.keys()) for i in self.subGdirected_with_annot.nodes},"Targets")
        self.tar = np.zeros(self.N)
        for k,v in dict_of_values.items():
            self.tar[k] = v
            if not OF:
                print(self.inverse_mapping[k],v)
            self.subGdirected_with_annot.add_edge(k,k)
            self.subG.add_edge(k,k)
        self.gen_instance(reset_targets=False,reset_pr=False)


    def add_optimization_variables(self,d,eps,beta):
        piX = np.zeros(len(d["pi"]))
        for k,v in d["pi"].items():
            piX[k] = v
        EX = np.zeros(len(d["E"]))
        for k,v in d["E"].items():
            EX[k] = v
        if 'X' in d.keys():
            XX = np.zeros(len(d["X"]))
            for k,v in d["X"].items():
                XX[k] = v
        else:
            XX = None
        self.pi = piX
        self.E = EX
        self.X = XX
        #for j,node in zip([piX[i[0]] for i in self.e_l],[i[0] for i in self.e_l]):
        #    if j == 0:
        #        print(node)
        #        print([piX[i[0]] for i in self.e_l])
        #        raise Exception("Found")
        self.b = EX/np.array([piX[i[0]] for i in self.e_l])
        count = {i:0 for i in range(self.N)}
        for z,el in enumerate(self.e_l):
            count[el[0]] += self.b[z]
        self.b = [(self.b[i]/count[el[0]] if self.b[i] else 0) for i,el in enumerate(self.e_l)]
        self.PR = sp.sparse.csc_matrix(np.tensordot(self.pr,[1 for _ in range(self.N)],axes=0))
        self.B = sp.sparse.csc_matrix((self.b,([e[1] for e in self.e_l],[e[0] for e in self.e_l])))
        self.W_hat = (1-eps)*self.W_sparse + eps*self.B

    def add_meta_after_optim(self,eps,beta,name):
        edge_weights = (1-eps)*self.W_sparse + eps*self.B
        nx.set_node_attributes(self.subGdirected_with_annot,{key:val for key,val in enumerate(self.pi)},"Pi"+name)
        nx.set_node_attributes(self.subGdirected_with_annot,{key:np.log2(val/self.stable[key]) for key,val in enumerate(self.pi)},"Proba_logratio"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                {(i,j):edge_weights[j,i] for (i,j) in self.subGdirected_with_annot.edges},
                                "weights"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                {(i,j):self.B[j,i] for (i,j) in self.subGdirected_with_annot.edges},"B"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                 {(i,j):edge_weights[j,i]*self.pi[i]/self.sum_nodes for (i,j) in self.subGdirected_with_annot.edges},
                                "Proba_edge_optim"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                 {(i,j):self.W_sparse[j,i]*self.stable[i]/self.sum_nodes for (i,j) in self.subGdirected_with_annot.edges},
                                "Proba_edge_stable"+name)
        nx.set_edge_attributes(self.subGdirected_with_annot,
                                 {(i,j):np.log2(edge_weights[j,i]*self.pi[i]/(self.W_sparse[j,i]*self.stable[i])) for (i,j) in self.subGdirected_with_annot.edges},
                                "Proba_edge_logratio")

    def control_optimization(self,beta):
        if self.optimized:
            tmp = self.biased_RWR(self.eps_crit,beta,1e-10)
            ceil = 1e-12
            m = min(min(np.log(tmp)),min(np.log(self.pi))) - 2
            M = max(max(np.log(tmp)),max(np.log(self.pi))) + 2
            plt.figure()
            sns.lineplot(x=[m,M],y=[m,M],ls='--',color='r',alpha=0.9)
            sns.scatterplot(x=np.log(ceil+tmp),y=np.log(ceil+self.pi),alpha=.5, s=60)
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
            edges_to_keep = [(self.e_l[i][0],self.e_l[i][1]) for i in range(len(self.E)) if self.b[i] > 0]
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
        obj = [self.optimize_biased_walk_normalized(e,beta)[4].getObjectiveValue() for e in Eps]
        sns.lineplot(x=Eps,y=obj)
        plt.ylabel("Objective optimum")
        plt.xlabel("Bias strength (epsilon)")
        plt.title("Bias strength controls objective optimum")


    def find_eps_critic(self,beta,sensitivity,OF=0):
        a,b = sensitivity,1-sensitivity
        d = self.optimize_biased_walk_normalized(b,beta,OF=OF)[4].getObjectiveValue()
        while b-a > sensitivity:
            _,pi,_,_,model = self.optimize_biased_walk_normalized((a+b)/2,beta,OF=OF)
            m = model.getObjectiveValue()
            if abs(m - d) < sensitivity :
                b = (a+b)/2
            else:
                a = (a+b)/2
        return (a+b)/2,pi

    def set_objective(self,objective):
        if objective == OBJ_VAR_FROM_STABLE:
            c = self.tar
            self.mo.obj = pyo.Objective(rule = lambda m: pyo.summation(m.X))
            #self.mo.setObjective(np.ones(self.ntarget)@self.X,GRB.MINIMIZE)
            xi = 0
            d = {}
            for i,val in enumerate(self.tar): 
                if val != 0:
                    d[xi] = (i,val)
                    #self.mo.addConstr(self.X[xi] >= self.pi[i]*self.norm[i][i]/val - 1)
                    #self.mo.addConstr(self.X[xi] >= 1 - self.pi[i]*self.norm[i][i]/val)
                    xi+=1
            #for i in range(self.N):
            #    print(self.norm[i][i])
            def rule_abs_1(model,xi):
                i,val = d[xi]
                
                return model.X[xi] >= model.pi[i]*self.norm[i][i]/val - 1
            def rule_abs_2(model,xi):
                i,val = d[xi]
                return model.X[xi] >= 1 - model.pi[i]*self.norm[i][i]/val

            self.mo.abs1 = pyo.Constraint(self.mo.Xrange,rule=rule_abs_1)
            self.mo.abs2 = pyo.Constraint(self.mo.Xrange,rule=rule_abs_2)
        elif objective == OBJ_ABSOLUTE:
            c = np.zeros(len(self.tar))
            for i,el in enumerate(self.tar):
                if el > 0: 
                    c[i] = 1
                elif el < 0 :
                    c[i] = -1
            self.mo.C = pyo.Param(self.mo.N,initialize=lambda m,i:-c[i]*self.norm[i,i])
            self.mo.obj = pyo.Objective(rule=lambda m : pyo.summation(self.mo.C,self.mo.pi))
            #self.mo.setObjective(-c@(self.norm@self.pi),GRB.MINIMIZE)
        elif objective == OBJ_WEIGHTED_ABSOLUTE:
            self.mo.C = pyo.Param(self.mo.N,initialize=lambda m,i:-self.tar[i]*self.norm[i,i])
            self.mo.obj = pyo.Objective(rule=lambda m : pyo.summation(self.mo.C,self.mo.pi))
        #elif objective == OBJ_QUADRATIC_FROM_STABLE:
        #    c = np.zeros(len(self.tar))
        #    N = self.norm.copy()
        #    for i,el in enumerate(self.tar):
        #        if el != 0: 
        #            c[i] = 1
        #        else:
        #            N[i][i] = 0
        #    self.mo.setObjective(self.tar@(np.diag(c)@self.tar) + self.pi@((N@N)@self.pi) - ((2*self.tar)@N)@self.pi,GRB.MINIMIZE)

    def plot_degrees(self,name):
        
        degree_sequence_with_nodes = sorted(((n,d) for n, d in self.subGdirected_with_annot.degree()),key = lambda x : x[1], reverse=True)
        nodes_sequence_degree_sorted = [el[0] for el in degree_sequence_with_nodes]
        degree_sequence = [el[1] for el in degree_sequence_with_nodes]
        
        targets = nx.get_node_attributes(self.subGdirected_with_annot,'Targets')
        #print(degree_sequence_with_nodes)
        target_degree_sequence_with_nodes = [(n,d) for n,d in self.subGdirected_with_annot.degree() if targets[n]]
        target_degree_sequence = [el[1] for el in target_degree_sequence_with_nodes]
        target_index_sequence = [nodes_sequence_degree_sorted.index(el[0]) for el in target_degree_sequence_with_nodes]
        target_name_sequence = [self.inverse_mapping[el[0]].split('_')[0] for el in target_degree_sequence_with_nodes]

        dmax = max(degree_sequence)
        fig = plt.figure("Degree of a "+name+"graph", figsize=(15, 15))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(4, 2)

        ax1 = fig.add_subplot(axgrid[:2, :])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.plot(target_index_sequence,target_degree_sequence,"ro")
        ta.allocate_text(fig,ax1,target_index_sequence,target_degree_sequence,
                target_name_sequence,
                x_scatter=range(len(degree_sequence)), y_scatter=degree_sequence,
                x_lines= [range(len(degree_sequence))],
                y_lines=[degree_sequence],
                min_distance = 0.02,
                max_distance = 0.15,
                margin = 0.015,
                textsize=10)
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")

        ax2 = fig.add_subplot(axgrid[2:, :])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        #ax2.hist(degree_sequence,bins=4)
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()



    def define_bins(self,deg_seq_sorted):
        val,counts = np.unique(deg_seq_sorted,return_counts=True)
        size = counts[0]
        bins = [(0,val[0])]
        bin_counts = [counts[0]]
        c = 0
        for i in range(1,len(val)):
            c+= counts[i]
            if c > size:
                bins.append((bins[-1][1]+1,val[i]))
                bin_counts.append(c)
                c = 0
        bins.append((bins[-1][1],val[-1]))
        bin_counts.append(c)
        return bins,bin_counts


    def sample_random_targets(self,N):
        targets = nx.get_node_attributes(self.subGdirected_with_annot,'Targets')
        degrees = self.subGdirected_with_annot.degree()
        degree_sequence_sorted = sorted([d for n, d in degrees])
        bins,bin_counts = self.define_bins(degree_sequence_sorted)
        bins_nodes = [[] for _ in bins]
        target_counts = [0 for _ in bins]
        random_targets = [[] for _ in range(N)]
        for n,d in degrees:
            for i, (m,M) in enumerate(bins):
                if d <= M:
                    if targets[n]:
                        target_counts[i] += 1
                    else :
                        bins_nodes[i].append(n)
                    break
        for i in range(N):
            rng = np.random.default_rng()
            for nodes,c in zip(bins_nodes,target_counts):
                random_targets[i] += list(rng.choice(nodes,c))
        return random_targets

    def plot_sampling_bins(self,experiment):
        targets = nx.get_node_attributes(self.subGdirected_with_annot,'Targets')
        degrees = self.subGdirected_with_annot.degree()
        degree_sequence_sorted = sorted([d for n, d in degrees])
        bins,bin_counts = self.define_bins(degree_sequence_sorted)
        bins_nodes = [[] for _ in bins]
        target_counts = [0 for _ in bins]
        for n,d in degrees:
            for i, (m,M) in enumerate(bins):
                if d <= M:
                    if targets[n]:
                        target_counts[i] += 1
                    else:
                        bins_nodes[i].append(n)
                    break
        fig = plt.figure("Sampling bins", figsize=(10, 5))
        ax = fig.add_subplot()
        p = ax.bar(range(len(bins)),bin_counts,tick_label=[str(el) for el in bins])
        ax.bar_label(p,labels=[str(el) + " targets" for el in target_counts])
        ax.set_xlabel("Degrees of nodes in bins",fontsize=14)
        ax.set_ylabel("Number of nodes in each bins")
        ax.set_title("Overview of degree-defined sampling bins "+experiment,fontsize=18)
        plt.show()


    def get_significative_nodes(self,beta,N,experiment="Test",folder='Test'):
        self.map_elements_to_int()
        self.eps_crit,pi = self.find_eps_critic(beta,1e-2,OF=1)
        #_,pi,_,_,_ = self.optimize_biased_walk_normalized(eps_crit,beta)
        df_experiment = pd.DataFrame({experiment:{self.inverse_mapping[j]:pi[j] for j in range(len(pi))}})
        self.add_meta_after_optim(self.eps_crit,beta,experiment)
        df,df_tar,df_eps = self.get_random_runs(beta,N,folder)
        def compute_pval_oneside(row):
            s = 0
            for i in range(N):
                if df_experiment[experiment][row.name] < row[i]:
                    s += 1
            return (s+1)/(N+1)
        df_experiment = df_experiment.join(pd.DataFrame(df.apply(compute_pval_oneside,axis=1),columns=["pvals"]))
        nx.set_node_attributes(self.subGdirected_with_annot,\
                               {key:df_experiment["pvals"][self.inverse_mapping[key]] for key,val in enumerate(pi)},
                               "pvals"+experiment)
        return df,df_tar,df_eps,df_experiment
        
    def get_random_runs(self,beta,N,folder):
        df = pd.DataFrame(index=self.inverse_mapping.values())
        targets = nx.get_node_attributes(self.subGdirected_with_annot,'Targets')
        target_degree_sorted = sorted([(n,d) for n, d in self.subGdirected_with_annot.degree() if targets[n]],
                                        key = lambda x : x[1])
        tar_vals = [self.tar[el[0]] for el in target_degree_sorted]
        df_target_values = pd.DataFrame(tar_vals,columns=["Value"])
        sample_list = self.sample_random_targets(N)
        for i,l in enumerate(sample_list) :
            df_target_values = df_target_values.join(pd.DataFrame(list(map(lambda x : self.inverse_mapping[x],l)),columns=[i]))
        epsilons = []
        times = []
        T0 = time.time()
        for i in range(N):
            b = True
            while(b):
                try:
                    tmp = cp.deepcopy(self)
                    #print({k:v for k,v in zip(sample_list[i],tar_vals)})
                    tmp.set_targets({tmp.inverse_mapping[k]:v for k,v in zip(sample_list[i],tar_vals)},OF=1)
                    #print(tmp.tar)
                    eps_crit,pi = tmp.find_eps_critic(beta,1e-2,OF=1)
                    b = False
                except:
                    sample_list[i] = self.sample_random_targets(1)[0]
                    b = True
            epsilons.append(eps_crit)
            #   _,pi,_,_,_= tmp.optimize_biased_walk_normalized(eps_crit,beta,OF=1)
            df_dictionary = pd.DataFrame({i:{tmp.inverse_mapping[j]:pi[j] for j in range(len(pi))}})
            tmp.save_run_in(folder,df_dictionary,eps_crit,df_target_values[["Value",i]])
            df = df.join(df_dictionary)
            if i%5 == 4:
                T1 = time.time()
                times.append(T1 - T0)
                T0 = time.time()
                print(str(i) + ' / ' + str(N))
                print(df.head())
                print(f'Time for last 5 runs : {times[-1]/60} min.')
                print(f'Average time per 5 runs : {np.mean(times)/60} min.')
        return df,df_target_values,pd.DataFrame(epsilons)

    def save_run_in(self,folder,df_pi,eps_crit,df_targets):
        exp_id = "".join(rd.choices(ALPHANUM,k=16))
        exp_folder = folder+exp_id+"/"
        os.makedirs(exp_folder)
        df_pi.to_csv(exp_folder+'pi.tsv',sep='\t')
        df_targets.to_csv(exp_folder+'targets.tsv',sep='\t',index=False)
        with open(exp_folder+'eps_crit.pickle','wb') as epsfile:
            pickle.dump(eps_crit,epsfile)
        with open(exp_folder+'w_hat.pickle','wb') as wfile:
            pickle.dump(self.W_hat,wfile)
        with open(exp_folder+'inverse_mapping.pickle','wb') as mapfile:
            pickle.dump(self.inverse_mapping,mapfile)



    def optimize_biased_walk_normalized(self,eps,beta,OF=0,objective=OBJ_VAR_FROM_STABLE,name="test"):
        if not self.is_set_stable:
            self.set_stable(["unbiased",beta])
        self.norm = np.diag([1/el for el in self.stable])
        if not OF :
            print("Model definition")
        self.mo = pyo.ConcreteModel()
        
        #self.mo.Params.OutputFlag = OF
        #self.mo.Params.Method = 2
        #mo.Params.Crossover = 0
        #self.mo.Params.NumericFocus = 3
        if not OF:
            print("Variable definition")
        self.mo.nVars = pyo.Param(initialize=self.N)
        self.mo.N = pyo.RangeSet(0,self.mo.nVars - 1)
        self.mo.pi = pyo.Var(self.mo.N,domain=pyo.NonNegativeReals,bounds=lambda i : (0,self.sum_nodes))
        self.mo.m = pyo.RangeSet(0,self.m-1)
        self.mo.E = pyo.Var(self.mo.m,domain=pyo.NonNegativeReals,bounds=lambda i : (0,self.sum_nodes))
        #self.E = self.mo.addMVar(shape=self.m,vtype=GRB.CONTINUOUS,name='E',lb=0,ub=self.sum_nodes)
        
        self.ntarget = int(sum(self.tar != 0))
        self.mo.Xrange = pyo.RangeSet(0,self.ntarget - 1)
        self.mo.X = pyo.Var(self.mo.Xrange,domain=pyo.NonNegativeReals)
        #self.X = self.mo.addMVar(shape = self.ntarget,vtype=GRB.CONTINUOUS,name='X',lb = 0)
        if not OF:
            print("Objective definition")
        #c = D_inv @ tar
        self.set_objective(objective)
        #self.mo.C = pyo.Param(self.mo.N,initialize=lambda m,i:-self.tar[i]*self.norm[i,i])
        #self.mo.obj = pyo.Objective(rule=lambda m : pyo.summation(self.mo.C,self.mo.pi))
        if not OF:
            print("Constr4 definition")
        indices_W = [list(self.subGdirected_with_annot.predecessors(i)) for i in range(self.N)]
        #indices_inM = np.split(self.inM.indices,self.inM.indptr)[1:-1]
        self.e_ld = {i:[] for i in range(self.N)}
        for j,edge in enumerate(self.e_l):
            self.e_ld[edge[1]].append(j)
        
        self.mo.indices_W_dict = {i : pyo.Set(within=self.mo.N,initialize=indices_W[i]) for i in range(self.N)}
        for set_d in self.mo.indices_W_dict.values():
            set_d.construct()
        #self.mo.W_sparse = pyo.Param(self.mo.indices_W,initialize=lambda m,i,j:self.W_sparse[i,j])
        
        self.mo.indices_inM_dict = {i : pyo.Set(within=self.mo.m,initialize=self.e_ld[i]) for i in range(self.N)}
        #self.mo.inM = pyo.Param(self.mo.indices_inM,initialize=lambda m,i,j:self.inM[i,j])
        for set_d in self.mo.indices_inM_dict.values():
            set_d.construct()
        if not OF :
            print("sets defined")

        def rule_constr_4(model,i):
             return sum(model.pi[j]*self.W_sparse[i,j] for j in model.indices_W_dict[i] )*(1-eps)*(1-beta) + \
                 sum(model.E[j]*self.inM[i,j] for j in model.indices_inM_dict[i])*eps*(1-beta) + beta*self.pr[i] == model.pi[i]
        def rule_constr_4_bis(model,i):
             return pyo.sum_product(model.pi,{j : self.W_sparse[i,j] for j in indices_W[i]},index=indices_W[i])*(1-eps)*(1-beta) + \
                 pyo.sum_product(model.E,{j:self.inM[i,j] for j in indices_inM[i]},index=indices_inM[i])*eps*(1-beta) + beta*self.pr[i] == model.pi[i]

        self.mo.c4 = pyo.Constraint(self.mo.N,expr = rule_constr_4)
        #self.mo.addConstr(self.pi - beta*self.pr - (1-eps)*(1-beta)*self.W_sparse@self.pi - eps*(1-beta)*self.inM@self.E == 0, name='c4')

        if not OF :                       
            print("Constr5 definition")
        #indices_outM = np.split(self.outM.indices,self.outM.indptr)[1:-1]
        self.e_ld = {i:[] for i in range(self.N)}
        for j,edge in enumerate(self.e_l):
            self.e_ld[edge[0]].append(j)
        self.mo.indices_outM_dict = {i: pyo.Set(within=self.mo.m,initialize=self.e_ld[i]) for i in range(self.N)}
        for set_d in self.mo.indices_outM_dict.values():
            set_d.construct()
        if not OF :
            print("sets defined")
        #self.mo.outM = pyo.Param(self.mo.indices_outM,initialize=lambda m,i,j:self.outM[i,j])
        def rule_constr_5(model,i):
            return sum(model.E[j]*self.outM[i,j] for j in model.indices_outM_dict[i]) == model.pi[i]
        def rule_constr_5_bis(model,i):
            return pyo.sum_product(model.E,{j:self.outM[i,j] for j in indices_outM[i]},index=indices_outM[i]) == model.pi[i]
        
        self.mo.c5 = pyo.Constraint(self.mo.N,rule=rule_constr_5)
        #self.mo.addConstr(self.outM@self.E - self.pi == 0, name='c5')
        if not OF :
            print("Constr piisdist definition")
        self.mo.piisdist = pyo.Constraint(rule=lambda m: pyo.summation(m.pi) == self.sum_nodes)
        #self.mo.addConstr(np.ones(self.N)@self.pi == self.sum_nodes,name='c6')
        if not OF :
            print("Writing Model")
        self.mo.write(filename = name + ".mps", io_options = {"symbolic_solver_labels":True})
        h = hp.Highs()
        h.setOptionValue("log_file",name+".log")
        h.setOptionValue("log_to_console",False)
        h.setOptionValue("output_flag",False)
        h.readModel(name + ".mps")

        h.setOptionValue("solver","ipm")
        #self.h.setOptionValue("ipm_optimality_tolerance",0.1)
        h.setOptionValue("threads",4)
        #self.h.setOptionValue("run_crossover",False)
        h.run()
        h.writeSolution(name + ".sol",0)
        d = sparse.parse_solution(name+".sol")
        
        self.add_optimization_variables(d,eps,beta)
        self.optimized = True
        
        return self.norm,self.pi,self.E,self.X,h
