######################
## EXTERNAL IMPORTS ##
######################

import scipy as sp

import scipy.sparse.linalg as spl
import scipy.optimize as spo
import scipy.sparse

import pickle
import os

import pyomo.environ as pyo
import highspy as hp

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

###################
## LOCAL IMPORTS ##
###################

import obrwr.futils as fu
from obrwr.rwr_fun import unbiased_RWR, biased_RWR

##########################
## CONSTANTS DEFINITION ##
##########################

## OBJECTIVES
OBJ_VAR_FROM_STABLE = 'var_from_stable'
OBJ_ABSOLUTE = 'absolute'
OBJ_WEIGHTED_ABSOLUTE = 'weighted_absolute'
OBJ_VAR_FROM_VAL = 'var_from_value'
OBJ_QUADRATIC_FROM_STABLE = 'quadratic_from_stable'

ALPHANUM =  [chr(i) for i in range(ord('a'),ord('z')+1)] +\
            [chr(i) for i in range(ord('A'),ord('Z')+1)] +\
            [chr(i) for i in range(ord('1'),ord('9')+1)]


class MyGraph:
    ''' This is the class which defines an object which encapsulates all information relevant
    for both running the optimization and the plots.
    It allows easy usage of the method '''

    def __init__(self,G,self_loop=False):
        ''' Initializes all needed attributes for  the MyGraphs objects '''
        if not G.is_directed():  # Store the cc of G sorted by their size (n° of nodes)
            self.sorted_components = sorted(nx.connected_components(G),key=len,reverse=True)
        else:
            self.sorted_components = sorted(nx.weakly_connected_components(G),key=len,reverse=True)
        
        
        self.subG = G.subgraph(self.sorted_components[0]).copy()
        # self.subG is the subgraph of the biggest component of the graph

        if self_loop :
            for node,deg in list(self.subG.out_degree):
                if deg == 0 :
                    self.subG.add_edge(node,node)
        

        self.forgotten_proteins = {"Connectivity" : fu.get_list_from_components(self.sorted_components[1:])}
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
        # When False, original labels are being used

        self.map_elements_to_int()
        
        self.gen_instance()


    
    def gen_instance(self,reset_targets=True,reset_pr=True) :
        ''' 
        Creates all graph theoretical related variables
        Allows for refreshing the matrices when labels and mapping might have changed
        
        Args:
            * reset_targets when True self.tar is erased
            * reset_pr when True reset source information 
        '''
        
        self.e_l = list(nx.DiGraph(self.subGdirected_with_annot).edges)
        # self.e_l list of edges of the network
        self.N = len(self.subGdirected_with_annot.nodes)
        # self.N is the size of the network

        if reset_pr:
            self.pr = np.zeros(self.N)      # self.pr is the source weights
            self.pr[0] = 1
            self.sum_nodes = sum(self.pr)   # self.sum_nodes allows to normalize self.pr into distribution
            self.is_set_stable = False      # True if RWR was run with current self.pr
        
        if reset_targets :
            self.tar = np.zeros(self.N)     # Array, self.tar[i] != 0 implies i is a target
        
        # self.A is the ADJACENCY matrix of
        # the graph subGdirected_with_annot
        self.A_sparse = sp.sparse.csr_matrix(nx.linalg.graphmatrix.adjacency_matrix(self.subGdirected_with_annot,
                                                                                    nodelist=range(self.N))).T
        self.A = self.A_sparse.todense()

        # D is the diagonal matrix of degrees
        tmp = np.sum(np.array(self.A), axis=0)
        self.D = np.asmatrix(np.diag([int(el) for el in tmp]))
        self.D_inv = np.asmatrix(np.diag([1/int(el) for el in tmp]))
        self.D_sparse = sp.sparse.csr_matrix(self.D)
        self.D_inv_sparse = spl.inv(self.D_sparse)

        # W is the walk matrix in the case of uniform RWR
        self.W_sparse = self.A_sparse*self.D_inv_sparse
        self.W = self.W_sparse.todense()
        
        # Identitiy matrix in sparse format
        self.I = sp.sparse.eye(self.N,format='csc')
        
        # self.m is the number of edges in the (directed) graph
        self.m = len(self.e_l)

        # self.outM is a Nxm matrix on the kth row there is
        # a 1 at line e_l[k][0] (source of kth edge)

        # self.inM is a Nxm matrix on the kth row there is
        # a 1 at line e_l[k][1] (target of kth edge)
        self.outM = sp.sparse.csc_matrix((np.ones(self.m),([el[0] for el in self.e_l],list(range(self.m)))),
                                         shape=(self.N,self.m))
        self.inM = sp.sparse.csc_matrix((np.ones(self.m),([el[1] for el in self.e_l],list(range(self.m)))),
                                        shape=(self.N,self.m))
        
        #Boolean to know wether optimization has taken place or not
        self.optimized = False

        self.dinv_array = np.array([self.D_inv[i[0],i[0]] for i in self.e_l])
        self.Dinv_forpi = sp.sparse.csc_matrix((self.dinv_array,
                                                (list(range(self.m)),[el[0] for el in self.e_l])))


    def get_targets(self):
        '''
        Returns the list of protein identifiers (int or string) for the targets in the experiment.
        '''
        attr = nx.get_node_attributes(self.subGdirected_with_annot,"Targets")
        return [el for el,v in attr.items() if v]

    def get_sources(self):
        '''
        Returns the list of protein identifiers (int or string) for the source in the experiment.
        '''
        attr = nx.get_node_attributes(self.subGdirected_with_annot,"Sources")
        return [el for el,v in attr.items() if v]
        
    def map_elements_to_int(self):
        ''' 
        Maps node labels to int based on self.mapping
        ''' 
        if not self.is_int:
            nx.relabel_nodes(self.subG,self.mapping,copy=False)
            nx.relabel_nodes(self.subGdirected_with_annot,self.mapping,copy=False)
            self.e_l = [(self.mapping[e[0]],self.mapping[e[1]]) for e in self.e_l]
            self.is_int = True

    def map_elements_to_names(self):
        '''
        Maps back node labels from int to what they were originally
        '''
        if self.is_int:
            nx.relabel_nodes(self.subG,self.inverse_mapping,copy=False)
            nx.relabel_nodes(self.subGdirected_with_annot,self.inverse_mapping,copy=False)
            self.e_l = [(self.inverse_mapping[e[0]],self.inverse_mapping[e[1]]) for e in self.e_l]
            self.is_int = False
    
    def set_stable(self,method):
        '''
        Set the reference stable, as of now only the unbiased random walk works
        
        Args:
            * method array : method[0] is the name of stable method, method[1:] are arguments.
        '''
        if method[0] == 'unbiased': # Unbiaed Random Walk with restart
            self.stable = unbiased_RWR(self,method[1],(1e-16) * self.sum_nodes ) # Compute nodes score for unbiased
            tmp_directed = self.subG.to_directed()           # Used to retrieve node degrees
            nx.set_edge_attributes(self.subGdirected_with_annot,  
                                   {edge:1/tmp_directed.out_degree[edge[0]] for edge in self.subGdirected_with_annot.edges},
                                "WeightsStable")             # Edge annotation
            nx.set_node_attributes(self.subGdirected_with_annot,
                                   {key:val for key,val in enumerate(self.stable)},
                                "Stable")                    # Node annotation with unbiased score
            self.is_set_stable = True
        
    def check_protein_list(self,protein_list,category,is_mapped=False):
        '''
        Checks a protein list for presence in the network.
        Stores the absent, annotated with a category.
        Returns proteins wwhich are present.
        
        Args:
            * protein_list : list of protein identifiers (either int is is_mapped == True, or string)
            * category : string , annotation for the missing proteins.
        
        Return:
            returns list of proteins ids, those present in the network
        '''
        tmp = []
        self.forgotten_proteins[category] = []
        if not is_mapped : # If ids in graph are string
            for el in protein_list:
                if el in self.mapping.keys():
                    tmp.append(self.mapping[el])
                else:
                    self.forgotten_proteins[category].append(el)
        else: #If ids in graph are int
            for el in protein_list:
                if el in self.inverse_mapping.keys():
                    tmp.append(el)
                else:
                    self.forgotten_proteins[category].append(el)
        return tmp

    def remove_unreachable_nodes(self,sources_list,is_mapped=False):
        '''
        Remove nodes in the network which can't be reached from sources.

        Args:
            * sources_list (list of strings/int) : list of proteins sources
            * is_mapped (bool) : False if string ids, otherwise int (internal representation).
        '''
        reachable = set(sources_list)
        for el in sources_list:
            reachable = reachable | nx.descendants(self.subGdirected_with_annot,el)
            # Computing union of reachable nodes
        removed = set(self.subGdirected_with_annot.nodes) - reachable
        # Unreachable nodes are those not in reachable
        self.forgotten_proteins["Unreachable from sources"] = [self.inverse_mapping[iprot] for iprot in removed]
        # Annot removed prots
        self.subGdirected_with_annot.remove_nodes_from(removed)
        self.subG.remove_nodes_from(removed)

    def remap(self):
        '''
        Recomputes a mapping, to run only if nodes are mapped.
        That is if the labels in subGdirected_with_annot are ints (internal representation)
        '''
        tmp_mapping = {el:i for i,el in enumerate(self.subGdirected_with_annot.nodes)} 
        self.subGdirected_with_annot = nx.relabel_nodes(self.subGdirected_with_annot,tmp_mapping,copy=True)
        self.subG = nx.relabel_nodes(self.subG,tmp_mapping,copy=True)
        self.mapping = {self.inverse_mapping[el]:i for el,i in tmp_mapping.items()}
        self.inverse_mapping = { i:prot for prot,i in self.mapping.items()}

    def set_sources(self,sources_list,sum_nodes,is_mapped=False):
        '''
        Set sources from list of proteins.
        Also sets the scaling factor for computation. Scaling factor is used in order to have not too small
        scores. Since the scores are a probability distribution, we can scale them by a factor without losing 
        information. Typically sum_nodes ~ 1e4, has to increases with network size in order to avoid floating
        point arithmetics rounding to zero errors.

        Args:
            * sources_list (list of strings/int) : list of proteins to set as sources
            * sum_nodes (float) : scaling factor
            * is_mapped (bool) : Wether ids are string or int.
        '''
        # Filtering sources 
        sources_list = self.check_protein_list(sources_list,"sources not in",is_mapped)
        self.remove_unreachable_nodes(sources_list,is_mapped=True) 
        sources_list = [self.inverse_mapping[el] for el in sources_list]
        # Remapping after removal of some nodes
        self.remap()
        # Regenerating attributes
        self.gen_instance()
        sources_list = [self.mapping[el] for el in sources_list]
        nsources = len(sources_list)
        # Setting the restart probability vector
        self.pr = np.zeros(self.N)
        for i in sources_list:
            self.pr[i] = sum_nodes/nsources
        self.sum_nodes = sum_nodes
        # Annotating sources
        tmp =  {i:(i in sources_list) for i in self.subGdirected_with_annot.nodes}
        nx.set_node_attributes(self.subGdirected_with_annot,  
                              tmp,"Sources")                                                                                        
        return sources_list

    def set_targets(self,dict_of_values,is_mapped=False,self_loops=False,OF=False):
        '''
        Sets the targets and their objective values.
        
        Args:
            * dict_of_values : dictionnary, keys are ids and values are floats
            * is_mapped (bool) : is True labels of Graph are int otherwise they are string ids
            * self_loop (bool): Wether to add a self loop to each target
            * OF (bool) : Wether to be verbose (if True) or not. 
        '''
        # Checking if target proteins are present
        targets = self.check_protein_list(list(dict_of_values.keys()),"targets not in",is_mapped)
        # Updating keys of dict
        dict_of_values = {target:dict_of_values[self.inverse_mapping[target]] for target in targets}
        # Annotating in the Graph
        nx.set_node_attributes(self.subGdirected_with_annot,
                               {i:(i in dict_of_values.keys()) for i in self.subGdirected_with_annot.nodes},
                               "Targets")
        # Defining an numpy array for target values
        self.tar = np.zeros(self.N)
        # Will be used during optimization
        for k,v in dict_of_values.items():                            
            self.tar[k] = v
            if OF:
                print(self.inverse_mapping[k],v)
            self.subGdirected_with_annot.add_edge(k,k)
            self.subG.add_edge(k,k)
        # Regenerate attributes because graphs might have changed
        self.gen_instance(reset_targets=False,reset_pr=False)         


    def add_optimization_variables(self,d,eps,beta):
        '''
        Create attributes for optimization results, after optimization.
        To be called by the optimization procedure.

        Args:
            * d (dict) : Result of parsing the result file from HiGHS
            * eps (float) : Bias strength
            * beta (float) : Restart probability
        '''
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
        # self.pi is the optimized vector score
        self.pi = piX
        # self.E is the optimized vector of edge bias (transformed for linearization)
        self.E = EX
        # self.X is the optimized vector of variables introduced for absolute values
        self.X = XX
        # We compute the transformation from E to B (delinearization).
        self.b = EX/np.array([piX[i[0]] for i in self.e_l])
        count = {i:0 for i in range(self.N)}
        for z,el in enumerate(self.e_l):
            count[el[0]] += self.b[z]
        self.b = [(self.b[i]/count[el[0]] if self.b[i] else 0) for i,el in enumerate(self.e_l)]
        self.PR = sp.sparse.csc_matrix(np.tensordot(self.pr,[1 for _ in range(self.N)],axes=0))
        self.B = sp.sparse.csc_matrix((self.b,([e[1] for e in self.e_l],[e[0] for e in self.e_l])))
        self.W_hat = (1-eps)*self.W_sparse + eps*self.B

    def add_meta_after_optim(self,eps,beta,name):
        '''
        Annotating the graph self.subGdirected_with_annot.
        Storing the attributes on nodes and edges (scores, probas, weights ...)

        Args :
            * eps (float)   : Bias Strength
            * beta (float)  : Restart probability
            * name (string) : The name of the experiment 
        '''
        edge_weights = (1-eps)*self.W_sparse + eps*self.B

        # Storing the Pi variable (OBRWR score) 
        tmp = {key:val for key,val in enumerate(self.pi)}
        nx.set_node_attributes(self.subGdirected_with_annot,tmp,"Pi"+name)

        # Storing log ratio between RWR and OBRWR scores
        tmp = {key:np.log2(val/self.stable[key]) for key,val in enumerate(self.pi)}
        nx.set_node_attributes(self.subGdirected_with_annot,tmp,"Proba_logratio"+name)

        # Storing the weights of the edges, they are the transition probability 
        tmp = {(i,j):edge_weights[j,i] for (i,j) in self.subGdirected_with_annot.edges}
        nx.set_edge_attributes(self.subGdirected_with_annot,tmp,"weights"+name)

        # Storing the Bias 
        tmp = {(i,j):self.B[j,i] for (i,j) in self.subGdirected_with_annot.edges}
        nx.set_edge_attributes(self.subGdirected_with_annot,tmp,"B"+name)

        # Edge probability defined as the product of node proba and edge weight.
        # Here using OBRWR probabilities
        tmp = {(i,j):edge_weights[j,i]*self.pi[i]/self.sum_nodes for (i,j) in self.subGdirected_with_annot.edges}
        nx.set_edge_attributes(self.subGdirected_with_annot,tmp,"Proba_edge_optim"+name)

        # Here using RWR probabilities
        tmp = {(i,j):self.W_sparse[j,i]*self.stable[i]/self.sum_nodes for (i,j) in self.subGdirected_with_annot.edges}
        nx.set_edge_attributes(self.subGdirected_with_annot,tmp,"Proba_edge_stable"+name)

        # Log ratio between the two edge probabilities
        tmp = {(i,j):np.log2(edge_weights[j,i]*self.pi[i]/(self.W_sparse[j,i]*self.stable[i])) for (i,j) in self.subGdirected_with_annot.edges}
        nx.set_edge_attributes(self.subGdirected_with_annot,tmp,"Proba_edge_logratio")

    def control_optimization(self,beta):
        '''
        Plotting the correspondence between Pi from HiGHS and Pi recomputed 
        by iterative simulation of OBRWR from the transition matrix obtained from HiGHS.
        It ensures the correspondence between Pi and W after optim (provides a minimum of control)

        It actually plots the log values of the scores.
        x-axis is the recomputed score
        y-axis is the HiGHS computed score

        We expect all dots to be on the main diagonal.

        Args:
            * beta (float) : Restart Probability
        '''
        if self.optimized:
            tmp = biased_RWR(self,self.eps_crit,beta,1e-10)
            ceil = 1e-12
            m = min(min(np.log(tmp)),min(np.log(self.pi))) - 2
            M = max(max(np.log(tmp)),max(np.log(self.pi))) + 2
            plt.figure()
            sns.lineplot(x=[m,M],y=[m,M],ls='--',color='r',alpha=0.9)
            sns.scatterplot(x=np.log(ceil+tmp),y=np.log(ceil+self.pi),alpha=.5, s=60)
            ax = plt.gca()
            ax.set_ylim(m,M)
            ax.set_xlim(m,M)
            plt.ylabel("Optimization Probabilities")
            plt.xlabel("Recomputed Probabilities from edge weights")
            plt.title("Control for correction of computed probabilities")

    def get_directed_subgraph(self):
        '''
        Select the edges which have a bias > 0.
        Optimization has to have been run.

        Returns:
            dig nx.DiGraph the directed Graph of the biased edges.
        '''
        if self.optimized:
            dig = self.subGdirected_with_annot.copy()
            edges_to_keep = [(self.e_l[i][0],self.e_l[i][1]) for i in range(len(self.E)) if self.b[i] > 0]
            dig.remove_edges_from([el for el in dig.edges if el not in edges_to_keep])
        else:
            raise Exeption("Optimization not run")
        return dig

    def get_directed_subgraph_from_sources(self):
        '''
        Constructs the network of edges with bias > 0.
        Inside this netowrk, select the nodes reachable from the sources.

        Returns:
            A networkx DiGraph
        '''
        dig = self.get_directed_subgraph()
        tmp = set()
        is_source = nx.get_node_attributes(self.subGdirected_with_annot, "Sources")
        for node in self.subGdirected_with_annot.nodes:
            if is_source[node]:
                print(nx.descendants(dig,node))
                tmp = tmp | nx.descendants(dig,node) | set([node])
        return dig.subgraph(tmp)

    def get_higher_edge_proba_subgraph(self,thr=1):
        '''
        Selects the edges based on the logratio of edge probabilities (OBRWR vs RWR),
        an edge is selected when the logratio is greater than threshold.
        
        Args:
            * thr (float) : Threshold for edge selection

        Returns:
            A networkx DiGraph
        '''
        dig = self.subGdirected_with_annot.copy()
        edges_to_remove = [(s,e) for s,e,v in dig.edges(data=True) if (v["Proba_edge_logratio"] < thr )]
        dig.remove_edges_from(edges_to_remove)
        return dig

    def plot_objective_eps(self,beta,OF=False,objective=OBJ_VAR_FROM_STABLE,n=20):
        '''
        Plot the objective value at optimum for a range of epsilon value between 0 and 1-1/n.
        That is for all i/n for i in 0, ..., n-1.

        x-axis is the bias strength [0,1[
        y-axis is the objective optimum for given eps.

        Args:
            * beta (float) : Restart probability
            * n (int) : How fine is the segmentation of the [0,1[ interval.
        '''
        Eps = np.arange(0,n)/n
        plt.figure()
        obj = [self.optimize_biased_walk_normalized(e,beta,OF=OF,objective=objective)[4].getObjectiveValue() for e in Eps]
        sns.lineplot(x=Eps,y=obj)
        plt.ylabel("Objective optimum")
        plt.xlabel("Bias strength (epsilon)")
        plt.title("Bias strength controls objective optimum")


    def find_eps_critic(self,beta,sensitivity,OF=False,objective=OBJ_VAR_FROM_STABLE,name="test"):
        '''
        This function finds the smallest value of epsilon for which the objective optimum is 0.
        It is a dichotomic search since the objective optimum is a decreasing function
        of epsilon (easy to show that the search space only gets bigger when epsilon augments).

        Args:
            * beta (float) : Restart probability
            * sensitivity (float) : Close to 0, how much precision on eps_crit we which to have.
            * OF (bool) : Verbosity flag

        Returns (eps_c, pi) :
            * eps_c (float) :The value of critical epsilon.
            * pi (np.array float) : The optimized scores (Pi) for critical epsilon.
        '''
        if OF:
            print('Looking for eps_crit')
        a,b = sensitivity,1-sensitivity
        d = self.optimize_biased_walk_normalized(b,beta,OF=OF,objective=objective,name=name)[4].getObjectiveValue()
        while b-a > sensitivity: # Dichotomy
            if OF :
                print(f'#####\nLow bound : {a} \nHigh bound : {b} \n#####')
            _,pi,_,_,model = self.optimize_biased_walk_normalized((a+b)/2,beta,objective=objective,OF=OF,name=name)
            m = model.getObjectiveValue()
            if abs(m - d) < 1e-5 :
                b = (a+b)/2
            else:
                a = (a+b)/2
        return (a+b)/2,pi

    def set_objective_OBJ_VAR_FROM_STABLE(self):
        '''
        Sets the objective as |pi/si - v|.
        This is done using the X dummy variables since absolute values are not linear.
        This is a common linearization trick for absolute values in LP programs.
        '''
        def rule_abs_1(model,xi,d):
            i,val = d[xi]
            return model.X[xi] >= model.pi[i]*self.norm[i][i] - val
        def rule_abs_2(model,xi,d):
            i,val = d[xi]
            return model.X[xi] >= val - model.pi[i]*self.norm[i][i]

        c = self.tar
        self.mo.obj = pyo.Objective(rule = lambda m: pyo.summation(m.X))
        xi = 0

        d = {}
        for i,val in enumerate(self.tar): 
            if val != 0:
                d[xi] = (i,val)
                xi+=1

        self.mo.abs1 = pyo.Constraint(self.mo.Xrange,rule=lambda m,x:rule_abs_1(m,x,d))
        self.mo.abs2 = pyo.Constraint(self.mo.Xrange,rule=lambda m,x:rule_abs_2(m,x,d))

    def set_objective_OBJ_VAR_FROM_VAL(self):
        '''
        Sets the objective as |pi-v/v|.
        This is done using the X dummy variables since absolute values are not linear.
        This is a common linearization trick for absolute values in LP programs.
        '''
        def rule_abs_1(model,xi,d):
            i,val = d[xi]
            return model.X[xi] >= (model.pi[i]-val)/val
        def rule_abs_2(model,xi,d):
            i,val = d[xi]
            return model.X[xi] >= (val - model.pi[i])/val

        c = self.tar
        self.mo.obj = pyo.Objective(rule = lambda m: pyo.summation(m.X))
        xi = 0

        d = {}
        for i,val in enumerate(self.tar): 
            if val != 0:
                d[xi] = (i,val)
                xi+=1

        self.mo.abs1 = pyo.Constraint(self.mo.Xrange,rule=lambda m,x:rule_abs_1(m,x,d))
        self.mo.abs2 = pyo.Constraint(self.mo.Xrange,rule=lambda m,x:rule_abs_2(m,x,d))


    def set_objective_OBJ_ABSOLUTE(self):
        c = np.zeros(len(self.tar))
        for i,el in enumerate(self.tar):
            if el > 0: 
                c[i] = 1
            elif el < 0 :
                c[i] = -1
        self.mo.C = pyo.Param(self.mo.N,initialize=lambda m,i:-c[i]*self.norm[i,i])
        self.mo.obj = pyo.Objective(rule=lambda m : pyo.summation(self.mo.C,self.mo.pi))
        
    def set_objective_OBJ_WEIGHTED_ABOSLUTE(self):
        self.mo.C = pyo.Param(self.mo.N,initialize=lambda m,i:-self.tar[i]*self.norm[i,i])
        self.mo.obj = pyo.Objective(rule=lambda m : pyo.summation(self.mo.C,self.mo.pi))
        
    def set_objective(self,objective):
        '''
        This method is a switch to choose between the different objective functions.
        If v is the target value, si the stable score and pi our optimization score.
        (We describe the objectivefor a single target, the sum over the targets is 
        actually optimized)  
        Three possibilities :
            OBJ_VAR_FROM_STABLE:
                 Here we minimize |pi/si - v| (matching the experimental variation)
            OBJ_ABSOLUTE:
                 Here we maximize pi/si  when v > 0 and minimize pi/si otherwise. (all nodes contribute the same)
            OBJ_WEIGHTED_ABSOLUTE:
                 Here we maximize v*pi/si (nodes with higher v contribute more to the objective)
        '''
        if objective == OBJ_VAR_FROM_STABLE:
            self.set_objective_OBJ_VAR_FROM_STABLE() 
        elif objective == OBJ_VAR_FROM_VAL:
            self.set_objective_OBJ_VAR_FROM_VAL()
        elif objective == OBJ_ABSOLUTE:
            self.set_objective_OBJ_ABSOLUTE()
        elif objective == OBJ_WEIGHTED_ABSOLUTE:
            self.set_objective_OBJ_WEIGHTED_ABOSLUTE()
            
    def plot_degrees(self,name):
        '''
        Plots some infography of the degree distribution of the nodes in the network.
        It highlights how the targets are distributed.

        Two subplots : 
            First subplot of degrees:
               x-axis Degree rank
               y-axis Degree
            Second subplot is  a histogram of degrees.

        Args:

           * name (string): The experiment name for titles in the plots
        '''
        degree_sequence_with_nodes = sorted(((n,d) for n, d in self.subGdirected_with_annot.degree()),key = lambda x : x[1], reverse=True)
        nodes_sequence_degree_sorted = [el[0] for el in degree_sequence_with_nodes]
        degree_sequence = [el[1] for el in degree_sequence_with_nodes]
        
        targets = nx.get_node_attributes(self.subGdirected_with_annot,'Targets')
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
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()


    def define_bins(self,deg_seq_sorted):
        '''
        Bins the nodes in same size bins.
        Here the choice is to take the first bin as all smallest degree proteins.
        All the other bins will be built by finding the next degree such that the defined bin 
        is just bigger than the first bin.

        Args : 
            * deg_seq_sorted (int list) : Sorted list of degrees
        
        Returns :
            * bins ((int,int) list) : Defines the boundaries of the bin [a,b]
            * bin_counts (int list) : Number of nodes in each bin
        '''
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
        '''
        Samples N sets of proteins which have the same degree distribution (in terms of bins)
        as the targets.
        These sets will be used as random targets to randomize the experiment and produce significance statistics.

        Args:
            * N (int) : Number of sample wanted.

        Returns:
            * random_targets ((alpha list) list) : Where alpha depends on the mapping status of the experiment,
                                                   either string or int.
        '''
        targets = nx.get_node_attributes(self.subGdirected_with_annot,'Targets')
        degrees = self.subGdirected_with_annot.degree()
        degree_sequence_sorted = sorted([d for n, d in degrees])
        bins,bin_counts = self.define_bins(degree_sequence_sorted)
        bins_nodes = [[] for _ in bins]
        target_counts = [0 for _ in bins]
        random_targets = [[] for _ in range(N)]

        for n,d in degrees: # Separate nodes into bins
            for i, (m,M) in enumerate(bins):
                if d <= M:
                    if targets[n]:
                        target_counts[i] += 1
                    else :
                        bins_nodes[i].append(n)
                    break
                
        for i in range(N): # Sampling the bins to create random sets
            rng = np.random.default_rng()
            for nodes,c in zip(bins_nodes,target_counts):
                random_targets[i] += list(rng.choice(nodes,c))
        return random_targets

    def plot_sampling_bins(self,experiment):
        '''
        Plots the bin sizes, their range as well as how many targets are in each bins.

        Args:
            * experiment (string) : Name of the experiment to personalize the plots
        '''
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


    def get_significative_nodes(self,beta,N,experiment="Test",folder='Test',prec=1e-1):
        '''
        Runs a number of randomization experiments.
        Stores the computation results in a folder for future statistical analysis.
        Also computes pvalues on a per node basis (without multiple test correction), 
        stores the information in the networkx DiGraph object as a node attribute .

        Args:
            * beta (float) : Restart probability
            * N (int) : Number of random samples
            * experiment (string) : Name of the experiment (for annotation purposes after pval computation)
            * folder (string) : Path to a folder to store the random runs

        Returns:
            * df (DataFrame) : 
            * df_tar (DataFrame)
            * df_eps (DataFrame)
            * df_experiment (DataFrame)
        '''
        self.map_elements_to_int()
        self.eps_crit,pi = self.find_eps_critic(beta,sensitivity= prec,OF=1,name=experiment) # Computing OBRWR on target from experiment
        df_experiment = pd.DataFrame({experiment:{self.inverse_mapping[j]:pi[j] for j in range(len(pi))}}) # Storing in a Dataframe
        self.add_meta_after_optim(self.eps_crit,beta,experiment)
        df,df_tar,df_eps = self.get_random_runs(beta,N,folder,sensitivity=prec,name=experiment) # Running the randomizations 
        def compute_pval_oneside(row):
            s = 0
            for i in range(N):
                if df_experiment[experiment][row.name] < row[i]:
                    s += 1
            return (s+1)/(N+1)
        # Computing/storing a per node pval based on score (Pi))
        df_experiment = df_experiment.join(pd.DataFrame(df.apply(compute_pval_oneside,axis=1),columns=["pvals"]))
        # Storing those pvalues in the networkx Graph
        nx.set_node_attributes(self.subGdirected_with_annot,\
                               {key:df_experiment["pvals"][self.inverse_mapping[key]] for key,val in enumerate(pi)},
                               "pvals"+experiment)
        return df,df_tar,df_eps,df_experiment
        
    def get_random_runs(self,beta,N,folder,sensitivity=1e-1,name="name"):
        '''
        This function generates the random targets.
        For each sample of random targets it will run OBRWR and save the output in a folder.
        A random folder name is generated to save the run.
        
        Args:
            * beta (float) : Restart probability
            * N (int) : Number of random runs to run
            * folder (string) : base folder in which the runs will be stored

        Returns:
            * df (Dataframe) : Score table (columns are runs, rows are proteins)
            * df_target_values : One column is "Value" (original target values), others are 
                                 corresponding targets for each random runs.
            * pd.DataFrame(epsilons) : Table which stores the epsilon for each random run.
        '''
        
        # Initializing the different dataframes and generating the random targets
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
        for i in range(N): #Running OBRWR for each random sample
            b = True
            while(b): # If an exception is raised during computation, we generate new targets.
                try:
                    tmp = cp.deepcopy(self)
                    tmp.set_targets({tmp.inverse_mapping[k]:v for k,v in zip(sample_list[i],tar_vals)},OF=1)
                    eps_crit,pi = tmp.find_eps_critic(beta,sensitivity,OF=False,name=name)
                    b = False
                except:
                    l = self.sample_random_targets(1)[0]
                    sample_list[i] = l
                    df_target_values[i] = pd.DataFrame(list(map(lambda x : self.inverse_mapping[x],l)))
                    b = True
            epsilons.append(eps_crit)
            df_dictionary = pd.DataFrame({i:{tmp.inverse_mapping[j]:pi[j] for j in range(len(pi))}})
            tmp.save_run_in(folder,df_dictionary,eps_crit,df_target_values[["Value",i]]) # Saving run results
            df = df.join(df_dictionary)
            if i%5 == 4: # This part of the function generates verbose output to follow ongoing computation
                T1 = time.time()
                times.append(T1 - T0)
                T0 = time.time()
                print(str(i) + ' / ' + str(N))
                print(df.head())
                print(f'Time for last 5 runs : {times[-1]/60} min.')
                print(f'Average time per 5 runs : {np.mean(times)/60} min.')
        
        return df,df_target_values,pd.DataFrame(epsilons)

    def save_run_in(self,folder,df_pi,eps_crit,df_targets):
        '''
        This function saves the different DataFrames and objects in a folder
        which name is randomly generated.

        DataFrames are stored as .tsv, other objects (eps_critnw_hat,inverse_mappign)
        are stored as pickle objects.
        
        Args:
            * folder (string) : path to base folder
            * df_pi (DataFrame) : rows are proteins, only one columns : the Pi for this run
            * eps_crit (float) : the computed bias strength
            * df_targets (DataFrame) : targets for this run

        '''
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

    def set_constraint4(self,eps,beta,OF):
        '''
        This function builds Constraint4.
        Only useful to separate code into readable chunks

        Args:
            * OF (bool) : Verbose if True
        '''
        def rule_constr_4(model,i):
            return sum(model.pi[j]*self.W_sparse[i,j] for j in model.indices_W_dict[i] )*(1-eps)*(1-beta) + \
                   sum(model.E[j]*self.inM[i,j] for j in model.indices_inM_dict[i])*eps*(1-beta) + beta*self.pr[i] == model.pi[i]

        indices_W = [list(self.subGdirected_with_annot.predecessors(i)) for i in range(self.N)]
        self.e_ld = {i:[] for i in range(self.N)}
        for j,edge in enumerate(self.e_l):
            self.e_ld[edge[1]].append(j)
        
        self.mo.indices_W_dict = {i : pyo.Set(within=self.mo.N,initialize=indices_W[i]) for i in range(self.N)}
        for set_d in self.mo.indices_W_dict.values():
            set_d.construct()
        
        self.mo.indices_inM_dict = {i : pyo.Set(within=self.mo.m,initialize=self.e_ld[i]) for i in range(self.N)}

        for set_d in self.mo.indices_inM_dict.values():
            set_d.construct()
            
        if  OF :
            print("sets defined")
        self.mo.c4 = pyo.Constraint(self.mo.N,expr = rule_constr_4)


    def set_constraint5(self,OF):
        '''
        This function builds Constraint5.
        Only useful to separate code into readable chunks

        Args:
            * OF (bool) : Verbose if True
        '''
        def rule_constr_5(model,i):
            return sum(model.E[j]*self.outM[i,j] for j in model.indices_outM_dict[i]) == model.pi[i]
        
        self.e_ld = {i:[] for i in range(self.N)}

        for j,edge in enumerate(self.e_l):
            self.e_ld[edge[0]].append(j)
        self.mo.indices_outM_dict = {i: pyo.Set(within=self.mo.m,initialize=self.e_ld[i]) for i in range(self.N)}

        for set_d in self.mo.indices_outM_dict.values():
            set_d.construct()

        if OF :
            print("sets defined")
        
        self.mo.c5 = pyo.Constraint(self.mo.N,rule=rule_constr_5)
        

    def optimize_biased_walk_normalized(self,eps,beta,OF=False,objective=OBJ_VAR_FROM_STABLE,name="test"):
        '''
        We use Pyomo as a modeller for LP program.
        And HiGHS as the actual optimizer.
        We use a custom function to read the result file after optimization.

        
        The function builds the different constraints and objective.
        It produces a .mps file, calls HiGHS through it's API.

        The function then proceeds to load the results and store as 
        attribute in the Graph.

        Args:
            * eps (float) : Strength bias to use
            * beta (float) : Restart probability to use 
            * OF (bool) : Verbose flag (verbose if True)
            * objective (int) : Can only take particular values (see constants at beginning of file)
            * name (string) : name of experiment when storing results

        Returns:
            * self.norm (numpy matrix) : Diagonal matrix with the inverse of RWR score 
            * self.pi   (numpy array)  : OBRWR score 
            * self.E    (numpy array)  : Edge weights for OBRWR
            * self.X    (numpy array)  : Dummy variable for the absolute value computation
            * h         (highs object) : highs optimizer object

        '''

        # Running RWR
        if not self.is_set_stable:
            self.set_stable(["unbiased",beta])
        self.norm = np.diag([1/el for el in self.stable])

        # Creating Pyomo model and necessary variables (inside the model)
        if OF :
            print("Model definition")
        self.mo = pyo.ConcreteModel()

        if OF:
            print("Variable definition")
        self.mo.nVars = pyo.Param(initialize=self.N) # Number of variables
        self.mo.N = pyo.RangeSet(0,self.mo.nVars - 1) # Set of node ids {0,...,N-1}
        self.mo.pi = pyo.Var(self.mo.N,domain=pyo.NonNegativeReals,bounds=lambda i : (0,self.sum_nodes))
        self.mo.m = pyo.RangeSet(0,self.m-1) # Set of edge ids {0,...,m-1}
        self.mo.E = pyo.Var(self.mo.m,domain=pyo.NonNegativeReals,bounds=lambda i : (0,self.sum_nodes)) 
        self.ntarget = int(sum(self.tar != 0)) # Number of targets
        self.mo.Xrange = pyo.RangeSet(0,self.ntarget - 1) # Set of identifiers for the dummy variable
        self.mo.X = pyo.Var(self.mo.Xrange,domain=pyo.NonNegativeReals)

        if OF:
            print("Objective definition")
        self.set_objective(objective)

        if OF:
            print("Constr4 definition")
        self.set_constraint4(eps,beta,OF)
        
        if OF :                       
            print("Constr5 definition")
        self.set_constraint5(OF)

        if OF :
            print("Constr piisdist definition")
        self.mo.piisdist = pyo.Constraint(rule=lambda m: pyo.summation(m.pi) == self.sum_nodes)

        if OF :
            print("Writing Model")
        # Writing the model as .mps file
        self.mo.write(filename = "./tmp/"+name + ".mps", io_options = {"symbolic_solver_labels":True})

        # Here we call highs to solve our LP
        h = hp.Highs()
        h.setOptionValue("log_file","./tmp/"+name+".log")
        h.setOptionValue("log_to_console", OF)
        h.setOptionValue("output_flag", OF)
        h.readModel("./tmp/"+name + ".mps")

        h.setOptionValue("solver","ipm")
        #h.setOptionValue("ipm_optimality_tolerance",(1e-14)*self.sum_nodes)
        h.setOptionValue("threads",0)
        h.setOptionValue("run_crossover","off")
        i = 0
        while i < 3 and h.getModelStatus() != hp.HighsModelStatus.kOptimal :
            h.run()
            i += 1
        h.writeSolution("./tmp/"+name + ".sol",0)

        # Here we read the solution
        d = fu.parse_solution("./tmp/"+name+".sol")

        # And store the attributes
        self.add_optimization_variables(d,eps,beta)
        self.optimized = True
        
        return self.norm,self.pi,self.E,self.X,h
