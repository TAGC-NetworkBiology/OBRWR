##
## Python script to run on the cluster.
## It will run OBRWR on a given file input file
##

# Imports
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from obrwr import obrwr_new as ob
import sys

# Constants
experiment= 'HPNObrwr'
file_name_PKN = "phonemesPKN.tsv"
file_name_PPI = "PPI/human_binary_network_gene.txt"
back_path = "/mnt/"
output_folder = back_path+"05_Output/"
input_folder = back_path+"00_InputData/"
folder_exp = back_path+'05_Output/'+experiment+"/"
folder_rd = folder_exp + 'random/'

sensitive_proteins_file = "_toptable.tsv"
list_of_activators = ['EGF', 'FGF1', 'HGF', 'IGF1', 'INS', 'NRG1','PBS','Serum']
cell_lines = ['BT20','BT549','MCF7','UACC812']
activator = list_of_activators[0]
cell_line = cell_lines[0]
N_rand = 1 

#Specific OBRWR
beta = 0.7

def read_phonemesPKN():
    df = pd.read_csv(output_folder+file_name_PKN,sep='\t')
    l = []
    for el in df['target']:
        if (('_' in el) and (not '__' in el)):
            l.append(el.split('_')[0])
        else:
            l.append(el)
    df['target'] = l
    l = []
    for el in df['source']:
        if (('_' in el) and (not '__' in el)):
            l.append(el.split('_')[0])
        else:
            l.append(el)
    df['source'] = l
    return df.drop('interaction',axis=1).drop_duplicates(ignore_index=True)

def read_PPI(df):
    nodes = set(df['source']).union(set(df['target'])) 
    df_PPI = pd.read_csv(input_folder + file_name_PPI,header=None,sep='\t')
    mask = [el in nodes for el in df_PPI[0]] or [el in nodes for el in df_PPI[1]]
    df_PPI = df_PPI[mask].drop_duplicates(ignore_index=True)
    df_PPI.columns = ['source','target']
    return df_PPI

def adding_phospho_from_hpn(file_name):
    df_psite = pd.read_csv(file_name,sep='\t')
    df_psite = df_psite.drop_duplicates()
    psites = [(el,el.split('_')[0]) for el in list(df_psite['ID'])]
    df_diff = pd.DataFrame(psites,columns=['source','target'])
    
    df_phon = pd.read_csv(output_folder+file_name_PKN,sep='\t')
    psites = list(df_psite['ID'])
    mask = [(el in psites) for el in df_phon['target']]
    df_phon = df_phon[mask].drop_duplicates(ignore_index=True)
    
    return df_diff,df_phon

def set_constants(file_name):
    global activator, cell_line, folder_exp, folder_rd
    s = file_name.split('/')[-1]
    l = s.split('_')
    activator = l[1]
    cell_line = l[0]
    folder_exp += cell_line+'/'+activator+'/'
    folder_rd = folder_exp + "random/"
    
def build_network(file_name):
    df = read_phonemesPKN()
    df_PPI = read_PPI(df)
    df_diff,df_phon = adding_phospho_from_hpn(file_name)
    
    G_phos = nx.from_pandas_edgelist(df,create_using=nx.DiGraph)
    G_hpn_psites = nx.DiGraph(nx.from_pandas_edgelist(df_diff))
    G_link = nx.from_pandas_edgelist(df_phon,create_using=nx.DiGraph)
    G_PPI = nx.DiGraph(nx.from_pandas_edgelist(df_PPI))
    
    G = nx.compose(G_phos,G_PPI)
    G = nx.compose(G,G_hpn_psites)
    G = nx.compose(G,G_link)
    
    myGraph = ob.MyGraph(G,self_loop=True)
    myGraph.set_sources(['EGF'],1e6)
    return myGraph,G

def get_targets(file_name,G):
    df_psite = pd.read_csv(file_name,sep='\t')
    df_psite = df_psite.drop_duplicates()
    df_psite = df_psite[df_psite['adj.P.Val'] < 5e-2]
    d = {}
    for rec in df_psite.to_dict('records'):
        if not rec['ID'] in d.keys():
            var = np.exp(np.abs(rec['logFC']))
            if not rec['ID'] in G.nodes():
                d[rec['ID'].split('_')[0]] = var
            else:
                d[rec['ID']] = var
    return d

def run_obrwr(myGraph):
    global experiment, cell_line, activator, beta, N_rand, folder_rd
    
    this_experiment = experiment+'_'+cell_line+'_'+ activator
    tmp = myGraph.get_significative_nodes(beta, N_rand, this_experiment, folder_rd)
    distribution_scores,targets_df,epsilons_df,df_exp = tmp
    distribution_scores.to_csv(folder_rd+"/distribution_signor_"+this_experiment+".tsv",sep='\t')
    targets_df.to_csv(folder_rd+"/random_targets_signor_"+this_experiment+".tsv",sep='\t',index=False)
    epsilons_df.to_csv(folder_rd+"/epsilons_signor_"+this_experiment+".tsv",sep='\t',index=False)
    df_exp.to_csv(folder_rd+"/pvals_signor_"+this_experiment+".tsv",sep='\t')

def finalize_optim_and_save(myGraph):
    global beta, experiment, cell_line, activator
    myGraph.map_elements_to_int()
    eps_crit,_ = myGraph.find_eps_critic(beta,1e-1)
    _ = myGraph.optimize_biased_walk_normalized(eps_crit,beta)
    myGraph.add_meta_after_optim(eps_crit,beta,'EGF')
    myGraph.eps_crit = eps_crit
    myGraph.map_elements_to_names()
    
    diG = myGraph.get_directed_subgraph()
    diG_from_source = myGraph.get_directed_subgraph_from_sources()
    diG_edge_proba = myGraph.get_higher_edge_proba_subgraph()
    
    this_experiment = experiment+'_'+cell_line+'_'+ activator
    
    nx.write_gml(diG,folder_exp + 'hpn_dig_' + this_experiment+'.gml')
    nx.write_gml(diG_from_source,folder_exp + 'hpn_digfrom_'+this_experiment+'.gml')
    nx.write_gml(diG_edge_proba,folder_exp + 'hpn_digedge_'+this_experiment+'.gml')
    nx.write_gml(myGraph.subGdirected_with_annot,folder_exp + 'hpn_subgraph_'+this_experiment+'.gml')
    
if __name__ == '__main__':
    file_name = sys.argv[1]
    set_constants(file_name)
    
    myG,nxG = build_network(file_name)
    target_dict = get_targets(file_name,nxG)
    myG.set_targets(target_dict,self_loops=False)
    
    run_obrwr(myG)
    finalize_optim_and_save(myG)