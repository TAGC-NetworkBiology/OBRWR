import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

def topo_dist_plot(ax,mG):
    G = mG.subGdirected_with_annot
    source = mG.get_sources()[0]
    L = mG.get_targets()
    L = [el for el in L if el != source]

    ax.axis('equal')

    scircle = mpatches.Circle((0, 0), .2, ec="none")
    scircle.set(color = cm.Reds_r(0))
    ax.text(0,0,mG.inverse_mapping[source],color='white',horizontalalignment='center',fontsize=12)
    ax.add_artist(scircle)

    distances = {el:[] for el in L}
    for el in distances.keys():
        count = sum(1 for _ in nx.all_shortest_paths(G,source,el))
        d = nx.shortest_path_length(G,source,el)
        distances[el] = (d,count)

    binned = [[] for _ in range(max([v[0] for v in distances.values()])+1)]
    for k,v in distances.items():
        binned[v[0]].append(k)

    maxR = len(binned)
    d = {}
    for i,l in enumerate(binned[1:]):
        r = i+1
        n = len(l)
        sradius = min(0.9*r*np.pi/n,.2)
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r * np.cos(an), r * np.sin(an),ls='--',c = 'grey', alpha= 0.5)
        for k in range(n):
            pos = np.array((np.cos(2*k*np.pi/n),np.sin(2*k*np.pi/n)))
            d[l[k]] = pos
            pos = r * pos
            ax.text(pos[0],pos[1],mG.inverse_mapping[l[k]],
                   horizontalalignment='center')
            el = mpatches.Circle(pos,sradius,ec="none")
            el.set(color = cm.Reds_r(r/maxR),alpha=1)
            ax.add_artist(el)

    ax.set(xticks=[],yticks=[],xlim=(-maxR,maxR),ylim=(-maxR,maxR))        
    ax.set(title='Targets topological distance to EGFR')
    return d

def triple_distance_plot(mG):
    targets = G.get_targets()
    f,ax = plt.subplots
