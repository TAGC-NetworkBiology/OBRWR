import networkx as nx
import matplotlib.pyplot as plt

def my_draw(G, pos=None, ax=None, **kwds):
    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor("w")
    if ax is None:
        if cf.axes:
            ax = cf.gca()
        else:
            ax = cf.add_axes((0, 0, 1, 1))

    if "with_labels" not in kwds:
        kwds["with_labels"] = "labels" in kwds

    node_collection = my_draw_networkx(G, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    return node_collection

def my_draw_networkx(G, pos=None, arrows=None, with_labels=True, **kwds):
    from inspect import signature

    # Get all valid keywords by inspecting the signatures of draw_networkx_nodes,
    # draw_networkx_edges, draw_networkx_labels

    valid_node_kwds = signature(nx.draw_networkx_nodes).parameters.keys()
    valid_edge_kwds = signature(nx.draw_networkx_edges).parameters.keys()
    valid_label_kwds = signature(nx.draw_networkx_labels).parameters.keys()

    # Create a set with all valid keywords across the three functions and
    # remove the arguments of this function (draw_networkx)
    valid_kwds = (valid_node_kwds | valid_edge_kwds | valid_label_kwds) - {
        "G",
        "pos",
        "arrows",
        "with_labels",
    }

    if any(k not in valid_kwds for k in kwds):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}

    if pos is None:
        pos = nx.drawing.spring_layout(G)  # default to spring layout

    node_collection = nx.draw_networkx_nodes(G, pos, **node_kwds)
    nx.draw_networkx_edges(G, pos, arrows=arrows, **edge_kwds)
    if with_labels:
        nx.draw_networkx_labels(G, pos, **label_kwds)
    plt.draw_if_interactive()
    return node_collection