import networkx as nx


def compute_star(G, id_node, label_node='label', label_edge='bond_type'):
    central_label = G.nodes(data=True)[id_node][label_node]

    neighs = []
    for id_neigh, labels_e in G[id_node].items():
        neigh_label = G.nodes(data=True)[id_neigh][label_node]
        extended_label = ''.join([labels_e[label_edge], neigh_label[0]])
        neighs.append(extended_label)
    neigh_labels = ''.join(sorted(neighs, key=str))
    new_label = ''.join([central_label[0], '_', neigh_labels])
    # print(new_label)
    return new_label


def compute_extended_labels(G, label_node='atom_symbol',
                            label_edge='bond_type'):
    for v in G.nodes():
        new_label = compute_star(G, v)
        G.nodes[v]['extended_label'] = [new_label]
        # print(v,new_label)
    return G


if __name__ == '__main__':
    from gklearn.utils import load_dataset
    Gs, y, labels = load_dataset(
        "/home/luc/TRAVAIL/DeepGED/Acyclic/dataset_bps.ds")
    for g in Gs:
        compute_extended_labels(g)
    # Verif, un label extended_label devrait apparaitre
    for v in Gs[12].nodes():
        print(Gs[12].nodes[v])
