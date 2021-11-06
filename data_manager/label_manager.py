def label_to_color(label):
    if label == 'C':
        return 0.1
    elif label == 'O':
        return 0.8


def nodes_to_color_sequence(G):
    return [label_to_color(c[1]['label'][0]) for c in G.nodes(data=True)]

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