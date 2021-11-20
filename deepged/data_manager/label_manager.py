import networkx as nx


def label_to_color(label):
    if label == 'C':
        return 0.1
    elif label == 'O':
        return 0.8


def nodes_to_color_sequence(G):
    return [label_to_color(c[1]['label'][0]) for c in G.nodes(data=True)]


def compute_star(G, id_node, label_node, label_edge):
    '''
    Calcule une string contenant les labels des noeuds voisins plus le label du noeud central.
    '''
    central_label = G.nodes(data=True)[id_node][label_node]
    neighs = []
    for id_neigh, labels_e in G[id_node].items():
        neigh_label = G.nodes(data=True)[id_neigh][label_node]
        extended_label = ''.join([labels_e[label_edge], neigh_label[0]])
        neighs.append(extended_label)
    neigh_labels = ''.join(sorted(neighs, key=str))
    new_label = ''.join([central_label[0], '_', neigh_labels])
    return new_label


def compute_extended_labels(G, label_node='atom_symbol', label_edge='bond_type'):
    '''
    Calcule l'ensemble des labels étendus pour un graphe G.
    Rajoute le nouveau label au graphe G
    '''
    for v in G.nodes():
        new_label = compute_star(G, v, label_node, label_edge)
        G.nodes[v]['extended_label'] = [new_label]
    return G


def build_node_dictionnary(GraphList,
                           node_label="atom_type", edge_label="bond_type"):
    '''
    Associe un index a chaque label rencontré dans les graphes
    Retourne un dictionnaire associant un label a un entier  et le nombre de labels d'aretes
    TODO : fonction a pythoniser et optimiser
    '''
    ensemble_labels = []
    for G in GraphList:
        for v in nx.nodes(G):
            if not G.nodes[v][node_label][0] in ensemble_labels:
                ensemble_labels.append(G.nodes[v][node_label][0])
    ensemble_labels.sort()
    # extraction d'un dictionnaire permettant de numéroter chaque label par un numéro.
    dict_labels = {}
    k = 0
    for label in ensemble_labels:
        dict_labels[label] = k
        k = k + 1
    nb_edge_labels = max(
        max([[int(G[e[0]][e[1]][edge_label]) for e in G.edges()] for G in GraphList]))

    return dict_labels, nb_edge_labels


if __name__ == '__main__':
    dataset_path = os.getenv('MAO_DATASET_PATH')
    Gs, y, labels = load_dataset(dataset_path)
    for g in Gs:
        compute_extended_labels(g, label_node='atom_symbol')
    # Verif, un label extended_label devrait apparaitre
    for v in Gs[12].nodes():
        print(Gs[12].nodes[v])
