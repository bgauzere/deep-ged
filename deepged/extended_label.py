'''
Ce module permet de créer un nouveau label pour chaque noeud d'un graphe networkx  prenant en compte ses voisins immédiats
'''
import networkx as nx
import os
from gklearn.utils import load_dataset


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


if __name__ == '__main__':
    dataset_path = os.getenv('MAO_DATASET_PATH')
    Gs, y, labels = load_dataset(dataset_path)
    for g in Gs:
        compute_extended_labels(g, label_node='atom_symbol')
    # Verif, un label extended_label devrait apparaitre
    for v in Gs[12].nodes():
        print(Gs[12].nodes[v])
