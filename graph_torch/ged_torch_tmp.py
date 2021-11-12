from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import svd
import rings
from gklearn.utils.graphfiles import loadDataset
import networkx as nx
import matplotlib

matplotlib.use('TkAgg')


# extraction of all atom labels
def build_node_dictionnary(GraphList):
    node_label = 'label'
    node_labels = []
    for G in GraphList:
        for v in nx.nodes(G):
            if not G.nodes[v][node_label][0] in node_labels:
                node_labels.append(G.nodes[v][node_label][0])
    node_labels.sort()
    # Extraction of a dictionary allowing to number each label by a number.
    dict = {}
    k = 0
    for label in node_labels:
        dict[label] = k
        k = k + 1

    return dict, max(max([[int(G[e[0]][e[1]]['bond_type']) for e in G.edges()] for G in GraphList]))


# Transforming a networkx to a torch tensor
def from_networkx_to_tensor(G, dict):
    A_g = torch.tensor(nx.to_scipy_sparse_matrix(G, dtype=int, weight='bond_type').todense(), dtype=torch.int)
    lab = [dict[G.nodes[v]['label'][0]] for v in nx.nodes(G)]

    return A_g.view(1, A_g.shape[0] * A_g.shape[1]), torch.tensor(lab)


def init_dataset(Gs, dict):
    for k in range(len(Gs)):
        A_k, l = from_networkx_to_tensor(Gs[k], dict)  # adjacency matrixes
        A[k, 0:A_k.shape[1]] = A_k[0]
        labels[k, 0:l.shape[0]] = l


# This function is used to construct a cost matrix C between two graphs g1 and g2, given the costs
def construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel):
    n = card[g1].item()
    m = card[g2].item()

    A1 = torch.zeros((n + 1, n + 1), dtype=torch.int, device=device)
    A1[0:n, 0:n] = A[g1][0:n * n].view(n, n)
    A2 = torch.zeros((m + 1, m + 1), dtype=torch.int, device=device)
    A2[0:m, 0:m] = A[g2][0:m * m].view(m, m)

    C = edgeInsDel * matrix_edgeInsDel(A1, A2)
    if nb_edge_labels > 1:
        for k in range(nb_edge_labels):
            for l in range(nb_edge_labels):
                if k != l:
                    C = C + edge_costs[k][l] * matrix_edgeSubst(A1, A2, k + 1, l + 1)

    l1 = labels[g1][0:n]
    l2 = labels[g2][0:m]
    D = torch.zeros((n + 1) * (m + 1), device=device)
    D[n * (m + 1):] = nodeInsDel
    D[n * (m + 1) + m] = 0
    D[[i * (m + 1) + m for i in range(n)]] = nodeInsDel
    D[[k for k in range(n * (m + 1)) if k % (m + 1) != m]] = torch.tensor([node_costs[l1[k // (
            m + 1)], l2[k % (m + 1)]] for k in range(n * (m + 1)) if k % (m + 1) != m], device=device)
    mask = torch.diag(torch.ones_like(D))
    C = mask * torch.diag(D) + (1. - mask) * C

    return C


def matrix_edgeInsDel(A1, A2):
    Abin1 = (A1 != torch.zeros((A1.shape[0], A1.shape[1]), device=device))
    Abin2 = (A2 != torch.zeros((A2.shape[0], A2.shape[1]), device=device))
    C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
    C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
    C12 = torch.logical_or(C1, C2).int()

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)


def matrix_edgeSubst(A1, A2, lab1, lab2):
    Abin1 = (A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]), device=device)).int()
    Abin2 = (A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]), device=device)).int()
    C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1)


# ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']
def lsape_populate_instance(first_graph, second_graph, average_node_cost, average_edge_cost, alpha, lbda, node_costs,
                            nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h):
    g, h = Gs[first_graph], Gs[second_graph]
    lsape_instance = [[0 for _ in range(len(g) + 1)]
                      for __ in range(len(h) + 1)]
    for g_node_index in range(len(g) + 1):
        for h_node_index in range(len(h) + 1):
            lsape_instance[h_node_index][g_node_index] = rings.compute_ring_distance(
                g, h, ring_g, ring_h, g_node_index, h_node_index, alpha, lbda, node_costs, nodeInsDel, edge_costs,
                edgeInsDel, first_graph, second_graph)
    for i in lsape_instance:
        i = torch.as_tensor(i)
    lsape_instance = torch.as_tensor(lsape_instance)
    return lsape_instance


# Finding an adequate mapping based on the given costs, without using the Frank Wolfe method
def mapping_from_cost_sans_FW(n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, ring_g, ring_h):
    c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                  edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)
    return x0


# Finding an adequate mapping based on the given costs, using the Frank Wolfe method, and the rings
def new_mapping_from_cost(C, n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, ring_g, ring_h):
    c = torch.diag(C)
    c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                  edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    D = C - torch.eye(C.shape[0]) * c
    x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)
    return svd.franck_wolfe(x0, D, c, 5, 15, n, m)


# Finding an adequate mapping based on the given costs, using the Frank Wolfe method, without the rings method
def mapping_from_cost(C, n, m):
    c = torch.diag(C)
    D = C - torch.eye(C.shape[0]) * c
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
    return svd.franck_wolfe(x0, D, c, 5, 15, n, m)


def mapping_from_cost_sans_rings_sans_fw(C, n, m):
    c = torch.diag(C)
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
    return x0


# A general function for finding an adequate mapping based on the given costs
def mapping_from_cost_method(C, n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, ring_g, ring_h,
                             rings_andor_fw):
    c = torch.diag(C)
    D = C - torch.eye(C.shape[0]) * c

    if (rings_andor_fw == 'rings_sans_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, node_costs, nodeInsDel,
                                      edge_costs, edgeInsDel, ring_g, ring_h)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)
        res = x0
    if (rings_andor_fw == 'rings_avec_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, node_costs, nodeInsDel,
                                      edge_costs, edgeInsDel, ring_g, ring_h)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)
        res = svd.franck_wolfe(x0, D, c, 5, 15, n, m)
    if (rings_andor_fw == 'sans_rings_avec_fw'):
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        res = svd.franck_wolfe(x0, D, c, 5, 15, n, m)
    if (rings_andor_fw == 'sans_rings_sans_fw'):
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        res = x0
    return res