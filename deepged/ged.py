import torch
from deepged.utils import from_networkx_to_tensor
import deepged.optim as optim


def matrix_edge_ins_del(A1, A2):
    '''
    Doc TODO
    '''
    Abin1 = (A1 != torch.zeros(
        (A1.shape[0], A1.shape[1])))
    Abin2 = (A2 != torch.zeros(
        (A2.shape[0], A2.shape[1])))
    C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
    C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
    C12 = torch.logical_or(C1, C2).int()

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)


def matrix_edge_subst(A1, A2, lab1, lab2):
    '''
    Doc TODO
    '''
    Abin1 = (
        A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]))).int()
    Abin2 = (
        A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]))).int()
    C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1).float()


def construct_cost_matrix(A_g1, A_g2, card, labels,
                          nb_edge_labels,
                          node_costs, edge_costs,
                          node_ins_del, edge_ins_del):
    '''
    Retourne une matrice carrée de taile (n+1) * (m +1) contenant les couts sur les noeuds et les aretes
    TODO : a analyser, tester et documenter
    ATTENTION : fonction copier/collé de model.py
    Pourquoi pas à transformer en "objet", indépendant de model.py
    '''
    n = card[0]
    m = card[1]

    A1 = torch.zeros((n + 1, n + 1), dtype=torch.int)
    A1[0:n, 0:n] = A_g1[0:n * n].view(n, n)
    A2 = torch.zeros((m + 1, m + 1), dtype=torch.int)
    A2[0:m, 0:m] = A_g2[0:m * m].view(m, m)
    A = matrix_edge_ins_del(A1, A2)
    C = edge_ins_del * A
    if nb_edge_labels > 1:
        for k in range(nb_edge_labels):
            for l in range(nb_edge_labels):
                if k != l:
                    C.add_(matrix_edge_subst(A1, A2, k + 1,
                                             l + 1).multiply_(edge_costs[k][l]))
    l1 = labels[0][0:n]
    l2 = labels[1][0:m]
    D = torch.zeros((n + 1) * (m + 1))
    D[n * (m + 1):] = node_ins_del
    D[n * (m + 1) + m] = 0
    D[[i * (m + 1) + m for i in range(n)]] = node_ins_del
    for k in range(n * (m + 1)):
        if k % (m + 1) != m:
            D[k] = node_costs[l1[k // (m + 1)]][l2[k % (m + 1)]]

    mask = torch.diag(torch.ones_like(D))
    C = mask * torch.diag(D) + (1. - mask) * C
    return C


def rearrange_costs(costs, nb_node_labels, nb_edge_labels):

    np_cns = torch.Tensor(costs[0])
    cndl = torch.Tensor(costs[1])
    np_ces = torch.Tensor(costs[2])
    cedl = torch.Tensor(costs[3])

    cns = torch.zeros((nb_node_labels, nb_node_labels))
    upper_part = torch.triu_indices(
        cns.shape[0], cns.shape[1], offset=1)
    cns[upper_part[0], upper_part[1]] = np_cns
    cns = cns + cns.T

    if nb_edge_labels > 1:
        ces = torch.zeros((nb_edge_labels, nb_edge_labels))
        upper_part = torch.triu_indices(
            ces.shape[0], ces.shape[1], offset=1)
        ces[upper_part[0], upper_part[1]] = np_ces
        ces = ces + ces.T
    else:
        ces = torch.zeros(0)

    return cns, cndl, ces, cedl


def ged(g1, g2, costs, dict_nodes, nb_node_labels, nb_edge_labels, node_label="label"):
    '''
    G1,G2 : networkx graphs
    costs: structure : avec cns, cndl, ces et cedl sous forme de vecteurs
    dict_nodes : dictionnary to associate labels to indexes used in costs

    '''

    cns, cndl, ces, cedl = rearrange_costs(
        costs, nb_node_labels, nb_edge_labels)

    A_g1, labels_1 = from_networkx_to_tensor(
        g1, dict_nodes, node_label)
    A_g2, labels_2 = from_networkx_to_tensor(
        g2, dict_nodes, node_label)

    n = g1.order()
    m = g2.order()
    # a externatliser
    C = construct_cost_matrix(
        A_g1, A_g2, [n, m], [labels_1, labels_2], nb_edge_labels,
        torch.Tensor(cns), torch.Tensor(ces), torch.Tensor(cndl), torch.Tensor(cedl))
    c = torch.diag(C)
    D = C - torch.eye(C.shape[0]) * c
    S = torch.exp(-.5*c.view(n+1, m+1))
    # S = optim.from_cost_to_similarity_exp(c.view(n+1, m+1))
    X = optim.sinkhorn_diff(S, 10).view((n+1)*(m+1), 1)
    X = optim.franck_wolfe(X, D, c, 5, 10, n, m)

    normalize_factor = 1.0
    nb_edge1 = (A_g1[0:n * n] != torch.zeros(n * n)).int().sum()
    nb_edge2 = (A_g2[0:m * m] != torch.zeros(m * m)).int().sum()
    normalize_factor = cndl * (n + m) + cedl * (nb_edge1 + nb_edge2)

    v = torch.flatten(X)
    ged = (.5 * v.T @ D @ v + c.T @ v)/normalize_factor
    return ged
