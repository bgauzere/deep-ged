import torch
from graph_torch import rings
import svd
from gklearn.utils.graphfiles import loadDataset
import networkx as nx

import os
dataset_path = os.getenv('MAO_DATASET_PATH')
Gs, y = loadDataset(dataset_path)


def from_weighs_to_costs(self):
    # We apply the ReLU (rectified linear unit) function element-wise
    relu = torch.nn.ReLU()
    cn = relu(self.node_weighs)
    ce = relu(self.edge_weighs)
    edgeInsDel = ce[-1]

    # Or we can use the exponential function
    # Returns a new tensor with the exponential of the elements of the input tensor
    '''
    #cn=torch.exp(self.node_weighs)
    #ce=torch.exp(self.edge_weighs)
    cn=self.node_weighs*self.node_weighs
    ce=self.edge_weighs*self.edge_weighs
    total_cost=cn.sum()+ce.sum()
    cn=cn/total_cost #/max
    ce=ce/total_cost
    edgeInsDel=ce[-1]
    '''

    # Initialization of the node costs
    node_costs = torch.zeros(
        (self.nb_labels, self.nb_labels), device=self.device)
    upper_part = torch.triu_indices(
        node_costs.shape[0], node_costs.shape[1], offset=1, device=self.device)
    node_costs[upper_part[0], upper_part[1]] = cn[0:-1]
    node_costs = node_costs + node_costs.T

    if self.nb_edge_labels > 1:
        edge_costs = torch.zeros(
            (self.nb_edge_labels, self.nb_edge_labels), device=self.device)
        upper_part = torch.triu_indices(
            edge_costs.shape[0], edge_costs.shape[1], offset=1, device=self.device)
        edge_costs[upper_part[0], upper_part[1]] = ce[0:-1]
        edge_costs = edge_costs + edge_costs.T
        del upper_part
        torch.cuda.empty_cache()
    else:
        edge_costs = torch.zeros(0, device=self.device)

    return node_costs, cn[-1], edge_costs, edgeInsDel


# Extraction of all atom labels
def build_node_dictionnary(GraphList, node_label, edge_label):
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
    print("node_labels : ", node_labels)

    return dict, max(max([[int(G[e[0]][e[1]][edge_label]) for e in G.edges()] for G in GraphList]))


# Transforming a networkx to a torch tensor
def from_networkx_to_tensor(G, dict, node_label):
    A = torch.tensor(nx.to_scipy_sparse_matrix(
        G, dtype=int, weight='bond_type').todense(), dtype=torch.int)
    lab = [dict[G.nodes[v][node_label][0]] for v in nx.nodes(G)]

    return (A.view(1, A.shape[0] * A.shape[1]), torch.tensor(lab))


# This function is used to construct a cost matrix C between two graphs g1 and g2, given the costs
def construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, node_label, node_label_dict, nb_edge_labels):
    n = g1.order()
    m = g2.order()

    A1 = torch.zeros((n + 1, n + 1))
    A, l1 = from_networkx_to_tensor(g1, node_label_dict, node_label)
    A1[0:n, 0:n] = A.view(n, n)
    A2 = torch.zeros((m + 1, m + 1))
    A, l2 = from_networkx_to_tensor(g2, node_label_dict, node_label)
    A2[0:m, 0:m] = A.view(m, m)

    C = edgeInsDel * matrix_edgeInsDel(A1, A2)
    if nb_edge_labels > 1:
        for k in range(nb_edge_labels):
            for l in range(nb_edge_labels):
                if k != l:
                    C.add_(matrix_edgeSubst(A1, A2, k + 1,
                           l + 1).multiply_(edge_costs[k][l]))
                    C = C + edge_costs[k][l] * \
                        matrix_edgeSubst(A1, A2, k + 1, l + 1)

    D = torch.zeros((n + 1) * (m + 1))
    D[n * (m + 1):] = nodeInsDel
    D[n * (m + 1) + m] = 0
    D[[i * (m + 1) + m for i in range(n)]] = nodeInsDel
    for k in range(n * (m + 1)):
        if k % (m + 1) != m:
            D[k] = node_costs[l1[k // (m + 1)]][l2[k % (m + 1)]]
    mask = torch.diag(torch.ones_like(D))
    C = mask * torch.diag(D)  # + (1. - mask)*C

    return C


def matrix_edgeInsDel(A1, A2):
    Abin1 = (A1 != torch.zeros((A1.shape[0], A1.shape[1])))
    Abin2 = (A2 != torch.zeros((A2.shape[0], A2.shape[1])))
    C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
    C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
    C12 = torch.logical_or(C1, C2).int()
    return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)


def matrix_edgeSubst(A1, A2, lab1, lab2):
    Abin1 = (A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]))).int()
    Abin2 = (A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]))).int()
    C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)
    return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1).float()


# ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']
def lsape_populate_instance(first_graph, second_graph, average_node_cost, average_edge_cost, alpha, lbda, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h):  # ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']
    g, h = Gs[first_graph], Gs[second_graph]
    average_cost = [average_node_cost, average_edge_cost]
    first_graph, second_graph = first_graph, second_graph

    # node_costs, nodeInsDel, edge_costs, edgeInsDel = from_weighs_to_costs()

    lsape_instance = [[0 for _ in range(len(g) + 1)] for __ in range(len(h) + 1)]
    for g_node_index in range(len(g) + 1):
        for h_node_index in range(len(h) + 1):
            lsape_instance[h_node_index][g_node_index] = rings.compute_ring_distance(g, h, ring_g, ring_h,
                                                                                     g_node_index, h_node_index, alpha,
                                                                                     lbda, node_costs, nodeInsDel,
                                                                                     edge_costs, edgeInsDel,
                                                                                     first_graph, second_graph)
    for i in lsape_instance:
        i = torch.as_tensor(i)
    lsape_instance = torch.as_tensor(lsape_instance)
    # print(type(lsape_instance))
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
        x0 = svd.eps_assign2(torch.exp(-.5 * c_0.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        res = x0

    elif (rings_andor_fw == 'rings_avec_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, node_costs, nodeInsDel,
                                      edge_costs, edgeInsDel, ring_g, ring_h)
        x0=svd.eps_assign2(torch.exp(-.5*c_0.view(n+1,m+1)),10).view((n+1)*(m+1),1)
        res = svd.franck_wolfe(x0, D, c, 5, 15, n, m)

    elif (rings_andor_fw == 'sans_rings_avec_fw'):
        x0=svd.eps_assign2(torch.exp(-.5*c.view(n+1,m+1)),10).view((n+1)*(m+1),1)
        res = svd.franck_wolfe(x0, D, c, 5, 15, n, m)

    elif (rings_andor_fw == 'sans_rings_sans_fw'):
        x0=svd.eps_assign2(torch.exp(-.5*c.view(n+1,m+1)),10).view((n+1)*(m+1),1)
        res = x0

    return res
