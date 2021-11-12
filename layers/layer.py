from graph_torch import rings, svd, ged_torch
from tqdm import tqdm
from torch import nn

import GPUtil
import torch
import sys
import networkx as nx
import numpy as np
import sys

class GedLayer(nn.Module):
    def __init__(self, nb_node_pair_label, nb_edge_pair_label, rings_andor_fw, node_label, node_label_dict, nb_edge_label):
        super(GedLayer, self).__init__()
        self.node_weighs = nn.Parameter(torch.tensor(1.0 / (nb_node_pair_label + nb_edge_pair_label + 2)) + (1e-3) *
                                        torch.rand(int(nb_node_pair_label + 1), requires_grad=True, device=self.device))
        self.edge_weighs = nn.Parameter(torch.tensor(1.0 / (nb_node_pair_label + nb_edge_pair_label + 2)) + (1e-3) *
                                        torch.rand(int(nb_edge_pair_label + 1), requires_grad=True, device=self.device))
        self.rings_andor_fw = rings_andor_fw
        self.node_label = node_label
        self.node_label_dict = node_label_dict
        self.nb_edge_label = nb_edge_label

    def forward(self, input):
        g1 = input[0]
        g2 = input[1]

        cns, cndl, ces, cedl = self.from_weighs_to_costs()

        n = g1.order()
        m = g2.order()
        if self.rings_andor_fw == 'rings_sans_fw':
            ring_g, ring_h = rings.build_rings(g1, cedl.size()), rings.build_rings(g2, cedl.size())
            C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, self.node_label, self.node_label_dict,
                                                self.nb_edge_labels)
            S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, ring_g, ring_h,
                                                   self.rings_andor_fw)
            # S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)

        elif self.rings_andor_fw == 'rings_avec_fw':
            ring_g, ring_h = rings.build_rings(g1, cedl.size()), rings.build_rings(g2, cedl.size())
            C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, self.node_label, self.node_label_dict,
                                                self.nb_edge_labels)
            S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, ring_g, ring_h,
                                                   self.rings_andor_fw)
            # S = new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h)
        elif self.rings_andor_fw == 'sans_rings_avec_fw':
            C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, self.node_label, self.node_label_dict,
                                                self.nb_edge_labels)
            S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, 0, 0, self.rings_andor_fw)
            # S = mapping_from_cost(C, n, m)
        elif self.rings_andor_fw == 'sans_rings_sans_fw':
            C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, self.node_label, self.node_label_dict,
                                                self.nb_edge_labels)
            S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, 0, 0, self.rings_andor_fw)

        else:
            print("Error : rings_andor_fw => value not understood")
            sys.exit()

        v = torch.flatten(S)
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], ) * c
        ged = (.5 * v.T @ D @ v + c.T @ v)
        return ged


# The class Net representing the neural network :

class Net(nn.Module):

    def __init__(self, GraphList, rings_andor_fw, normalize=False, node_label='label'):
        super(Net, self).__init__()
        self.rings_andor_fw = rings_andor_fw
        self.normalize = normalize
        self.node_label = node_label
        self.GraphList = GraphList
        dict, self.nb_edge_labels = self.build_node_dictionnary(GraphList)
        self.nb_labels = len(dict)
        # print("nb_edge_labels = ", self.nb_edge_labels)
        # self.device = torch.device("cuda:0")  # 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device("cpu")
        nb_node_pair_label = self.nb_labels * (self.nb_labels - 1) / 2.0
        nb_edge_pair_label = int(self.nb_edge_labels * (self.nb_edge_labels - 1) / 2)

        self.node_weighs = nn.Parameter(
            torch.tensor(1.0 / (nb_node_pair_label + nb_edge_pair_label + 2)) + (1e-3) * torch.rand(
                int(self.nb_labels * (self.nb_labels - 1) / 2 + 1), requires_grad=True,
                device=self.device))  # all substitution costs+ nodeIns/Del. old version: 0 node subs, 1 nodeIns/Del, 2 : edgeSubs, 3 edgeIns/Del
        self.edge_weighs = nn.Parameter(
            torch.tensor(1.0 / (nb_node_pair_label + nb_edge_pair_label + 2)) + (1e-3) * torch.rand(
                nb_edge_pair_label + 1, requires_grad=True, device=self.device))  # edgeIns/Del

        print("Length of node_weights : ",np.shape(self.node_weighs))
        print("Nb node pair label : ", nb_node_pair_label)
        print("Length of edge_weights : ",np.shape(self.edge_weighs))
        print("Nb edge pair label : ", nb_edge_pair_label)
        print(dict)
        # sys.exit()

        self.card = torch.tensor([G.order() for G in GraphList]).to(self.device)
        card_max = self.card.max()
        self.A = torch.zeros((len(GraphList), card_max * card_max), dtype=torch.int, device=self.device)
        self.labels = torch.zeros((len(GraphList), card_max), dtype=torch.int, device=self.device)  # node labels
        for k in range(len(GraphList)):
            A, l = self.from_networkx_to_tensor(GraphList[k], dict)
            self.A[k, 0:A.shape[1]] = A[0]
            self.labels[k, 0:l.shape[0]] = l
        # print('adjacency matrices', self.A)
        # print('node labels', self.labels)
        # print('order of the graphs', self.card)

    # The forward pass of the neural network
    def forward(self, input):
        self = self.to(self.device)
        input = input.to(self.device)
        ged = torch.zeros(len(input)).to(self.device)
        node_costs, nodeInsDel, edge_costs, edgeInsDel = self.from_weighs_to_costs()

        # Here, we empty the cache so that the gpu doesn't keep all the information all along
        torch.cuda.empty_cache()
        GPUtil.showUtilization(all=True)

        for k in tqdm(range(len(input))):
            # GPUtil.showUtilization(all=True)
            g1 = input[k][0].to(self.device)
            g2 = input[k][1].to(self.device)
            n = self.card[g1]
            m = self.card[g2]

            if self.rings_andor_fw == 'rings_sans_fw':
                self.ring_g, self.ring_h = rings.build_rings(self.GraphList[g1], edgeInsDel.size()), rings.build_rings(self.GraphList[g2], edgeInsDel.size())
                C = self.construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
                c = torch.diag(C)
                D = C - torch.eye(C.shape[0]) * c

                c_0 = self.lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, self.GraphList)
                S = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)

                # S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)

            elif self.rings_andor_fw == 'rings_avec_fw':
                self.ring_g, self.ring_h = rings.build_rings(self.GraphList[g1], edgeInsDel.size()), rings.build_rings(self.GraphList[g2], edgeInsDel.size())
                C = self.construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
                c = torch.diag(C)
                D = C - torch.eye(C.shape[0]) * c

                c_0 = self.lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, self.GraphList)
                x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)
                S = svd.franck_wolfe(x0, D, c, 5, 15, n, m)
                # S = new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h)
            elif self.rings_andor_fw == 'sans_rings_avec_fw':
                C = self.construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
                c = torch.diag(C)
                D = C - torch.eye(C.shape[0]) * c
                x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
                S = svd.franck_wolfe(x0, D, c, 5, 15, n, m)

                # S = mapping_from_cost(C, n, m)
            elif self.rings_andor_fw == 'sans_rings_sans_fw':
                C = self.construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
                c = torch.diag(C)
                D = C - torch.eye(C.shape[0]) * c
                S = svd.eps_assigment_from_mapping(torch.exp(-c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
            else:
                print("Error : rings_andor_fw => value not understood")
                sys.exit()

            v = torch.flatten(S).to(self.device)

            # Detaching from the current graph, so that the result will never require gradient
            S = S.detach()
            normalize_factor = 1.0
            if self.normalize:
                nb_edge1 = (self.A[g1][0:n * n] != torch.zeros(n * n, device=self.device)).int().sum()
                nb_edge2 = (self.A[g2][0:m * m] != torch.zeros(m * m, device=self.device)).int().sum()
                normalize_factor = nodeInsDel * (n + m) + edgeInsDel * (nb_edge1 + nb_edge2)

            ged[k] = (.5 * v.T @ D @ v + c.T @ v) / normalize_factor

            # We delete C after every iteration so that it's not kept in memory, because torch keeps it automatically
            del C
            torch.cuda.empty_cache()

        max = torch.max(ged)
        min = torch.min(ged)
        ged = (ged - min) / (max - min)

        return ged


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
        node_costs = torch.zeros((self.nb_labels, self.nb_labels), device=self.device)
        upper_part = torch.triu_indices(node_costs.shape[0], node_costs.shape[1], offset=1, device=self.device)
        node_costs[upper_part[0], upper_part[1]] = cn[0:-1]
        node_costs = node_costs + node_costs.T

        if self.nb_edge_labels > 1:
            edge_costs = torch.zeros((self.nb_edge_labels, self.nb_edge_labels), device=self.device)
            upper_part = torch.triu_indices(edge_costs.shape[0], edge_costs.shape[1], offset=1, device=self.device)
            edge_costs[upper_part[0], upper_part[1]] = ce[0:-1]
            edge_costs = edge_costs + edge_costs.T
            del upper_part
            torch.cuda.empty_cache()
        else:
            edge_costs = torch.zeros(0, device=self.device)

        return node_costs, cn[-1], edge_costs, edgeInsDel


    # Extraction of all atom labels
    def build_node_dictionnary(self, GraphList):
        node_labels = []
        for G in GraphList:
            for v in nx.nodes(G):
                if not G.nodes[v][self.node_label][0] in node_labels:
                    node_labels.append(G.nodes[v][self.node_label][0])
        node_labels.sort()
        # Extraction of a dictionary allowing to number each label by a number.
        dict = {}
        k = 0
        for label in node_labels:
            dict[label] = k
            k = k + 1
        print("node_labels : ", node_labels)

        return dict, max(max([[int(G[e[0]][e[1]]['bond_type']) for e in G.edges()] for G in GraphList]))


    # Transforming a networkx to a torch tensor
    def from_networkx_to_tensor(self, G, dict):
        A = torch.tensor(nx.to_scipy_sparse_matrix(G, dtype=int, weight='bond_type').todense(), dtype=torch.int)
        lab = [dict[G.nodes[v][self.node_label][0]] for v in nx.nodes(G)]
        print(torch.tensor(lab))
        return (A.view(1, A.shape[0] * A.shape[1]), torch.tensor(lab))


    # This function is used to construct a cost matrix C between two graphs g1 and g2, given the costs
    def construct_cost_matrix(self, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel):
        n = self.card[g1].item()
        m = self.card[g2].item()

        # We use the no_grad to disable gradient calculation, that will reduce memory consumption
        with torch.no_grad():
            A1 = torch.zeros((n + 1, n + 1), dtype=torch.int, device=self.device)
            A1[0:n, 0:n] = self.A[g1][0:n * n].view(n, n)
            A2 = torch.zeros((m + 1, m + 1), dtype=torch.int, device=self.device)
            A2[0:m, 0:m] = self.A[g2][0:m * m].view(m, m)

        # costs: 0 node subs, 1 nodeIns/Del, 2 : edgeSubs, 3 edgeIns/Del

        # C=cost[3]*torch.cat([torch.cat([C12[l][k] for k in range(n+1)],1) for l in range(n+1)])
        C = edgeInsDel * self.matrix_edgeInsDel(A1, A2)
        if self.nb_edge_labels > 1:
            for k in range(self.nb_edge_labels):
                for l in range(self.nb_edge_labels):
                    if k != l:
                        C.add_(self.matrix_edgeSubst(A1, A2, k + 1, l + 1).multiply_(edge_costs[k][l]))
                        C = C + edge_costs[k][l] * self.matrix_edgeSubst(A1, A2, k + 1, l + 1)

        l1 = self.labels[g1][0:n]
        l2 = self.labels[g2][0:m]
        D = torch.zeros((n + 1) * (m + 1), device=self.device)
        D[n * (m + 1):] = nodeInsDel
        D[n * (m + 1) + m] = 0
        D[[i * (m + 1) + m for i in range(n)]] = nodeInsDel
        for k in range(n * (m + 1)):
            if k % (m + 1) != m:
                D[k] = node_costs[l1[k // (m + 1)]][l2[k % (m + 1)]]
        mask = torch.diag(torch.ones_like(D))
        C = mask * torch.diag(D)  # + (1. - mask)*C

        return C


    def matrix_edgeInsDel(self, A1, A2):
        Abin1 = (A1 != torch.zeros((A1.shape[0], A1.shape[1]), device=self.device))
        Abin2 = (A2 != torch.zeros((A2.shape[0], A2.shape[1]), device=self.device))
        C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
        C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
        C12 = torch.logical_or(C1, C2).int()
        return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)


    def matrix_edgeSubst(self, A1, A2, lab1, lab2):
        Abin1 = (A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]), device=self.device)).int()
        Abin2 = (A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]), device=self.device)).int()
        C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)
        return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1).float()


    # ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']
    def lsape_populate_instance(self, first_graph, second_graph, average_node_cost, average_edge_cost, alpha, lbda, GraphList):
        g, h = GraphList[first_graph], GraphList[second_graph]
        self.average_cost = [average_node_cost, average_edge_cost]
        self.first_graph, self.second_graph = first_graph, second_graph

        node_costs, nodeInsDel, edge_costs, edgeInsDel = self.from_weighs_to_costs()

        lsape_instance = [[0 for _ in range(len(g) + 1)] for __ in range(len(h) + 1)]
        for g_node_index in range(len(g) + 1):
            for h_node_index in range(len(h) + 1):
                lsape_instance[h_node_index][g_node_index] = rings.compute_ring_distance(g, h, self.ring_g, self.ring_h,
                                                                                         g_node_index, h_node_index,
                                                                                         alpha, lbda, node_costs,
                                                                                         nodeInsDel, edge_costs,
                                                                                         edgeInsDel, first_graph,
                                                                                         second_graph)
        for i in lsape_instance:
            i = torch.as_tensor(i)
        lsape_instance = torch.as_tensor(lsape_instance)
        return lsape_instance


    # Calculating a mapping based on the cost matrix C, using the rings function and a derivable Hungarian approximation
    def mapping_from_cost_sans_FW(self, n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, GraphList):
        c_0 = self.lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, GraphList)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n + 1) * (m + 1), 1)
        return x0


    # Calculating a mapping based on the cost matrix C, not using the rings function and using a derivable Hungarian approximation
    def new_mapping_from_cost(self, C, n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel):
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], device=self.device) * c
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c), 10).view((n + 1) * (m + 1), 1)
        return x0


    # Calculating a mapping based on the cost matrix C, not using the rings function and using the Frank Wolfe algorithm
    def mapping_from_cost(self, C, n, m):
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], device=self.device) * c
        x0 = svd.eps_assigment_from_mapping(torch.exp(-.5 * c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        x = svd.franck_wolfe(x0, D, c, 5, 10, n, m)

        def print_grad(grad):
            if (grad.norm() != 0.0):
                print(grad)

        return x

