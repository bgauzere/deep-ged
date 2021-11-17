import torch
import torch.nn as nn
import torch.nn.functional as F
import svd
from svd import iterated_power as compute_major_axis
import GPUtil
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#torch.autograd.set_detect_anomaly(True)
import rings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl
from data_manager.label_manager import *
from gklearn.utils.graphfiles import loadDataset
import sys
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')




class Evaluator():
    def __init__(self, GraphList, normalize=False, node_label='label'):
        self.normalize = normalize
        self.node_label = node_label
        dict, self.nb_edge_labels = self.build_node_dictionnary(GraphList)
        self.nb_labels = len(dict)
        print(self.nb_edge_labels)
        # self.device=torch.device("cuda:0")
        self.device = torch.device('cpu')
        self.card = torch.tensor([G.order() for G in GraphList]).to(self.device)
        card_max = self.card.max()
        self.A = torch.empty((len(GraphList), card_max * card_max), dtype=torch.int, device=self.device)
        self.labels = torch.empty((len(GraphList), card_max), dtype=torch.int, device=self.device)
        print(self.A.shape)
        for k in range(len(GraphList)):
            A, l = self.from_networkx_to_tensor(GraphList[k], dict)
            self.A[k, 0:A.shape[1]] = A[0]
            self.labels[k, 0:l.shape[0]] = l
        print('adjacency matrices', self.A)
        print('node labels', self.labels)
        print('order of the graphs', self.card)

    def classification(self, train_data, test_data, train_label, test_label, k, cns, ces, cndl, cedl):

        n_graph_test = len(test_graph)
        n_graph_train = len(train_graph)
        ged_matrix_train = np.zeros((n_graph_train, n_graph_train))
        ged_matrix_test = np.zeros((n_graph_test, n_graph_train))

        for g1 in tqdm(range(len(train_data))):
            for g2 in range(len(train_data)):
                n = self.card[g1]
                m = self.card[g2]
                # with torch.no_grad():
                C = self.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl)

                # self.ring_g,self.ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
                # c_0=self.lsape_populate_instance(g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel)

                # S=self.mapping_from_similarity(C,n,m)
                #            print('g1,g2=',g1.item(),g2.item())
                S = self.mapping_from_cost(C, n, m)
                v = torch.flatten(S)

                normalize_factor = 1.0
                if self.normalize:
                    nb_edge1 = (self.A[g1][0:n * n] != torch.zeros(n * n)).int().sum()
                    nb_edge2 = (self.A[g2][0:m * m] != torch.zeros(m * m)).int().sum()
                    normalize_factor = cndl * (n + m) + cedl * (nb_edge1 + nb_edge2)
                c = torch.diag(C)
                D = C - torch.eye(C.shape[0], device=self.device) * c
                ged = (.5 * v.T @ D @ v + c.T @ v) / normalize_factor
                ged_matrix_train[g1, g2] = ged

        for g1 in tqdm(range(len(test_data))):
            for g2 in range(len(train_data)):
                n = self.card[g1]
                m = self.card[g2]
                # with torch.no_grad():
                C = self.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl)

                # self.ring_g,self.ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
                # c_0=self.lsape_populate_instance(g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel)

                # S=self.mapping_from_similarity(C,n,m)
                #            print('g1,g2=',g1.item(),g2.item())
                S = self.mapping_from_cost(C, n, m)
                v = torch.flatten(S)

                normalize_factor = 1.0
                if self.normalize:
                    nb_edge1 = (self.A[g1][0:n * n] != torch.zeros(n * n)).int().sum()
                    nb_edge2 = (self.A[g2][0:m * m] != torch.zeros(m * m)).int().sum()
                    normalize_factor = cndl * (n + m) + cedl * (nb_edge1 + nb_edge2)
                c = torch.diag(C)
                D = C - torch.eye(C.shape[0], device=self.device) * c
                ged = (.5 * v.T @ D @ v + c.T @ v) / normalize_factor
                ged_matrix_test[g1, g2] = ged

        # Definition of the KNN classifier. precomputed stands for "precomputed distances", which we did by computing GED
        classifier = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        classifier.fit(ged_matrix_train, train_label)
        y_pred_test = classifier.predict(ged_matrix_test)

        print("Test Results : ")
        print(classification_report(test_label, y_pred_test))
        print(confusion_matrix(test_label, y_pred_test))

        plt.subplot(121)
        plt.imshow(ged_matrix_train)
        plt.subplot(122)
        plt.imshow(ged_matrix_test)
        plt.show()

    # def forward(self, input):
    #     ged = torch.zeros(len(input)).to(self.device)
    #     node_costs, nodeInsDel, edge_costs, edgeInsDel = self.from_weighs_to_costs()
    #
    #     torch.cuda.empty_cache()
    #     GPUtil.showUtilization(all=True)
    #
    #     # print('weighs:',self.weighs.device,'device:',self.device,'card:',self.card.device,'A:',self.A.device,'labels:',self.labels.device)
    #     for k in tqdm(range(len(input))):
    #         # print('Dans le forward')
    #         # GPUtil.showUtilization(all=True)
    #
    #         g1 = input[k][0]
    #         g2 = input[k][1]
    #         n = self.card[g1]
    #         m = self.card[g2]
    #         # with torch.no_grad():
    #         C = self.construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
    #
    #         # self.ring_g,self.ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
    #         # c_0=self.lsape_populate_instance(g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel)
    #
    #         # S=self.mapping_from_similarity(C,n,m)
    #         #            print('g1,g2=',g1.item(),g2.item())
    #         S = self.mapping_from_cost(C, n, m)
    #         v = torch.flatten(S)
    #
    #         normalize_factor = 1.0
    #         if self.normalize:
    #             nb_edge1 = (self.A[g1][0:n * n] != torch.zeros(n * n, device=self.device)).int().sum()
    #             nb_edge2 = (self.A[g2][0:m * m] != torch.zeros(m * m, device=self.device)).int().sum()
    #             normalize_factor = nodeInsDel * (n + m) + edgeInsDel * (nb_edge1 + nb_edge2)
    #         c = torch.diag(C)
    #         D = C - torch.eye(C.shape[0], device=self.device) * c
    #         ged[k] = (.5 * v.T @ D @ v + c.T @ v) / normalize_factor
    #
    #     max = torch.max(ged)
    #     min = torch.min(ged)
    #     ged = (ged - min) / (max - min)
    #
    #     return ged

    def from_weighs_to_costs(self):
        relu = torch.nn.ReLU()
        # cn=torch.exp(self.node_weighs)
        # ce=torch.exp(self.edge_weighs)
        # cn=self.node_weighs*self.node_weighs
        # ce=self.edge_weighs*self.edge_weighs
        cn = relu(self.node_weighs)
        ce = relu(self.edge_weighs)

        # total_cost=cn.sum()+ce.sum()
        # cn=cn/total_cost
        # ce=ce/total_cost
        edgeInsDel = ce[-1]

        node_costs = torch.zeros((self.nb_labels, self.nb_labels), device=self.device)
        upper_part = torch.triu_indices(node_costs.shape[0], node_costs.shape[1], offset=1, device=self.device)
        node_costs[upper_part[0], upper_part[1]] = cn[0:-1]
        node_costs = node_costs + node_costs.T

        if self.nb_edge_labels > 1:
            edge_costs = torch.zeros((self.nb_edge_labels, self.nb_edge_labels), device=self.device)
            upper_part = torch.triu_indices(edge_costs.shape[0], edge_costs.shape[1], offset=1, device=self.device)
            edge_costs[upper_part[0], upper_part[1]] = ce[0:-1]
            edge_costs = edge_costs + edge_costs.T
        else:
            edge_costs = torch.zeros(0, device=self.device)

        return node_costs, cn[-1], edge_costs, edgeInsDel

    def build_node_dictionnary(self, GraphList):
        # extraction de tous les labels d'atomes
        node_labels = []
        for G in GraphList:
            for v in nx.nodes(G):
                if not G.nodes[v][self.node_label][0] in node_labels:
                    node_labels.append(G.nodes[v][self.node_label][0])
        node_labels.sort()
        # extraction d'un dictionnaire permettant de numéroter chaque label par un numéro.
        dict = {}
        k = 0
        for label in node_labels:
            dict[label] = k
            k = k + 1
        print(node_labels)
        print(dict, len(dict))

        return dict, max(max([[int(G[e[0]][e[1]]['bond_type']) for e in G.edges()] for G in GraphList]))

    def from_networkx_to_tensor(self, G, dict):
        A = torch.tensor(nx.to_scipy_sparse_matrix(G, dtype=int, weight='bond_type').todense(), dtype=torch.int)
        lab = [dict[G.nodes[v][self.node_label][0]] for v in nx.nodes(G)]

        return (A.view(1, A.shape[0] * A.shape[1]), torch.tensor(lab))

    def construct_cost_matrix(self, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel):
        n = self.card[g1].item()
        m = self.card[g2].item()
        with torch.no_grad():
            A1 = torch.zeros((n + 1, n + 1), dtype=torch.int, device=self.device)
            A1[0:n, 0:n] = self.A[g1][0:n * n].view(n, n)
            A2 = torch.zeros((m + 1, m + 1), dtype=torch.int, device=self.device)
            A2[0:m, 0:m] = self.A[g2][0:m * m].view(m, m)
            A = self.matrix_edgeInsDel(A1, A2)

        # costs: 0 node subs, 1 nodeIns/Del, 2 : edgeSubs, 3 edgeIns/Del

        # C=cost[3]*torch.cat([torch.cat([C12[l][k] for k in range(n+1)],1) for l in range(n+1)])
        # Pas bien sur mais cela semble fonctionner.
        C = edgeInsDel * A
        if self.nb_edge_labels > 1:
            for k in range(self.nb_edge_labels):
                for l in range(self.nb_edge_labels):
                    if k != l:
                        C.add_(self.matrix_edgeSubst(A1, A2, k + 1, l + 1).multiply_(edge_costs[k][l]))

        # C=cost[3]*torch.tensor(np.array([ [  k!=l and A1[k//(m+1),l//(m+1)]^A2[k%(m+1),l%(m+1)] for k in range((n+1)*(m+1))] for l in range((n+1)*(m+1))]),device=self.device)

        l1 = self.labels[g1][0:n]
        l2 = self.labels[g2][0:m]
        D = torch.zeros((n + 1) * (m + 1), device=self.device)
        D[n * (m + 1):] = nodeInsDel
        D[n * (m + 1) + m] = 0
        D[[i * (m + 1) + m for i in range(n)]] = nodeInsDel
        for k in range(n * (m + 1)):
            if k % (m + 1) != m:
                D[k] = node_costs[l1[k // (m + 1)]][l2[k % (m + 1)]]  # self.get_node_costs(l1[k//(m+1)],l2[k%(m+1)])

        # D[[k for k in range(n*(m+1)) if k%(m+1) != m]]=torch.tensor([node_costs[l1[k//(m+1)],l2[k%(m+1)]] for k in range(n*(m+1)) if k%(m+1) != m],device=self.device )
        with torch.no_grad():
            mask = torch.diag(torch.ones_like(D))
        C = mask * torch.diag(D) + (1. - mask) * C

        # C[range(len(C)),range(len(C))]=D

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

    def similarity_from_cost(self, C):
        N = C.shape[0]

        # return (torch.norm(C,p='fro')*torch.eye(N,device=self.device) -C)
        return (C.max() * torch.eye(N, device=self.device) - C)

    def lsape_populate_instance(self, first_graph, second_graph, average_node_cost, average_edge_cost, alpha,
                                lbda):  # ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']
        g, h = Gs[first_graph], Gs[second_graph]
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
        lsape_instance = torch.as_tensor(lsape_instance, device=self.device)
        # print(type(lsape_instance))
        return lsape_instance

    def mapping_from_cost(self, C, n, m):
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], device=self.device) * c
        x0 = svd.eps_assign2(torch.exp(-.5 * c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)

        x = svd.franck_wolfe(x0, D, c, 5, 10, n, m)

        def print_grad(grad):
            if (grad.norm() != 0.0):
                print(grad)

        #        x0.register_hook(print_grad)
        return x

    def mapping_from_similarity(self, C, n, m):
        M = self.similarity_from_cost(C)
        first_ev = compute_major_axis(M)
        # first_ev=self.iterated_power(M,inv=True)
        if (first_ev.sum() < 0):
            first_ev = -first_ev
        S = torch.exp(first_ev.view(n + 1, m + 1))  # enforce the difference, accelerate the convergence.
        S = self.eps_assigment_from_mapping(S)
        return S

    def eps_assigment_from_mapping(self, S):
        ones_n = torch.ones(S.shape[0], device=S.device)
        ones_m = torch.ones(S.shape[1], device=S.device)

        Sk = S
        for i in range(20):
            D = torch.diag(1.0 / (Sk @ ones_m))
            D[D.shape[0] - 1, D.shape[1] - 1] = 1.0
            Sk1 = D @ Sk
            D = torch.diag(1.0 / (ones_n @ Sk1))
            D[D.shape[0] - 1, D.shape[1] - 1] = 1.0
            Sk = Sk1 @ D

        return Sk


if __name__ == "__main__":
    Gs, y = loadDataset('../DeepGED/MAO/dataset.ds')
    for g in Gs:
        compute_extended_labels(g)
    batch_size = 1

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}


    rings_andor_fw = "sans_rings_avec_fw"
    weights = "experts"

    cns = None

    train_graph = torch.load('../pickle_files/'+rings_andor_fw+'/train_graph',map_location=torch.device('cpu'),pickle_module=pkl)
    test_graph = torch.load('../pickle_files/'+rings_andor_fw+'/test_graph',map_location=torch.device('cpu'),pickle_module=pkl)
    train_label = torch.load('../pickle_files/'+rings_andor_fw+'/train_label',map_location=torch.device('cpu'),pickle_module=pkl)
    test_label = torch.load('../pickle_files/'+rings_andor_fw+'/test_label',map_location=torch.device('cpu'),pickle_module=pkl)

    if weights == "learned":
        cns = torch.load('../pickle_files/'+rings_andor_fw+'/nodeSub_min', map_location=torch.device('cpu'), pickle_module=pkl)
        cndl = torch.load('../pickle_files/'+rings_andor_fw+'/nodeInsDel_min', map_location=torch.device('cpu'), pickle_module=pkl)
        cedl = torch.load('../pickle_files/'+rings_andor_fw+'/edgeInsDel_min', map_location=torch.device('cpu'), pickle_module=pkl)
        ces = torch.load('../pickle_files/'+rings_andor_fw+'/edgeSub_min', map_location=torch.device('cpu'), pickle_module=pkl)
    elif weights == "init":
        cns = torch.load('../pickle_files/'+rings_andor_fw+'/nodeSubInit', map_location=torch.device('cpu'), pickle_module=pkl)
        cndl = torch.load('../pickle_files/'+rings_andor_fw+'/nodeInsDelInit', map_location=torch.device('cpu'), pickle_module=pkl)
        cedl= torch.load('../pickle_files/'+rings_andor_fw+'/edgeInsDelInit', map_location=torch.device('cpu'), pickle_module=pkl)
        ces= torch.load('../pickle_files/'+rings_andor_fw+'/edgeSubInit', map_location=torch.device('cpu'), pickle_module=pkl)
    elif weights == "experts":
        cns = torch.load('../pickle_files/'+rings_andor_fw+'/nodeSub_min', map_location=torch.device('cpu'), pickle_module=pkl)
        cndl = torch.load('../pickle_files/'+rings_andor_fw+'/nodeInsDel_min', map_location=torch.device('cpu'), pickle_module=pkl)
        cedl = torch.load('../pickle_files/'+rings_andor_fw+'/edgeInsDel_min', map_location=torch.device('cpu'), pickle_module=pkl)
        ces = torch.load('../pickle_files/'+rings_andor_fw+'/edgeSub_min', map_location=torch.device('cpu'), pickle_module=pkl)
        cns = torch.ones(cns.shape)
        for i in range(cns.shape[0]):
            cns[i, i] = 0

        ces = torch.ones(ces.shape)
        for i in range(ces.shape[0]):
            ces[i, i] = 0

        cndl = torch.tensor(3.0)
        cedl = torch.tensor(3.0)
    else:
        sys.exit("Error : weights are not defined")

    if cndl is not None:
        print(cns)
        model = Evaluator(Gs, normalize=True, node_label='label')
        model.classification(train_graph, test_graph, train_label, test_label, 5, cns, ces, cndl, cedl)
    else:
        sys.exit("Error : weights are not defined")
