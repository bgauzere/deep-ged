'''
TODO : Ne marche pas
TODO : A modifier pour prendre en compte les matrices d'adjacence ?
Possiblement redondant avec main
'''
import sys
import os
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch
from gklearn.utils.graphfiles import loadDataset
import deepged.rings as rings
import deepged.svd as svd
from deepged.data_manager.label_manager import compute_extended_labels, build_node_dictionnary


from deepged.utils import from_networkx_to_tensor
matplotlib.use('TkAgg')


class Evaluator():
    def __init__(self, nb_labels, nb_edge_labels, dict_nodes, rings_andor_fw='sans_rings_sans_fw',
                 normalize=False, node_label="label",
                 verbose=True):

        super(Evaluator, self).__init__()
        self.dict_nodes = dict_nodes
        self.nb_edge_labels = nb_edge_labels
        self.nb_labels = nb_labels

        self.normalize = normalize

        self.normalize = normalize
        self.node_label = node_label
        self.rings_andor_fw = rings_andor_fw

        # TODO : a virer autre part ?
        self.device = torch.device('cpu')

    def init_weights(self, cns, cndl, ces, cedl):
        '''
        Initialize the set of weights. Do not require from_weight_to_cost
        '''
        self.node_weights = cns
        self.edge_weights = ces
        self.node_ins_del = cndl
        self.edge_ins_del = cedl

    def get_weights(self):
        return self.node_weights, self.node_ins_del, self.edge_weights, self.edge_ins_del

    def forward(self, graphs):
        '''
        almost same forward than in model, except from_weight_to_cost is replaced by
        :param graphs: tuple de graphes networkx
        :return: predicted GED between both graphs
        '''

        g1 = graphs[0]
        g2 = graphs[1]

        cns, cndl, ces, cedl = self.get_weights()

        A_g1, labels_1 = from_networkx_to_tensor(g1, self.dict_nodes, self.node_label, self.device)
        A_g2, labels_2 = from_networkx_to_tensor(g2, self.dict_nodes, self.node_label, self.device)

        n = g1.order()
        m = g2.order()

        C = self.construct_cost_matrix(A_g1, A_g2, [n, m], [labels_1, labels_2], cns, ces, cndl, cedl)
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], device=self.device) * c

        if self.rings_andor_fw == 'rings_sans_fw':
            self.ring_g, self.ring_h = rings.build_rings(
                graphs[0], cedl.size()), rings.build_rings(graphs[1], cedl.size())
            c_0 = self.lsape_populate_instance(g1, g2, cns, ces, cndl, cedl)
            # print(C.shape, c_0.shape)
            S = svd.eps_assign2(torch.exp(-.5 * c_0.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        elif self.rings_andor_fw == 'rings_avec_fw':
            self.ring_g, self.ring_h = rings.build_rings(
                graphs[0], cedl.size()), rings.build_rings(graphs[1], cedl.size())
            c_0 = self.lsape_populate_instance(g1, g2, cns, ces, cndl, cedl)
            # print(C.shape, c_0.shape)
            x0 = svd.eps_assign2(
                torch.exp(-.5 * c_0.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
            S = svd.franck_wolfe(x0, D, c, 5, 10, n, m)
        elif self.rings_andor_fw == 'sans_rings_avec_fw':
            x0 = svd.eps_assign2(
                torch.exp(-.5 * c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
            S = svd.franck_wolfe(x0, D, c, 5, 10, n, m)
        elif self.rings_andor_fw == 'sans_rings_sans_fw':
            S = svd.eps_assign2(
                torch.exp(-.5 * c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        else:
            print("Error : rings_andor_fw => value not understood")
            sys.exit()

        normalize_factor = 1.0
        if self.normalize:
            nb_edge1 = (A_g1[0:n * n] != torch.zeros(n * n, device=self.device)).int().sum()
            nb_edge2 = (A_g2[0:m * m] != torch.zeros(m * m, device=self.device)).int().sum()
            normalize_factor = cndl * (n + m) + cedl * (nb_edge1 + nb_edge2)

        v = torch.flatten(S)
        ged = (.5 * v.T @ D @ v + c.T @ v) / normalize_factor
        return ged


    def construct_cost_matrix(self, A_g1, A_g2, card, labels,
                              node_costs, edge_costs, node_ins_del, edge_ins_del):
        '''
        Retourne une matrice carrée de taile (n+1) * (m +1) contenant les couts sur les noeuds et les aretes
        TODO : a analyser, tester et documenter
        '''

        n = card[0]
        m = card[1]

        A1 = torch.zeros((n + 1, n + 1), dtype=torch.int, device=self.device)
        A1[0:n, 0:n] = A_g1[0:n * n].view(n, n)
        A2 = torch.zeros((m + 1, m + 1), dtype=torch.int, device=self.device)
        A2[0:m, 0:m] = A_g2[0:m * m].view(m, m)
        A = self.matrix_edge_ins_del(A1, A2)

        # costs: 0 node subs, 1 nodeIns/Del, 2 : edgeSubs, 3 edgeIns/Del

        # C=cost[3]*torch.cat([torch.cat([C12[l][k] for k in range(n+1)],1) for l in range(n+1)])
        # Pas bien sur mais cela semble fonctionner.
        C = edge_ins_del * A
        if self.nb_edge_labels > 1:
            for k in range(self.nb_edge_labels):
                for l in range(self.nb_edge_labels):
                    if k != l:
                        C.add_(self.matrix_edge_subst(A1, A2, k + 1, l + 1).multiply_(edge_costs[k][l]))

        # C=cost[3]*torch.tensor(np.array([ [  k!=l and A1[k//(m+1),l//(m+1)]^A2[k%(m+1),l%(m+1)] for k in range((n+1)*(m+1))] for l in range((n+1)*(m+1))]),device=self.device)

        l1 = labels[0][0:n]
        l2 = labels[1][0:m]
        D = torch.zeros((n + 1) * (m + 1), device=self.device)
        D[n * (m + 1):] = node_ins_del
        D[n * (m + 1) + m] = 0
        D[[i * (m + 1) + m for i in range(n)]] = node_ins_del
        for k in range(n * (m + 1)):
            if k % (m + 1) != m:
                # self.get_node_costs(l1[k//(m+1)],l2[k%(m+1)])
                D[k] = node_costs[l1[k // (m + 1)]][l2[k % (m + 1)]]

                # D[[k for k in range(n*(m+1)) if k%(m+1) != m]]=torch.tensor([node_costs[l1[k//(m+1)],l2[k%(m+1)]] for k in range(n*(m+1)) if k%(m+1) != m],device=self.device )

        mask = torch.diag(torch.ones_like(D))
        C = mask * torch.diag(D) + (1. - mask) * C

        # C[range(len(C)),range(len(C))]=D

        return C

    def matrix_edge_ins_del(self, A1, A2):
        '''
        Doc TODO
        '''
        Abin1 = (A1 != torch.zeros(
            (A1.shape[0], A1.shape[1]), device=self.device))
        Abin2 = (A2 != torch.zeros(
            (A2.shape[0], A2.shape[1]), device=self.device))
        C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
        C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
        C12 = torch.logical_or(C1, C2).int()

        return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)

    def matrix_edge_subst(self, A1, A2, lab1, lab2):
        '''
        Doc TODO
        '''
        Abin1 = (
                A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]), device=self.device)).int()
        Abin2 = (
                A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]), device=self.device)).int()
        C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

        return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1).float()

    def similarity_from_cost(self, C):
        '''
        Transforme une matrice de couts C en une matrice de similarité
        Retoune la matrice de similarité
        TODO :
         * a verifier
         * a mettre dans un autre fichier ?
        '''
        N = C.shape[0]

        # return (torch.norm(C,p='fro')*torch.eye(N,device=self.device) -C)
        return (C.max() * torch.eye(N, device=self.device) - C)

    # TODO :  La fonction plus haut ne semble pas fonctionner, voici l'ancienne version (A discuter )
    def lsape_populate_instance(self, first_graph, second_graph, average_node_cost, average_edge_cost, alpha,
                                lbda):

        self.average_cost = [average_node_cost, average_edge_cost]
        self.first_graph, self.second_graph = first_graph, second_graph

        node_costs, nodeInsDel, edge_costs, edgeInsDel = self.from_weights_to_costs()

        lsape_instance = [[0 for _ in range(len(first_graph) + 1)]
                          for __ in range(len(second_graph) + 1)]
        for g_node_index in range(len(first_graph) + 1):
            for h_node_index in range(len(second_graph) + 1):
                lsape_instance[h_node_index][g_node_index] = rings.compute_ring_distance(self.ring_g, self.ring_h,
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


    def classification(self, Gs, train_data, test_data, train_label, test_label, k):
        """
        Compute a matrix of distances of each pair of graph in train_data, and use it to learn a KNN that will be used
        over the matrix distance between test_data and train_data
        :param Gs: the list of networkx graphs
        :param train_data: index of data in train
        :param test_data: index of data in test
        :param train_label: true label in train
        :param test_label: true label test
        :param k: value of k neighborhood
        :return: both distance matrices
        """
        n_graph_test = len(test_data)
        n_graph_train = len(train_data)
        ged_matrix_train = np.zeros((n_graph_train, n_graph_train))
        ged_matrix_test = np.zeros((n_graph_test, n_graph_train))

        for i in tqdm(range(len(train_data))):
            g1 = Gs[train_data[i]]
            for j in range(len(train_data)):
                g2 = Gs[train_data[j]]
                ged = self.forward((g1,g2))
                ged_matrix_train[i, j] = ged

        for i in tqdm(range(len(test_data))):
            g1 = Gs[test_data[i]]
            for j in range(len(train_data)):
                g2 = Gs[train_data[j]]
                ged = self.forward((g1,g2))
                ged_matrix_test[i, j] = ged

        # Definition of the KNN classifier. precomputed stands for "precomputed distances", which we did by computing GED
        classifier = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        classifier.fit(ged_matrix_train, train_label)
        y_pred_test = classifier.predict(ged_matrix_test)

        print("Test Results : ")
        print(classification_report(test_label, y_pred_test))
        print(confusion_matrix(test_label, y_pred_test))

        # plt.subplot(121)
        # plt.imshow(ged_matrix_train)
        # plt.subplot(122)
        # plt.imshow(ged_matrix_test)
        # plt.show()

        return ged_matrix_train, ged_matrix_test


if __name__ == "__main__":
    dataset_path = os.getenv("MAO_DATASET_PATH")
    Gs, y = loadDataset(dataset_path)
    for g in Gs:
        compute_extended_labels(g, label_node="label")
    batch_size = 1

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    rings_andor_fw = "sans_rings_avec_fw"
    which_weights = ["init", "learned", "experts"]

    cns = None

    train_graph = torch.load('./pickle_files/'+rings_andor_fw+'/train_graph',
                             map_location=torch.device('cpu'), pickle_module=pkl)
    test_graph = torch.load('./pickle_files/'+rings_andor_fw+'/test_graph',
                            map_location=torch.device('cpu'), pickle_module=pkl)
    train_label = torch.load('./pickle_files/'+rings_andor_fw+'/train_label',
                             map_location=torch.device('cpu'), pickle_module=pkl)
    test_label = torch.load('./pickle_files/'+rings_andor_fw+'/test_label',
                            map_location=torch.device('cpu'), pickle_module=pkl)

    ged_train = [None]*len(which_weights)
    ged_test = [None]*len(which_weights)
    for i in range(len(which_weights)):
        weights = which_weights[i]

        if weights == "learned":
            cns = torch.load('./pickle_files/'+rings_andor_fw+'/nodeSub_min',
                             map_location=torch.device('cpu'), pickle_module=pkl)
            cndl = torch.load('./pickle_files/'+rings_andor_fw+'/nodeInsDel_min',
                              map_location=torch.device('cpu'), pickle_module=pkl)
            cedl = torch.load('./pickle_files/'+rings_andor_fw+'/edgeInsDel_min',
                              map_location=torch.device('cpu'), pickle_module=pkl)
            ces = torch.load('./pickle_files/'+rings_andor_fw+'/edgeSub_min',
                             map_location=torch.device('cpu'), pickle_module=pkl)
        elif weights == "init":
            cns = torch.load('./pickle_files/'+rings_andor_fw+'/nodeSubInit',
                             map_location=torch.device('cpu'), pickle_module=pkl)
            cndl = torch.load('./pickle_files/'+rings_andor_fw+'/nodeInsDelInit',
                              map_location=torch.device('cpu'), pickle_module=pkl)
            cedl = torch.load('./pickle_files/'+rings_andor_fw+'/edgeInsDelInit',
                              map_location=torch.device('cpu'), pickle_module=pkl)
            ces = torch.load('./pickle_files/'+rings_andor_fw+'/edgeSubInit',
                             map_location=torch.device('cpu'), pickle_module=pkl)
        elif weights == "experts":
            cns = torch.load('./pickle_files/'+rings_andor_fw+'/nodeSub_min',
                             map_location=torch.device('cpu'), pickle_module=pkl)
            cndl = torch.load('./pickle_files/'+rings_andor_fw+'/nodeInsDel_min',
                              map_location=torch.device('cpu'), pickle_module=pkl)
            cedl = torch.load('./pickle_files/'+rings_andor_fw+'/edgeInsDel_min',
                              map_location=torch.device('cpu'), pickle_module=pkl)
            ces = torch.load('./pickle_files/'+rings_andor_fw+'/edgeSub_min',
                             map_location=torch.device('cpu'), pickle_module=pkl)

            # couts constants pour tests
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

            node_label = "extended_label"  # -> parametre
            edge_label = "bond_type"  # parametre
            node_labels, nb_edge_labels = build_node_dictionnary(
                Gs, node_label, edge_label)
            nb_labels = len(node_labels)

            model = Evaluator(nb_labels, nb_edge_labels, node_labels, rings_andor_fw, normalize=True,
                             node_label=node_label)

            model.init_weights(cns, cndl, ces, cedl)
            ged_train[i], ged_test[i] = model.classification(Gs, train_graph, test_graph,
                                                       train_label, test_label, 5)

        else:
            sys.exit("Error : weights are not defined")

    for i in range(len(ged_train)):
        min = 0
        max = np.max(ged_train)
        plt.subplot(len(ged_train), 2, (i*2)+1)
        plt.imshow(ged_train[i], vmin=0, vmax=max)
        plt.title(which_weights[i] + " costs : train")
        plt.subplot(len(ged_train), 2, (i*2)+2)
        plt.imshow(ged_test[i], vmin=0, vmax=max)
        plt.title(which_weights[i] + " costs : test")

    plt.show()