'''
Implemente la classe GedLayer qui permet d'opitimiser les couts de la ged pour fitter une propriété donnée
TODO :
 * Faire des classes filles pour implemeter les différentes stratégies
 * Structure pour reunir les couts ? 
'''
from torch import nn
import numpy as np
import torch
import sys

import deepged.rings
import deepged.svd as svd
from deepged.svd import iterated_power as compute_major_axis


class GedLayer(nn.Module):
    def __init__(self,  nb_labels, nb_edge_labels, rings_andor_fw='sans_rings_sans_fw',
                 normalize=False, node_label="label",
                 verbose=True):

        super(GedLayer, self).__init__()

        self.nb_edge_labels = nb_edge_labels
        self.nb_labels = nb_labels

        self.normalize = normalize

        self.normalize = normalize
        self.node_label = node_label
        self.rings_andor_fw = rings_andor_fw

        # TODO : a virer autre part ?
        self.device = torch.device('cpu')
        self._init_weights()

    def _init_weights(self):
        """
        Initialise les poids pour les paires de labels de noeuds et d'edges
        """
        # Partie tri sup d'une matrice de nb_labels par nb_labels
        nb_node_pair_label = int(self.nb_labels * (self.nb_labels - 1) / 2.0)
        nb_edge_pair_label = int(
            self.nb_edge_labels * (self.nb_edge_labels - 1) / 2)

        nweights = (1e-1) * (1.1 *
                             np.random.rand(nb_node_pair_label + 1))
        nweights[-1] = .3

        eweights = (1e-1) * (1.1 *
                             np.random.rand(nb_edge_pair_label + 1))
        eweights[-1] = .2

        self.node_weights = nn.Parameter(torch.tensor(
            nweights, requires_grad=True, dtype=torch.float,
            device=self.device))
        self.edge_weights = nn.Parameter(torch.tensor(
            eweights, requires_grad=True, dtype=torch.float,
            device=self.device))

    def forward(self, graph, adjacenceMatrix, graphCard, labels):
        '''
        graph : tuple de deux graphes
        adjacenceMatrix : tuple de deux matrices d'adjacences
        graphCard : tuple des deux tailles des graphes (TODO : a extraire de g1 et g2)
        labels : ? (TODO)
        TODO : il me semble qu'il suffirait de passer juste les deux networkx graphes
        '''

        g1 = graph[0]
        g2 = graph[1]

        A_g1 = adjacenceMatrix[0]
        A_g2 = adjacenceMatrix[1]

        cns, cndl, ces, cedl = self.from_weights_to_costs()

        n = graphCard[0]
        m = graphCard[1]

        C = self.construct_cost_matrix(
            A_g1, A_g2, graphCard, labels, cns, ces, cndl, cedl)
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], device=self.device) * c

        if self.rings_andor_fw == 'rings_sans_fw':
            self.ring_g, self.ring_h = rings.build_rings(
                graph[0], cedl.size()), rings.build_rings(graph[1], cedl.size())
            c_0 = self.lsape_populate_instance(g1, g2, cns, ces, cndl, cedl)
            print(C.shape, c_0.shape)
            S = svd.eps_assign2(
                torch.exp(-.5 * c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
        elif self.rings_andor_fw == 'rings_avec_fw':
            self.ring_g, self.ring_h = rings.build_rings(
                graph[0], cedl.size()), rings.build_rings(graph[1], cedl.size())
            c_0 = self.lsape_populate_instance(g1, g2, cns, ces, cndl, cedl)
            print(C.shape, c_0.shape)
            x0 = svd.eps_assign2(
                torch.exp(-.5 * c.view(n + 1, m + 1)), 10).view((n + 1) * (m + 1), 1)
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
            nb_edge1 = (A_g1[0:n * n] != torch.zeros(n *
                        n, device=self.device)).int().sum()
            nb_edge2 = (A_g2[0:m * m] != torch.zeros(m *
                        m, device=self.device)).int().sum()
            normalize_factor = cndl * (n + m) + cedl * (nb_edge1 + nb_edge2)

        v = torch.flatten(S)
        ged = (.5 * v.T @ D @ v + c.T @ v)/normalize_factor
        return ged

    def from_weights_to_costs(self):
        """
        Transforme les poids en couts de ged en les rendant poisitifs
        un seul cout de suppresion/insertion.
        """
        # We apply the ReLU (rectified linear unit) function element-wise
        relu = torch.nn.ReLU()
        cn = relu(self.node_weights)
        ce = relu(self.edge_weights)
        edge_ins_del = ce[-1]
        node_ins_del = cn[-1]

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

        return node_costs, node_ins_del, edge_costs, edge_ins_del

    def construct_cost_matrix(self, A_g1, A_g2, card, labels,
                              node_costs, edge_costs, node_ins_del, edge_ins_del):
        '''
        Retourne une matrice carrée de taile (n+1) * (m +1) contenant les couts sur les noeuds et les aretes
        TODO : a analyser, tester et documenter
        '''

        n = card[0].item()
        m = card[1].item()
        with torch.no_grad():
            A1 = torch.zeros((n + 1, n + 1), dtype=torch.int,
                             device=self.device)
            A1[0:n, 0:n] = A_g1[0:n * n].view(n, n)
            A2 = torch.zeros((m + 1, m + 1), dtype=torch.int,
                             device=self.device)
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
                        C.add_(self.matrix_edge_subst(A1, A2, k + 1,
                                                      l + 1).multiply_(edge_costs[k][l]))

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
        with torch.no_grad():
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

    def lsape_populate_instance(self, first_graph, second_graph, average_node_cost, average_edge_cost, alpha, lbda):
        '''
        Calcule les couts entre noeuds par les rings.
        TODO : nom à changer ?
        first et second graph sont des graphes ou des index ?
        a mettre dans un autre fichier
        '''

        # first_ev=self.iterated_power(M,inv=True)
        if (first_ev.sum() < 0):
            first_ev = -first_ev
        # enforce the difference, accelerate the convergence.
        S = torch.exp(first_ev.view(n + 1, m + 1))
        S = self.eps_assigment_from_mapping(S)
        return S

    def eps_assigment_from_mapping(self, S):
        '''
        Calcul un mapping à partir de S
        QUESTION : S similarité ou mapping ?
        TODO : fonction du meme nom dans svd.py
        '''
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