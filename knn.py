from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import svd
import rings
from svd import iterated_power as compute_major_axis
from gklearn.utils.graphfiles import loadDataset
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')


def build_node_dictionnary(GraphList):
    # extraction de tous les labels d'atomes
    node_label = 'label'
    node_labels = []
    for G in Gs:
        for v in nx.nodes(G):
            if not G.nodes[v][node_label][0] in node_labels:
                node_labels.append(G.nodes[v][node_label][0])
    node_labels.sort()
    # extraction d'un dictionnaire permettant de numéroter chaque label par un numéro.
    dict = {}
    k = 0
    for label in node_labels:
        dict[label] = k
        k = k+1
    #print(node_labels)
    #print(dict, len(dict))

    return dict,max(max([[int(G[e[0]][e[1]]['bond_type']) for e in G.edges()] for G in GraphList]))


def from_networkx_to_tensor(G,dict):    
    A_g=torch.tensor(nx.to_scipy_sparse_matrix(G,dtype=int,weight='bond_type').todense(),dtype=torch.int)        
    lab=[dict[G.nodes[v]['label'][0]] for v in nx.nodes(G)]

    return A_g.view(1,A_g.shape[0]*A_g.shape[1]),torch.tensor(lab)

def init_dataset(Gs,dict):
    for k in range(len(Gs)):
        A_k,l=from_networkx_to_tensor(Gs[k],dict)             
        A[k,0:A_k.shape[1]]=A_k[0]
        labels[k,0:l.shape[0]]=l

def construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel):
    n = card[g1].item()
    m = card[g2].item()

    A1 = torch.zeros((n+1, n+1), dtype=torch.int, device=device)
    A1[0:n, 0:n] = A[g1][0:n*n].view(n, n)
    A2 = torch.zeros((m+1, m+1), dtype=torch.int, device=device)
    A2[0:m, 0:m] = A[g2][0:m*m].view(m, m)
   
    C = edgeInsDel*matrix_edgeInsDel(A1, A2)
    if nb_edge_labels > 1:
        for k in range(nb_edge_labels):
            for l in range(nb_edge_labels):
                if k != l:
                    C = C+edge_costs[k][l]*matrix_edgeSubst(A1, A2, k+1, l+1)

    l1 = labels[g1][0:n]
    l2 = labels[g2][0:m]
    D = torch.zeros((n+1)*(m+1), device=device)
    D[n*(m+1):] = nodeInsDel
    D[n*(m+1)+m] = 0
    D[[i*(m+1)+m for i in range(n)]] = nodeInsDel
    D[[k for k in range(n*(m+1)) if k % (m+1) != m]] = torch.tensor([node_costs[l1[k//(
        m+1)], l2[k % (m+1)]] for k in range(n*(m+1)) if k % (m+1) != m], device=device)
    mask = torch.diag(torch.ones_like(D))
    C = mask*torch.diag(D) + (1. - mask)*C

    return C


def matrix_edgeInsDel(A1, A2):
    Abin1 = (A1 != torch.zeros((A1.shape[0], A1.shape[1]), device=device))
    Abin2 = (A2 != torch.zeros((A2.shape[0], A2.shape[1]), device=device))
    C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
    C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
    C12 = torch.logical_or(C1, C2).int()

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)


def matrix_edgeSubst(A1, A2, lab1, lab2):
    Abin1 = (A1 == lab1*torch.ones((A1.shape[0], A1.shape[1]), device=device)).int()
    Abin2 = (A2 == lab2*torch.ones((A2.shape[0], A2.shape[1]), device=device)).int()
    C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1)


def lsape_populate_instance(first_graph, second_graph, average_node_cost, average_edge_cost, alpha, lbda, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h):
    # ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']

    g, h = Gs[first_graph], Gs[second_graph]

    lsape_instance = [[0 for _ in range(len(g) + 1)]
                      for __ in range(len(h) + 1)]
    for g_node_index in range(len(g) + 1):
        for h_node_index in range(len(h) + 1):
            lsape_instance[h_node_index][g_node_index] = rings.compute_ring_distance(
                g, h, ring_g, ring_h, g_node_index, h_node_index, alpha, lbda, node_costs, nodeInsDel, edge_costs, edgeInsDel, first_graph, second_graph)
    for i in lsape_instance:
        i = torch.as_tensor(i)
    lsape_instance = torch.as_tensor(lsape_instance)
    return lsape_instance


def mapping_from_cost_sans_FW(n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, ring_g, ring_h):
    c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                  edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c_0), 10).view((n+1)*(m+1), 1)
    return x0


def mapping_from_cost(C, n, m):
    c = torch.diag(C)
    D = C-torch.eye(C.shape[0], device=device)*c
    S = torch.exp(-c.view(n+1, m+1))
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
    return svd.franck_wolfe(x0, D, c, 5, 15, n, m)

def primary_ged(k,inputs,node_costs, edge_costs, nodeInsDel, edgeInsDel):
    g1 = inputs[k][0]
    g2 = inputs[k][1]
    n = card[g1]
    m = card[g2]

    #ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())

    C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)

    # S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)
    S = mapping_from_cost(C, n, m)
    v = torch.flatten(S)

    normalize_factor = 1.0
    if normalize:
        nb_edge1 = (A[g1][0:n*n] != torch.zeros(n *
                                                n, device=device)).int().sum()
        nb_edge2 = (A[g2][0:m*m] != torch.zeros(m *
                                                m, device=device)).int().sum()
        normalize_factor = nodeInsDel*(n+m)+edgeInsDel*(nb_edge1+nb_edge2)
    c = torch.diag(C)
    D = C-torch.eye(C.shape[0], device=device)*c
    ged_k = (.5*v.t()@D@v+c.T@v)/normalize_factor

    return ged_k
    


def gedd(data, node_costs, edge_costs, nodeInsDel, edgeInsDel):
    inputs = data.to(device)
    ged = torch.zeros(len(inputs)).to(device)

    for k in range(len(inputs)):
        ged[k] = primary_ged(k,inputs,node_costs, edge_costs, nodeInsDel, edgeInsDel)

    return ged

def ged_to_pkl(data, node_costs, edge_costs, nodeInsDel, edgeInsDel):
    ged_res = gedd(data, node_costs, edge_costs, nodeInsDel, edgeInsDel) 
    print('ged : ', ged_res)
    print('ged.size() : ', ged_res.size())

    torch.save(ged_res,'ged.pt')
    with open("ged.pkl", "wb") as fout:
        pkl.dump(ged_res, fout, protocol=pkl.HIGHEST_PROTOCOL)

    ged=ged_res.detach().numpy()   
    plt.plot(ged)
    plt.title('ged for train set')
    plt.show()
    return ged_res

def ged_from_pkl(filename):
    #gedx = torch.load(filename)
    with open(filename, "rb") as fout:
        ged_pkl = pkl.load(fout)
    new_ged_pkl = ged_pkl.detach().numpy()
    return new_ged_pkl

def train_data(train_size,class1, class2):
    nb_elt = int(train_size*(train_size-1)/2)
    data = torch.empty((nb_elt, 2), dtype=torch.int)
    train_graphs = torch.cat((class1, class2), 0)
    print('train_graphs : ', train_graphs)
    #yt = torch.ones(nb_elt)
    couples = torch.triu_indices(train_size, train_size, offset=1)
    data[0:nb_elt, 0] = train_graphs[couples[0, :]]
    data[0:nb_elt, 1] = train_graphs[couples[1, :]]
    #print('couples : ', couples[0, :], couples.shape)
    return data,train_graphs

def train_test_data(train_size, test_size, class1, class2):
    nb_elt = int(train_size*(train_size-1)/2)
    data = torch.empty((nb_elt, 2), dtype=torch.int)

    if train_size % 2 == 0:
        nb_class1=int(train_size/2)
        nb_class2=int(train_size/2)
    else:
        nb_class1=int(train_size/2)+1
        nb_class2=int(train_size/2)
        
    if test_size % 2 == 0:
        nb_class1_test=int(test_size/2)
        nb_class2_test=int(test_size/2)
    else:
        nb_class1_test=int(test_size/2)+1
        nb_class2_test=int(test_size/2)
        
    print((torch.abs(10000*torch.randn(nb_class1)).int()%class1.size()[0]).long())
    random_class1=class1[(torch.abs(10000*torch.randn(nb_class1)).int()%class1.size()[0]).long()]
    random_class2=class2[(torch.abs(10000*torch.randn(nb_class2)).int()%class2.size()[0]).long()]

    random_class1_test=class1[(torch.abs(10000*torch.randn(nb_class1_test)).int()%class1.size()[0]).long()]
    random_class2_test=class2[(torch.abs(10000*torch.randn(nb_class2_test)).int()%class2.size()[0]).long()]

    train_graphs=torch.cat((random_class1,random_class2),0)
    test_graphs=torch.cat((random_class1_test,random_class2_test),0)

    couples = torch.triu_indices(train_size, train_size, offset=1)
    data[0:nb_elt, 0] = train_graphs[couples[0, :]]
    data[0:nb_elt, 1] = train_graphs[couples[1, :]]

    '''
    couples_test=torch.triu_indices(test_size,train_size,offset=1)
    nb_elt_test=int((test_size-1)*(train_size))
    data_test=torch.empty((nb_elt_test,2),dtype=torch.int)
    data_test[0:nb_elt_test,0]=test_graphs[couples_test[0]]
    data_test[0:nb_elt_test,1]=test_graphs[couples_test[1]]
    '''
    return data, train_graphs, test_graphs


def knn(train_graphs, train_size, class1, class2, ged_pkl):
    triu_indices = torch.triu_indices(row=train_size, col=train_size, offset=1)
    D = torch.zeros((train_size, train_size))
    D[triu_indices[0, :], triu_indices[1, :]] = ged_pkl
    D = D+D.t()
    plt.matshow(D)
    plt.colorbar()
    plt.title('D ')
    plt.show()

    classifier = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    print(D.shape)
    # plt.plot(ytrain)
    # plt.show()
    #print('ged size : ', ged_pkl.size())


    ytrain = []
    for g in train_graphs:
        ytrain.append(y[g])
    print('ytrain : ', ytrain)
    print('train_size = ', train_size)
    
    # plt.plot(ytrain)
    # plt.show()
    #print('ged size : ', ged_pkl.size())

    classifier.fit(D, ytrain)  # train_size*train_size
    y_pred = classifier.predict(D)  # test_size*train_size
    print(y_pred)
    #print(confusion_matrix(y_test, y_pred))

    print(classification_report(ytrain, y_pred))
    #print(classification_report(data_test, y_pred))

    plt.plot(D[12, :], label='D[12, :]')
    plt.plot(ytrain, label='ytrain')
    plt.title('ytrain and D[12, :]')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Gs, y = loadDataset('../DeepGED/MAO/dataset.ds')
    card = torch.tensor([G.order() for G in Gs]).to(device)
    card_max = card.max()
    A = torch.empty((len(Gs), card_max*card_max), dtype=torch.int, device=device)
    labels = torch.empty((len(Gs), card_max), dtype=torch.int, device=device)
    normalize = False
    dict,nb_edge_labels = build_node_dictionnary(Gs)
    init_dataset(Gs,dict)
    nb = len(Gs)
    class1 = torch.tensor([k for k in range(len(y)) if y[k] == 1])
    class2 = torch.tensor([k for k in range(len(y)) if y[k] == 0])

    node_costs = torch.tensor([[0.0000e+00, 2.1417e-04, 1.4265e-04, 1.4876e-04, 4.2080e-05, 6.0872e-04,
                            2.0225e-04, 1.5615e-04, 8.1159e-04, 5.7478e-04, 7.1602e-05, 8.1477e-05,
                            3.7777e-04, 3.7794e-04, 6.6344e-04, 1.2428e-04, 7.8725e-04, 1.8384e-04,
                            4.7986e-05],
                           [2.1417e-04, 0.0000e+00, 7.1073e-04, 6.5603e-04, 9.5320e-04, 5.8584e-05,
                            2.6930e-04, 6.3340e-05, 1.7450e-04, 2.6896e-04, 8.0216e-04, 2.3394e-05,
                            6.5480e-04, 1.4686e-04, 9.4018e-04, 2.1455e-04, 1.2443e-04, 2.6400e-04,
                            3.5339e-04],
                           [1.4265e-04, 7.1073e-04, 0.0000e+00, 1.9181e-04, 8.8574e-04, 9.4753e-04,
                            3.5654e-04, 3.9683e-05, 4.3312e-05, 2.8949e-05, 8.1884e-05, 2.5145e-04,
                            4.8013e-05, 1.3623e-04, 7.5691e-05, 5.8886e-04, 3.0807e-05, 1.0630e-05,
                            3.8683e-04],
                           [1.4876e-04, 6.5603e-04, 1.9181e-04, 0.0000e+00, 9.7899e-04, 1.2052e-04,
                            2.3864e-04, 9.4093e-05, 6.6583e-05, 4.3135e-04, 4.2605e-05, 4.9183e-04,
                            5.5029e-04, 6.1907e-04, 7.8500e-04, 4.4327e-04, 8.1684e-05, 6.2143e-04,
                            6.8563e-05],
                           [4.2080e-05, 9.5320e-04, 8.8574e-04, 9.7899e-04, 0.0000e+00, 1.5868e-04,
                            3.2499e-04, 8.7428e-04, 1.6966e-04, 6.6663e-04, 6.7565e-05, 1.0199e-03,
                            5.3650e-05, 3.9485e-04, 1.3087e-04, 9.1877e-04, 2.3080e-04, 8.0990e-04,
                            1.3594e-04],
                           [6.0872e-04, 5.8584e-05, 9.4753e-04, 1.2052e-04, 1.5868e-04, 0.0000e+00,
                            4.1385e-04, 1.2610e-04, 4.3998e-05, 9.6344e-04, 1.0217e-03, 2.6727e-05,
                            9.7008e-05, 8.4777e-04, 4.2021e-05, 2.6550e-05, 1.6204e-04, 6.8144e-04,
                            4.2484e-05],
                           [2.0225e-04, 2.6930e-04, 3.5654e-04, 2.3864e-04, 3.2499e-04, 4.1385e-04,
                            0.0000e+00, 3.9279e-04, 6.7772e-04, 2.6301e-04, 3.7975e-04, 1.4898e-04,
                            3.3335e-04, 3.9055e-04, 5.3765e-05, 1.0666e-04, 8.4510e-04, 4.5808e-05,
                            1.7649e-04],
                           [1.5615e-04, 6.3340e-05, 3.9683e-05, 9.4093e-05, 8.7428e-04, 1.2610e-04,
                            3.9279e-04, 0.0000e+00, 6.7877e-04, 7.7846e-04, 6.6429e-05, 6.9824e-05,
                            2.2679e-04, 4.5791e-04, 2.0854e-04, 5.6237e-04, 6.1879e-05, 1.2327e-04,
                            5.5628e-05],
                           [8.1159e-04, 1.7450e-04, 4.3312e-05, 6.6583e-05, 1.6966e-04, 4.3998e-05,
                            6.7772e-04, 6.7877e-04, 0.0000e+00, 8.4517e-05, 4.0259e-05, 5.7845e-04,
                            1.4896e-04, 5.0939e-05, 1.9496e-04, 9.9730e-04, 4.2647e-05, 1.4748e-04,
                            8.9371e-05],
                           [5.7478e-04, 2.6896e-04, 2.8949e-05, 4.3135e-04, 6.6663e-04, 9.6344e-04,
                            2.6301e-04, 7.7846e-04, 8.4517e-05, 0.0000e+00, 1.7674e-04, 1.6068e-04,
                            2.2346e-05, 8.3865e-05, 5.9050e-04, 2.5126e-04, 7.1116e-05, 2.1961e-04,
                            3.7895e-04],
                           [7.1602e-05, 8.0216e-04, 8.1884e-05, 4.2605e-05, 6.7565e-05, 1.0217e-03,
                            3.7975e-04, 6.6429e-05, 4.0259e-05, 1.7674e-04, 0.0000e+00, 2.8413e-04,
                            6.3511e-04, 8.8566e-04, 2.1421e-04, 6.1044e-04, 1.8297e-05, 3.7086e-04,
                            7.5775e-04],
                           [8.1477e-05, 2.3394e-05, 2.5145e-04, 4.9183e-04, 1.0199e-03, 2.6727e-05,
                            1.4898e-04, 6.9824e-05, 5.7845e-04, 1.6068e-04, 2.8413e-04, 0.0000e+00,
                            3.2569e-04, 1.1300e-04, 2.2427e-04, 2.0428e-04, 8.5494e-04, 4.8170e-04,
                            1.2983e-05],
                           [3.7777e-04, 6.5480e-04, 4.8013e-05, 5.5029e-04, 5.3650e-05, 9.7008e-05,
                            3.3335e-04, 2.2679e-04, 1.4896e-04, 2.2346e-05, 6.3511e-04, 3.2569e-04,
                            0.0000e+00, 7.2481e-04, 2.5192e-04, 9.8129e-04, 2.5114e-04, 7.5161e-04,
                            7.1826e-04],
                           [3.7794e-04, 1.4686e-04, 1.3623e-04, 6.1907e-04, 3.9485e-04, 8.4777e-04,
                            3.9055e-04, 4.5791e-04, 5.0939e-05, 8.3865e-05, 8.8566e-04, 1.1300e-04,
                            7.2481e-04, 0.0000e+00, 7.2420e-04, 3.2423e-04, 8.2467e-04, 4.1701e-05,
                            1.5990e-04],
                           [6.6344e-04, 9.4018e-04, 7.5691e-05, 7.8500e-04, 1.3087e-04, 4.2021e-05,
                            5.3765e-05, 2.0854e-04, 1.9496e-04, 5.9050e-04, 2.1421e-04, 2.2427e-04,
                            2.5192e-04, 7.2420e-04, 0.0000e+00, 5.9012e-05, 1.3361e-04, 1.4604e-04,
                            4.0830e-05],
                           [1.2428e-04, 2.1455e-04, 5.8886e-04, 4.4327e-04, 9.1877e-04, 2.6550e-05,
                            1.0666e-04, 5.6237e-04, 9.9730e-04, 2.5126e-04, 6.1044e-04, 2.0428e-04,
                            9.8129e-04, 3.2423e-04, 5.9012e-05, 0.0000e+00, 1.3253e-04, 7.0482e-04,
                            3.0487e-04],
                           [7.8725e-04, 1.2443e-04, 3.0807e-05, 8.1684e-05, 2.3080e-04, 1.6204e-04,
                            8.4510e-04, 6.1879e-05, 4.2647e-05, 7.1116e-05, 1.8297e-05, 8.5494e-04,
                            2.5114e-04, 8.2467e-04, 1.3361e-04, 1.3253e-04, 0.0000e+00, 9.0885e-04,
                            8.9417e-04],
                           [1.8384e-04, 2.6400e-04, 1.0630e-05, 6.2143e-04, 8.0990e-04, 6.8144e-04,
                            4.5808e-05, 1.2327e-04, 1.4748e-04, 2.1961e-04, 3.7086e-04, 4.8170e-04,
                            7.5161e-04, 4.1701e-05, 1.4604e-04, 7.0482e-04, 9.0885e-04, 0.0000e+00,
                            1.4390e-04],
                           [4.7986e-05, 3.5339e-04, 3.8683e-04, 6.8563e-05, 1.3594e-04, 4.2484e-05,
                            1.7649e-04, 5.5628e-05, 8.9371e-05, 3.7895e-04, 7.5775e-04, 1.2983e-05,
                            7.1826e-04, 1.5990e-04, 4.0830e-05, 3.0487e-04, 8.9417e-04, 1.4390e-04,
                            0.0000e+00]])


    nodeInsDel = torch.tensor(0.006543666590005159)
    edgeInsDel = torch.tensor(0.19699111580848694)
    edge_costs = torch.tensor([[0.0000e+00, 0.15198137, 0.16227092],
                            [0.15198137, 0.0000e+00, 0.15802163],
                            [0.16227092, 0.15802163, 0.0000e+00]])
    

    g1=0
    g2=1

    C=construct_cost_matrix(g1,g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
    #print('C :',C)
    plt.matshow(C)
    plt.title('Cost matrix for g1=0 and g2=1')
    plt.show()


    node_costs_2 = torch.ones(node_costs.shape)
    for i in range(node_costs_2.shape[0]):
        node_costs_2[i,i] = 0

    edge_costs_2 = torch.ones(edge_costs.shape)
    for i in range(edge_costs_2.shape[0]):
        edge_costs_2[i,i] = 0

    print('edge_costs_2 :  ',edge_costs_2)

    nodeInsDel_2 = torch.tensor(3.0)
    edgeInsDel_2 = torch.tensor(3.0)

    '''
    train_size=61   #90_10
    test_size=7
    '''

    train_size = 68
    data,train_graphs = train_data(train_size, class1, class2)

    ged_pkl=ged_to_pkl(data, node_costs_2, edge_costs_2, nodeInsDel_2, edgeInsDel_2)
    #print(ged)
    knn(train_graphs, train_size, class1, class2, ged_pkl)


    filename='ged.pkl'
    new_ged_pkl = ged_from_pkl(filename)

    print()