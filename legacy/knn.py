from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os
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
    for G in Gs:
        for v in nx.nodes(G):
            if not G.nodes[v][node_label][0] in node_labels:
                node_labels.append(G.nodes[v][node_label][0])
    node_labels.sort()
    # Extraction of a dictionary allowing to number each label by a number.
    dict = {}
    k = 0
    for label in node_labels:
        dict[label] = k
        k = k+1

    return dict, max(max([[int(G[e[0]][e[1]]['bond_type']) for e in G.edges()] for G in GraphList]))

# Transforming a networkx to a torch tensor


def from_networkx_to_tensor(G, dict):
    A_g = torch.tensor(nx.to_scipy_sparse_matrix(
        G, dtype=int, weight='bond_type').todense(), dtype=torch.int)
    lab = [dict[G.nodes[v]['label'][0]] for v in nx.nodes(G)]

    return A_g.view(1, A_g.shape[0]*A_g.shape[1]), torch.tensor(lab)


def init_dataset(Gs, dict):
    for k in range(len(Gs)):
        A_k, l = from_networkx_to_tensor(Gs[k], dict)  # adjacency matrixes
        A[k, 0:A_k.shape[1]] = A_k[0]
        labels[k, 0:l.shape[0]] = l

# This function is used to construct a cost matrix C between two graphs g1 and g2, given the costs


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
    Abin1 = (
        A1 == lab1*torch.ones((A1.shape[0], A1.shape[1]), device=device)).int()
    Abin2 = (
        A2 == lab2*torch.ones((A2.shape[0], A2.shape[1]), device=device)).int()
    C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

    return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1)

# ring_g, ring_h come from global ring with all graphs in so ring_g = rings['g'] and ring_h = rings['h']


def lsape_populate_instance(first_graph, second_graph, average_node_cost, average_edge_cost, alpha, lbda, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h):
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

# Finding an adequate mapping based on the given costs, without using the Frank Wolfe method


def mapping_from_cost_sans_FW(n, m, g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel, ring_g, ring_h):
    c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                  edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c_0), 10).view((n+1)*(m+1), 1)
    return x0

# Finding an adequate mapping based on the given costs, using the Frank Wolfe method, and the rings
def new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h): 
    c=torch.diag(C)       
    c_0 =lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    D=C-torch.eye(C.shape[0],device=device)*c
    x0=svd.eps_assigment_from_mapping(torch.exp(-c_0),10).view((n+1)*(m+1),1)
    return svd.franck_wolfe(x0,D,c,5,15,n,m)

# Finding an adequate mapping based on the given costs, using the Frank Wolfe method, without the rings method
def mapping_from_cost(C, n, m):
    c = torch.diag(C)
    D = C-torch.eye(C.shape[0], device=device)*c
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
    return svd.franck_wolfe(x0, D, c, 5, 15, n, m)

def mapping_from_cost_sans_rings_sans_fw(C, n, m):
    c = torch.diag(C)
    x0 = svd.eps_assigment_from_mapping(
        torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
    return x0

# A general function for finding an adequate mapping based on the given costs
def mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw):
    c = torch.diag(C) 
    D = C-torch.eye(C.shape[0],device=device)*c

    if (rings_andor_fw=='rings_sans_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n+1)*(m+1), 1)
        res = x0
    elif (rings_andor_fw=='rings_avec_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0),10).view((n+1)*(m+1),1)
        res = svd.franck_wolfe(x0,D,c,5,15,n,m)
    elif (rings_andor_fw=='sans_rings_avec_fw'):
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
        res = svd.franck_wolfe(x0, D, c, 5, 15, n, m)
    else:
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
        res = x0
    return res

# Calculation of the ged for a given pair of graphs :
def new_primary_ged(g1,g2,n,m,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw):
    if (rings_andor_fw=='rings_sans_fw'):
        ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
        #S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)
    elif (rings_andor_fw=='rings_avec_fw'):
        ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
        #S = new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h)
    elif (rings_andor_fw=='sans_rings_avec_fw'):
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, 0, 0, rings_andor_fw)
        #S = mapping_from_cost(C, n, m)
    else:
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, 0, 0, rings_andor_fw)
        #S = mapping_from_cost_sans_rings_sans_fw(C,n,m)

    v = torch.flatten(S)

    normalize_factor = 1.0
    if normalize:
        nb_edge1 = (A[g1][0:n*n] != torch.zeros(n *n, device=device)).int().sum()
        nb_edge2 = (A[g2][0:m*m] != torch.zeros(m *m, device=device)).int().sum()
        normalize_factor = nodeInsDel*(n+m)+edgeInsDel*(nb_edge1+nb_edge2)
    c = torch.diag(C)
    D = C-torch.eye(C.shape[0], device=device)*c
    ged_k = (.5*v.t()@D@v+c.T@v)/normalize_factor

    return ged_k

# Calculating the geds of the couples of graphs in the train and the test data, given a previous splitting of the data
def calculates_distances(rings_andor_fw, node_costs, edge_costs, nodeInsDel, edgeInsDel,train_D, valid_D,train_L,valid_L): #returns two matrices of distances (train and test)
    n_train=len(train_L)
    n_valid=len(valid_L)

    # Matrices of distances for train and test
    D_train=np.zeros((n_train,n_train))
    D_valid=np.zeros((n_valid,n_train))

    # Calculate ged between couples of graphs (train graphs*train graphs)
    for i,g1_idx in enumerate(train_D): #:51
        print(i)
        for j,g2_idx in enumerate(train_D):
            print(j,g2_idx)
            n=Gs[g1_idx].order()
            m=Gs[g2_idx].order()
            
            ged_k=new_primary_ged(g1_idx,g2_idx,n,m,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw)
            # Filling the corresponding matrix of ged D_train :
            if ged_k<0:
                ged_k = 255
            D_train[i,j]=abs(ged_k)

    print("now test : ")
    # Calculate ged between couples of graphs (test graphs*train graphs)
    for i,g1_idx in enumerate(valid_D):
        print(i,g1_idx)
        for j,g2_idx in enumerate(train_D):
            print(j,g2_idx)
            n=Gs[g1_idx].order()
            m=Gs[g2_idx].order()
            
            # Filling the corresponding matrix of ged D_valid :
            ged_k=new_primary_ged(g1_idx,g2_idx,n,m,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw)
            if ged_k<0:
                ged_k = 255
            D_valid[i,j]=abs(ged_k)

    plt.subplot(121)
    plt.imshow(D_train)
    plt.subplot(122)
    plt.imshow(D_valid)
    # plt.savefig(rings_andor_fw+".png")
    plt.show()

    print("D_valid.shape = ", D_valid.shape)
    return D_train,D_valid,train_L,valid_L 

def knn_ines(D_train,D_valid,train_L,valid_L): 
    # a knn classifier to evaluate the performance of our model
     
    classifier = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    classifier.fit(D_train, train_L)  # train_size*train_size
    y_pred_train = classifier.predict(D_train)
    y_pred_valid = classifier.predict(D_valid)
    print('y_pred_train : ', y_pred_train)
    print('y_pred_valid : ', y_pred_valid)
    #print(confusion_matrix(y_test, y_pred))
    #print('acc : ', np.mean(y_pred == y)*100)
    print("valid_L = ",valid_L, len(valid_L))
    print("Accuracy of the valid : ")
    print(classification_report(valid_L, y_pred_valid)) 
    print("Accuracy of the train : ")
    print(classification_report(train_L, y_pred_train))
    print(confusion_matrix(valid_L, y_pred_valid))


if __name__ == "__main__":

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rings_andor_fw = 'sans_rings_avec_fw'
    device = 'cpu'
    path_dataset = os.getenv('MAO_DATASET_PATH')
    Gs, y = loadDataset(path_dataset)
    card = torch.tensor([G.order() for G in Gs]).to(device)
    card_max = card.max()
    A = torch.empty((len(Gs), card_max*card_max), dtype=torch.int, device=device)
    labels = torch.empty((len(Gs), card_max), dtype=torch.int, device=device)
    normalize = False
    dict,nb_edge_labels = build_node_dictionnary(Gs)
    init_dataset(Gs,dict)
    
    nodeSubInit=torch.load('pickle_files/'+rings_andor_fw+'/nodeSubInit',map_location=torch.device('cpu'),pickle_module=pkl)
    nodeInsDelInit=torch.load('pickle_files/'+rings_andor_fw+'/nodeInsDelInit',map_location=torch.device('cpu'),pickle_module=pkl)
    edgeInsDelInit=torch.load('pickle_files/'+rings_andor_fw+'/edgeInsDelInit',map_location=torch.device('cpu'),pickle_module=pkl)
    edgeSubInit=torch.load('pickle_files/'+rings_andor_fw+'/edgeSubInit',map_location=torch.device('cpu'),pickle_module=pkl)

    new_node_costs = torch.load('pickle_files/'+rings_andor_fw+'/nodeSub_min',map_location=torch.device('cpu'),pickle_module=pkl)
    new_nodeInsDel = torch.load('pickle_files/'+rings_andor_fw+'/nodeInsDel_min',map_location=torch.device('cpu'),pickle_module=pkl)
    new_edgeInsDel = torch.load('pickle_files/'+rings_andor_fw+'/edgeInsDel_min',map_location=torch.device('cpu'),pickle_module=pkl)
    new_edge_costs = torch.load('pickle_files/'+rings_andor_fw+'/edgeSub_min',map_location=torch.device('cpu'),pickle_module=pkl)
   
    nodeSubInit.requires_grad=False
    nodeInsDelInit.requires_grad=False
    edgeInsDelInit.requires_grad=False
    edgeSubInit.requires_grad=False

    new_node_costs.requires_grad=False
    new_nodeInsDel.requires_grad=False
    new_edgeInsDel.requires_grad=False
    new_edge_costs.requires_grad=False

    # Experts' costs
    node_costs_2 = torch.ones(new_node_costs.shape)
    for i in range(node_costs_2.shape[0]):
        node_costs_2[i,i] = 0

    edge_costs_2 = torch.ones(new_edge_costs.shape)
    for i in range(edge_costs_2.shape[0]):
        edge_costs_2[i,i] = 0

    nodeInsDel_2 = torch.tensor(3.0)
    edgeInsDel_2 = torch.tensor(3.0)
    

 #'sans_rings_sans_fw' #sans_rings_avec_fw
   
    #(1) with no_grad, 68 mol : 86 valid 54 train
    #() sans no_grad avec 40 mol : 81 train 75 valid
    #(2) .detach(), 68 mol : 57 train and valid
    #(3) with no_grad, 68 mol : valid   train
    #(4) avec no_grad avec 40 mol : 44 train 75 valid
    
    train_D = torch.load('pickle_files/'+rings_andor_fw+'/train_graph',map_location=torch.device('cpu'),pickle_module=pkl)
    valid_D = torch.load('pickle_files/'+rings_andor_fw+'/test_graph',map_location=torch.device('cpu'),pickle_module=pkl)
    train_L = torch.load('pickle_files/'+rings_andor_fw+'/train_label',map_location=torch.device('cpu'),pickle_module=pkl)
    valid_L = torch.load('pickle_files/'+rings_andor_fw+'/test_label',map_location=torch.device('cpu'),pickle_module=pkl)

    print("train_D = ",train_D, len(train_D))
    print("train_L = ",train_L)
    print("valid_D = ",valid_D, len(valid_D))
    print("valid_L = ",valid_L) 
    
    # D_train,D_valid,train_L,valid_L=calculates_distances(rings_andor_fw, new_node_costs, new_edge_costs, new_nodeInsDel, new_edgeInsDel,train_D, valid_D,train_L,valid_L)
    D_train,D_valid,train_L,valid_L=calculates_distances(rings_andor_fw, nodeSubInit, edgeSubInit, nodeInsDelInit, edgeInsDelInit,train_D, valid_D,train_L,valid_L)

    # D_train,D_valid,train_L,valid_L=calculates_distances(rings_andor_fw, node_costs_2, edge_costs_2, nodeInsDel_2, edgeInsDel_2,train_D, valid_D,train_L,valid_L)
    print(D_train,len(D_train),len(D_train[0]))
    knn_ines(D_train,D_valid,train_L,valid_L)
