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
from sklearn.model_selection import train_test_split
from sklearn import datasets
from torch.utils.data import DataLoader, random_split, TensorDataset
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

    return dict,max(max([[int(G[e[0]][e[1]]['bond_type']) for e in G.edges()] for G in GraphList]))


def from_networkx_to_tensor(G,dict):    
    A_g = torch.tensor(nx.to_scipy_sparse_matrix(G,dtype=int,weight='bond_type').todense(),dtype=torch.int)        
    lab = [dict[G.nodes[v]['label'][0]] for v in nx.nodes(G)]

    return A_g.view(1,A_g.shape[0]*A_g.shape[1]),torch.tensor(lab)

def init_dataset(Gs,dict):
    for k in range(len(Gs)):
        A_k,l = from_networkx_to_tensor(Gs[k],dict)   #adjacency matrixes          
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
    # Finding an adequate mapping based on the given costs, without using the Frank Wolfe method
    c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                  edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n+1)*(m+1), 1)
    return x0

def new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h): 
    # Finding an adequate mapping based on the given costs, using the Frank Wolfe method
    c=torch.diag(C)       
    c_0 =lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,
                                edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
    D=C-torch.eye(C.shape[0],device=device)*c
    x0=svd.eps_assigment_from_mapping(torch.exp(-c_0),10).view((n+1)*(m+1),1)
    return svd.franck_wolfe(x0,D,c,5,15,n,m)

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

def mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw):
    # A general function for finding an adequate mapping based on the given costs
    c = torch.diag(C) 
    D = C-torch.eye(C.shape[0],device=device)*c

    if (rings_andor_fw=='rings_sans_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0), 10).view((n+1)*(m+1), 1)
        res = x0
    if (rings_andor_fw=='rings_avec_fw'):
        c_0 = lsape_populate_instance(g1, g2, node_costs, edge_costs, nodeInsDel,edgeInsDel, node_costs, nodeInsDel, edge_costs, edgeInsDel, ring_g, ring_h)
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c_0),10).view((n+1)*(m+1),1)
        res = svd.franck_wolfe(x0,D,c,5,15,n,m)
    if (rings_andor_fw=='sans_rings_avec_fw'):
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
        res = svd.franck_wolfe(x0, D, c, 5, 15, n, m)
    if (rings_andor_fw=='sans_rings_sans_fw'):
        x0 = svd.eps_assigment_from_mapping(torch.exp(-c.view(n+1, m+1)), 10).view((n+1)*(m+1), 1)
        res = x0
    return res


def primary_ged(k,inputs,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw):
    # calculation of the ged for a given pair of graphs :

    g1 = inputs[k][0]
    g2 = inputs[k][1]
    n = card[g1]
    m = card[g2]

    if (rings_andor_fw=='rings_sans_fw'):
        ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
        #S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)
    
    if (rings_andor_fw=='rings_avec_fw'):
        ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
        #S = new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h)
    if (rings_andor_fw=='sans_rings_avec_fw'):
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, 0, 0, rings_andor_fw)
        #S = mapping_from_cost(C, n, m)
    if (rings_andor_fw=='sans_rings_sans_fw'):
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
    


def gedd(data, node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw):
    # calculation of the ged for all pairs of graphs :
    inputs = data.to(device)
    ged = torch.zeros(len(inputs)).to(device)
    for k in range(len(inputs)):
        ged[k] = primary_ged(k,inputs,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw)

    return ged

def ged_to_pkl(data, node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw): 
    # storing result of ged into pickle file
    
    filename = 'ged_'+rings_andor_fw+'.pkl'
    ged_res = gedd(data, node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw) 
    print('ged : ', ged_res)
    print('ged.size() : ', ged_res.size())

    with open(filename, "wb") as fout:
        pkl.dump(ged_res, fout, protocol=pkl.HIGHEST_PROTOCOL)

    ged=ged_res.detach().numpy()   
    plt.plot(ged)
    plt.title('ged for train set')
    plt.show()
    return ged_res

def ged_from_pkl(rings_andor_fw):  
    # getting result of ged from pickle file
    
    filename='ged_'+rings_andor_fw+'.pkl'
    with open(filename, "rb") as fout:
        ged_pkl = pkl.load(fout)
    #new_ged_pkl = ged_pkl.detach().numpy()
    return ged_pkl  #new_ged_pkl

def train_data(train_size,class1, class2):
    # Splitting data
    nb_elt = int(train_size*(train_size-1)/2)
    data = torch.empty((nb_elt, 2), dtype=torch.int)
    train_graphs = torch.cat((class1, class2), 0)
    #print('train_graphs : ', train_graphs)
    couples = torch.triu_indices(train_size, train_size, offset=1)
    data[0:nb_elt, 0] = train_graphs[couples[0, :]]
    data[0:nb_elt, 1] = train_graphs[couples[1, :]]
    #print('couples : ', couples[0, :], couples.shape)
    return data,train_graphs

def new_class(class1,random_class1):
    #Creating a new class different from the given class 
    tmp=class1.numpy()
    res=[]
    for i in range(len(tmp)-1):
        if tmp[i] in random_class1:
            res.append(i)
    tmp=np.delete(tmp,res)
    class1=torch.tensor(tmp)
    return class1

def tr_test_data(train_size, class1, class2):
    nb_class1=12
    nb_class2=int((nb_class1-1)/2)
    #nb_elt=int(nb_class1*(nb_class1+2*nb_class2-1)/2)
    nb_elt=56
    print("nb_elt = ",nb_elt)
    #nb_elt = int(train_size*(train_size-1)/2)
    
    data = torch.empty((nb_elt, 2), dtype=torch.int)
    yt=torch.ones(nb_elt)
    random_class1=class1[(torch.abs(10000*torch.randn(30)).int()%class1.size()[0]).long()]
    random_class2=class2[(torch.abs(10000*torch.randn(26)).int()%class2.size()[0]).long()]
    train_graphs=torch.cat((random_class1,random_class2),0)

    couples=torch.triu_indices(56,56,offset=1)  #(train_size,train_size,offset=1)
    data[0:nb_elt,0]=train_graphs[couples[0,0:nb_elt]]
    data[0:nb_elt,1]=train_graphs[couples[1,0:nb_elt]]
    for k in range(nb_elt):
        if (y[data[k][0]]!=y[data[k][1]]):
            yt[k]=-1.0      
    #[train_D, valid_D,train_L,valid_L]= train_test_split(data,yt, test_size=0.25,train_size=0.75, shuffle=True) #, stratify=yt)
    [train_D, valid_D,train_L,valid_L]= train_test_split(data,yt, test_size=0.25,train_size=0.75, shuffle=True, stratify=yt) 

    #DatasetTrain = TensorDataset(train_D, train_L)
    #DatasetValid=TensorDataset(valid_D, valid_L)

    #trainloader=torch.utils.data.DataLoader(DatasetTrain,batch_size=len(train_D),shuffle=True,drop_last=True, num_workers=0)
    #validationloader=torch.utils.data.DataLoader(DatasetValid, batch_size=8, drop_last=True,num_workers=0)

    print(len(train_D), len(valid_D))
    print("len data = ",len(data),data,'\n')
    return data,train_graphs,train_D,valid_D,train_L,valid_L

def new_primary_ged(g1,g2,n,m,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw):
    # calculation of the ged for a given pair of graphs :

    if (rings_andor_fw=='rings_sans_fw'):
        ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
        #S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)
    
    if (rings_andor_fw=='rings_avec_fw'):
        ring_g,ring_h = rings.build_rings(g1,edgeInsDel.size()), rings.build_rings(g2,edgeInsDel.size())
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
        #S = new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h)
    if (rings_andor_fw=='sans_rings_avec_fw'):
        C = construct_cost_matrix(g1, g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
        S = mapping_from_cost_method(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, 0, 0, rings_andor_fw)
        #S = mapping_from_cost(C, n, m)
    if (rings_andor_fw=='sans_rings_sans_fw'):
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


def calculates_distances(rings_andor_fw, node_costs, edge_costs, nodeInsDel, edgeInsDel,train_D, valid_D,train_L,valid_L): #returns two matrices of distances (train and test)
    # Calculating the geds of the couples of graphs in the train and the test data, given a previous splitting of the data
    
    my_list=[i for i in range(68)] 
    #[train_D, valid_D,train_L,valid_L]= train_test_split(my_list,y[:8], test_size=0.25,train_size=0.75, shuffle=True) 
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
            '''
            ring_g,ring_h = rings.build_rings(g1_idx,edgeInsDel.size()), rings.build_rings(g2_idx,edgeInsDel.size())
            C = construct_cost_matrix(g1_idx, g2_idx, node_costs, edge_costs, nodeInsDel, edgeInsDel)
            S = mapping_from_cost_method(C,n,m,g1_idx,g2_idx,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
            v = torch.flatten(S)

            normalize_factor = 1.0
            c = torch.diag(C)
            D = C-torch.eye(C.shape[0], device=device)*c
            ged_k = (.5*v.t()@D@v+c.T@v)/normalize_factor
            '''
            ged_k=new_primary_ged(g1_idx,g2_idx,n,m,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw)
            # Filling the corresponding matrix of ged D_train :
            D_train[i,j]=ged_k

    print("now test : ")
    # Calculate ged between couples of graphs (test graphs*train graphs)
    for i,g1_idx in enumerate(valid_D):
        print(i,g1_idx)
        for j,g2_idx in enumerate(train_D):
            print(j,g2_idx)
            n=Gs[g1_idx].order()
            m=Gs[g2_idx].order()
            '''
            ring_g,ring_h = rings.build_rings(g1_idx,edgeInsDel.size()), rings.build_rings(g2_idx,edgeInsDel.size())
            C = construct_cost_matrix(g1_idx, g2_idx, node_costs, edge_costs, nodeInsDel, edgeInsDel)
            S = mapping_from_cost_method(C,n,m,g1_idx,g2_idx,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h, rings_andor_fw)
            v = torch.flatten(S)

            normalize_factor = 1.0
            c = torch.diag(C)
            D = C-torch.eye(C.shape[0], device=device)*c
            ged_k = (.5*v.t()@D@v+c.T@v)/normalize_factor
            '''
            # Filling the corresponding matrix of ged D_valid :
            ged_k=new_primary_ged(g1_idx,g2_idx,n,m,node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw)
            D_valid[i,j]=ged_k
            
    print("D_valid.shape = ", D_valid.shape)
    return D_train,D_valid,train_L,valid_L 

def knn_ines(D_train,D_valid,train_L,valid_L): 
    # a knn classifier to evaluate the performance of our model
     
    classifier = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    print(len(train_L))
    classifier.fit(D_train, train_L)  # train_size*train_size
    y_pred_train = classifier.predict(D_train)
    y_pred_valid = classifier.predict(D_valid)
    print('y_pred_train : ', y_pred_train)
    print('y_pred_valid : ', y_pred_valid)
    #print(confusion_matrix(y_test, y_pred))
    #print('acc : ', np.mean(y_pred == y)*100)
    print("valid_L = ",valid_L, len(valid_L))
    print(classification_report(valid_L, y_pred_valid)) 
    print(classification_report(train_L, y_pred_train))


def knn(train_graphs, train_size, ged_pkl,train_D,valid_D,train_L,valid_L):
    new_train_size = 56
    triu_indices = torch.triu_indices(row=new_train_size, col=new_train_size, offset=1)
    D = torch.zeros((new_train_size, new_train_size))
    D[triu_indices[0, :], triu_indices[1, :]][:56] = ged_pkl
    D = D+D.t()
    test_size=int((new_train_size*25)/100)
    D_test = torch.zeros((test_size, new_train_size))


    #triu_indices_test = torch.triu_indices(row=test_size, col=new_train_size, offset=1)
    #D_test[triu_indices_test[0, :], triu_indices_test[1, :]] = ged_pkl
    #D_test = D_test + D_test.t()

    '''
    plt.matshow(D)
    plt.colorbar()
    plt.title('D ')
    plt.show()
    '''
    classifier = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    print('D_test.shape : ',D_test.shape)

    ytrain = []
    for g in train_graphs:
        ytrain.append(y[g])
    print('ytrain : ', ytrain)
    print('train_size = ', train_size)
    print("new_train_size = ",new_train_size)
    
    # plt.plot(ytrain)
    # plt.show()
    #print('ged size : ', ged_pkl.size())

    print("train_L = ",len(train_L))
    classifier.fit(D, ytrain)  # train_size*train_size
    #y_pred = classifier.predict(D)  # test_size*train_size
    y_pred = classifier.predict(D_test) #valid_D
    print('y_pred : ', y_pred)
    #print(confusion_matrix(y_test, y_pred))
    print('acc : ', np.mean(y_pred == y)*100)
    print("valid_L = ",valid_L, len(valid_L))
    print(classification_report(valid_L, y_pred)) #valid_L,y_pred
    #print(classification_report(data_test, y_pred)) #add data_test in parameters

    plt.plot(D[12, :], label='D[12, :]')
    plt.plot(ytrain, label='ytrain')
    plt.title('ytrain and D[12, :]')
    plt.legend()
    plt.show()

def main_testing_function(train_size, rings_andor_fw, node_costs, edge_costs, nodeInsDel, edgeInsDel, class1, class2):
    #data,train_graphs = train_data(train_size, class1, class2)
    data,train_graphs,train_D,valid_D,train_L,valid_L = tr_test_data(train_size, class1, class2)
    ged_pkl=ged_to_pkl(data, node_costs, edge_costs, nodeInsDel, edgeInsDel, rings_andor_fw)

    g1=0
    g2=1
    C=construct_cost_matrix(g1,g2, node_costs, edge_costs, nodeInsDel, edgeInsDel)
    print('Cost matrix :',len(C),C)
    plt.matshow(C)
    plt.title('Cost matrix for g1=0 and g2=1')
    plt.show()

    new_ged_pkl = ged_from_pkl(rings_andor_fw)

    knn(train_graphs, train_size, new_ged_pkl, train_D,valid_D,train_L,valid_L)



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

    new_node_costs = torch.load('../Luc/deepged/nodeSub_min',pickle_module=pkl)
    new_nodeInsDel = torch.load('../Luc/deepged/nodeInsDel_min',pickle_module=pkl)
    new_edgeInsDel = torch.load('../Luc/deepged/edgeInsDel_min',pickle_module=pkl)
    new_edge_costs = torch.load('../Luc/deepged/edgeSub_min',pickle_module=pkl)
    new_node_costs.requires_grad=False
    new_nodeInsDel.requires_grad=False
    new_edgeInsDel.requires_grad=False
    new_edge_costs.requires_grad=False

    
    node_costs_2 = torch.ones(new_node_costs.shape)
    for i in range(node_costs_2.shape[0]):
        node_costs_2[i,i] = 0

    edge_costs_2 = torch.ones(new_edge_costs.shape)
    for i in range(edge_costs_2.shape[0]):
        edge_costs_2[i,i] = 0

    nodeInsDel_2 = torch.tensor(3.0)
    edgeInsDel_2 = torch.tensor(3.0)
    

    rings_andor_fw = 'rings_sans_fw'  #'sans_rings_sans_fw' #sans_rings_avec_fw
    #train_size = 68
    nb_class1=12
    nb_class2=int((nb_class1-1)/2)
    train_size=nb_class1+nb_class2
    #train_size=51
    #main_testing_function(train_size, rings_andor_fw, new_node_costs, new_edge_costs, new_nodeInsDel, new_edgeInsDel, class1, class2)

    train_D = torch.load('../Luc/deepged/train_D',pickle_module=pkl) 
    valid_D = torch.load('../Luc/deepged/valid_D',pickle_module=pkl) 
    train_L = torch.load('../Luc/deepged/train_L',pickle_module=pkl) 
    valid_L = torch.load('../Luc/deepged/valid_L',pickle_module=pkl) 

    print("train_D = ",train_D, len(train_D))
    print("train_L = ",train_L)
    print("valid_D = ",valid_D, len(valid_D))
    print("valid_L = ",valid_L)
    
    #D_train,D_valid,train_L,valid_L=calculates_distances(rings_andor_fw, new_node_costs, new_edge_costs, new_nodeInsDel, new_edgeInsDel,train_D, valid_D,train_L,valid_L)
    D_train,D_valid,train_L,valid_L=calculates_distances(rings_andor_fw, node_costs_2, edge_costs_2, nodeInsDel_2, edgeInsDel_2,train_D, valid_D,train_L,valid_L)
    print(D_train,len(D_train),len(D_train[0]))
    knn_ines(D_train,D_valid,train_L,valid_L)



    '''
    #filename='/home/ines/Downloads/D_mao.pickle'
    D=new_ged_pkl
    clf = KNeighborsClassifier(3, metric="precomputed")
    clf.fit(D, y)
    ypred = clf.predict(D)
    print(np.mean(ypred == y)*100)
    '''
    print()