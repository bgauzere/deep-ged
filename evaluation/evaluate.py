from gklearn.utils.graphfiles import loadDataset
from data_manager import DataSet
from graph_torch import ged_torch, rings
from sklearn import neighbors, metrics
from tqdm import tqdm
import sys
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def classification(train_data, test_data, train_label, test_label, graphList, node_label, edge_label, k = 5, cns=None, ces=None, cndl=None, cedl=None, rings_andor_fw="rings_sans_fw"):
    """

    :param train_data: list of graph in training set
    :param test_data: list of graph in test set
    :param train_label: corresponding list of label in training set
    :param test_label: corresponding list of label in test set
    :param graphList: list of networkx graphs
    :param node_label: label on nodes used in networkx graphs
    :param edge_label: label on edges used in networkx graphs
    :param k: number of neigbors in KNN
    :param cns: cost node substitution
    :param ces: cost edge substitution
    :param cndl: cost node delete and insert
    :param cedl: cost edge delete and insert
    :param rings_andor_fw: method for ged computation
    :return:
    """

    node_label_dict, nb_edge_labels = ged_torch.build_node_dictionnary(graphList, node_label, edge_label)
    n_graph_test = len(test_graph)
    n_graph_train = len(train_graph)
    ged_matrix_train = np.zeros((n_graph_train, n_graph_train))
    ged_matrix_test = np.zeros((n_graph_test, n_graph_train))

    # Compute the ged matrix for each couple in the training data
    for i in tqdm(range(len(train_data))):
        for j in range(len(train_data)):
            g1 = graphList[train_data[i]]
            g2 = graphList[train_data[j]]
            ged = compute_ged(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels, rings_andor_fw)
            ged_matrix_train[i, j] = ged

    # Compute the ged between each graph in the test set and each graph in the training set
    for i in tqdm(range(len(test_data))):
        for j in range(len(train_data)):
            g1 = graphList[test_data[i]]
            g2 = graphList[train_data[j]]
            ged = compute_ged(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels, rings_andor_fw)
            ged_matrix_test[i, j] = ged

    # Definition of the KNN classifier. precomputed stands for "precomputed distances", which we did by computing GED
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    classifier.fit(ged_matrix_train, train_label)
    y_pred_test = classifier.predict(ged_matrix_test)

    print("Test Results : ")
    print(metrics.classification_report(test_label, y_pred_test))
    print(metrics.confusion_matrix(test_label, y_pred_test))

    plt.subplot(121)
    plt.imshow(ged_matrix_train)
    plt.subplot(122)
    plt.imshow(ged_matrix_test)
    plt.show()


def compute_ged(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels, rings_andor_fw):
    n = g1.order()
    m = g2.order()
    if rings_andor_fw == 'rings_sans_fw':
        ring_g, ring_h = rings.build_rings(g1, cedl.size()), rings.build_rings(g2, cedl.size())
        C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels)
        S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, ring_g, ring_h,
                                     rings_andor_fw)
        # S=mapping_from_cost_sans_FW(n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel,ring_g,ring_h)

    elif rings_andor_fw == 'rings_avec_fw':
        ring_g, ring_h = rings.build_rings(g1, cedl.size()), rings.build_rings(g2, cedl.size())
        C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels)
        S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, ring_g, ring_h,
                                     rings_andor_fw)
        # S = new_mapping_from_cost(C,n,m,g1,g2,node_costs,edge_costs,nodeInsDel,edgeInsDel, ring_g, ring_h)
    elif rings_andor_fw == 'sans_rings_avec_fw':
        C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels)
        S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, 0, 0, rings_andor_fw)
        # S = mapping_from_cost(C, n, m)
    elif rings_andor_fw == 'sans_rings_sans_fw':
        C = ged_torch.construct_cost_matrix(g1, g2, cns, ces, cndl, cedl, node_label, node_label_dict, nb_edge_labels)
        S = ged_torch.mapping_from_cost_method(C, n, m, g1, g2, cns, ces, cndl, cedl, 0, 0, rings_andor_fw)

    else:
        print("Error : rings_andor_fw => value not understood")
        sys.exit()

    v = torch.flatten(S)
    c = torch.diag(C)
    D = C - torch.eye(C.shape[0], ) * c
    ged = (.5 * v.T @ D @ v + c.T @ v)
    return ged


if __name__ == "__main__":
    Gs, y = loadDataset('../DeepGED/MAO/dataset.ds')
    batch_size = 1

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}


    rings_andor_fw = "sans_rings_sans_fw"
    weights = "learned"

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
        nodeSubInit = torch.load('pickle_files/'+rings_andor_fw+'/nodeSubInit', map_location=torch.device('cpu'), pickle_module=pkl)
        nodeInsDelInit = torch.load('pickle_files/'+rings_andor_fw+'/nodeInsDelInit', map_location=torch.device('cpu'), pickle_module=pkl)
        edgeInsDelInit = torch.load('pickle_files/'+rings_andor_fw+'/edgeInsDelInit', map_location=torch.device('cpu'), pickle_module=pkl)
        edgeSubInit = torch.load('pickle_files/'+rings_andor_fw+'/edgeSubInit', map_location=torch.device('cpu'), pickle_module=pkl)
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

    classification(train_graph, test_graph, train_label, test_label, Gs, 'label', 'bond_type', 5,
                   cns=cns, ces=ces, cndl=cndl, cedl=cedl)
