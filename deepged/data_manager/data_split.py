'''
This function splits the graph list Gs into two distinct sets of couples of graphs
One for training the model and one for testing it
'''
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle as pkl
import random
import numpy as np
import os


def splitting(Gs, y, saving_path=None, already_divided=False):
    graph_idx = torch.arange(0, len(Gs), dtype=torch.int64)

    if already_divided and saving_path is not None:
        print("Already divided dataset : loading...")
        train_graph = torch.load('pickle_files/' + saving_path + '/train_graph', map_location=torch.device('cpu'),
                                 pickle_module=pkl)
        valid_graph = torch.load('pickle_files/' + saving_path + '/valid_graph', map_location=torch.device('cpu'),
                                 pickle_module=pkl)
        test_graph = torch.load('pickle_files/' + saving_path + '/test_graph', map_location=torch.device('cpu'),
                                pickle_module=pkl)
        train_label = torch.load('pickle_files/' + saving_path + '/train_label', map_location=torch.device('cpu'),
                                 pickle_module=pkl)
        valid_label = torch.load('pickle_files/' + saving_path + '/valid_label', map_location=torch.device('cpu'),
                                 pickle_module=pkl)
        test_label = torch.load('pickle_files/' + saving_path + '/test_label', map_location=torch.device('cpu'),
                                pickle_module=pkl)
    else:
        [train_graph, valid_graph, train_label, valid_label] = train_test_split(graph_idx, y, test_size=0.40,
                                                                                train_size=0.60, shuffle=True,
                                                                                stratify=y)

        [valid_graph, test_graph, valid_label, test_label] = train_test_split(valid_graph, valid_label, test_size=0.50,
                                                                              train_size=0.50, shuffle=True,
                                                                              stratify=valid_label)

    couples_train, yt = creating_couples_after_splitting(train_graph, y)
    couples_valid, yv = creating_couples_after_splitting(valid_graph, y)
    couples_test, yte = creating_couples_after_splitting(test_graph, y)
    yt = torch.tensor(yt)
    yv = torch.tensor(yv)
    yte = torch.tensor(yte)
    DatasetTrain = TensorDataset(couples_train, yt)
    DatasetValid = TensorDataset(couples_valid, yv)
    DatasetTest = TensorDataset(couples_test, yte)

    trainloader = torch.utils.data.DataLoader(DatasetTrain, batch_size=len(couples_train), shuffle=True, drop_last=True,
                                              num_workers=0)  # 128, len(couples_train)
    validationloader = torch.utils.data.DataLoader(DatasetValid, batch_size=len(couples_valid), drop_last=True,
                                                   num_workers=0)  # 64,128,len(couples_test_train)

    testloader = torch.utils.data.DataLoader(DatasetTest, batch_size=len(couples_test), drop_last=True,
                                             num_workers=0)  # 64,128,len(couples_test_train)

    print(len(trainloader), len(validationloader))
    print(len(trainloader), len(validationloader))

    if not os.path.exists('pickle_files/'+saving_path):
        os.makedirs('pickle_files/'+saving_path)

    if saving_path is not None and not already_divided:
        torch.save(train_graph, 'pickle_files/' + saving_path +
                   '/train_graph', pickle_module=pkl)
        torch.save(valid_graph, 'pickle_files/' + saving_path +
                   '/valid_graph', pickle_module=pkl)
        torch.save(test_graph, 'pickle_files/' + saving_path +
                   '/test_graph', pickle_module=pkl)
        torch.save(train_label, 'pickle_files/' + saving_path +
                   '/train_label', pickle_module=pkl)
        torch.save(valid_label, 'pickle_files/' + saving_path +
                   '/valid_label', pickle_module=pkl)
        torch.save(test_label, 'pickle_files/' + saving_path +
                   '/test_label', pickle_module=pkl)

    return trainloader, validationloader, testloader


def creating_couples_after_splitting(train_D, y):
    '''
    Associe des index couples de graphes  à leur similarité de classes
    ! réservé à la classif !
    '''
    couples_train = []
    class1 = [k for k in train_D if y[k] == 1]
    class0 = [k for k in train_D if y[k] == 0]
    nb_class1 = len(class1)
    nb_class0 = min(len(class0), int((nb_class1-1)/2))
    new_train = class1+class0[0:nb_class0]
    for i, elt1 in enumerate(new_train[0:nb_class1]):
        for elt2 in new_train[i+1:]:
            couples_train.append([elt1, elt2])
    # for i, g1_idx in enumerate(train_D):
    #     for j, g2_idx in enumerate(train_D):
    #         n = g1_idx
    #         m = g2_idx
    #         couples_train.append([n, m])
    yt = np.ones(len(couples_train))
    for k, [g1_idx, g2_idx] in enumerate(couples_train):
        if (y[g1_idx] != y[g2_idx]):
            yt[k] = -1.0
    return torch.tensor(couples_train), yt
