# This function splits the graph list Gs into two distinct sets of couples of graphs
# One for training the model and one for testing it
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



    # We make sure that the two sets contain distinct graphs
    train_graph, valid_graph = different_sets(train_graph, valid_graph, Gs)

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
        torch.save(train_graph, 'pickle_files/' + saving_path + '/train_graph', pickle_module=pkl)
        torch.save(valid_graph, 'pickle_files/' + saving_path + '/valid_graph', pickle_module=pkl)
        torch.save(test_graph, 'pickle_files/' + saving_path + '/test_graph', pickle_module=pkl)
        torch.save(train_label, 'pickle_files/' + saving_path + '/train_label', pickle_module=pkl)
        torch.save(valid_label, 'pickle_files/' + saving_path + '/valid_label', pickle_module=pkl)
        torch.save(test_label, 'pickle_files/' + saving_path + '/test_label', pickle_module=pkl)

    return trainloader, validationloader, testloader


# Verifying that the two sets contain different graphs

def different_sets(my_train_D, my_valid_D, Gs):
    cp = my_valid_D
    for i in range(len(my_valid_D)):
        if my_valid_D[i] in my_train_D:
            tmp = random.choice(Gs)
            if tmp not in my_train_D:
                cp[i] = tmp
    my_valid_D = cp

    return my_train_D, my_valid_D


def creating_couples_after_splitting(train_D, y):
    couples_train = []

    for i, g1_idx in enumerate(train_D):
        for j, g2_idx in enumerate(train_D):
            n = g1_idx
            m = g2_idx
            couples_train.append([n, m])
    yt = np.ones(len(couples_train))
    for k in couples_train:
        if (y[k[0]] != y[k[1]]):
            yt[k] = -1.0

    return torch.tensor(couples_train), yt