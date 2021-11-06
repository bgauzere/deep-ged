# This function splits the graph list Gs into two distinct sets of couples of graphs
# One for training the model and one for testing it
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle as pkl
import random
import numpy as np

def splitting(Gs, y):
    my_list = [i for i in range(len(Gs))]

    [train_D, valid_D, train_L, valid_L] = train_test_split(my_list, y, test_size=0.20, train_size=0.80, shuffle=True,
                                                            stratify=y)  # we stratify so that y is used as the class labels
    # We make sure that the two sets contain distinct graphs
    train_D, valid_D = different_sets(train_D, valid_D, Gs)

    couples_train, yt, couples_test_train, yv = creating_couples_after_splitting(train_D, valid_D, y)
    yt = torch.tensor(yt)
    yv = torch.tensor(yv)
    DatasetTrain = TensorDataset(couples_train, yt)
    DatasetValid = TensorDataset(couples_test_train, yv)

    trainloader = torch.utils.data.DataLoader(DatasetTrain, batch_size=len(couples_train), shuffle=True, drop_last=True,
                                              num_workers=0)  # 128, len(couples_train)
    validationloader = torch.utils.data.DataLoader(DatasetValid, batch_size=len(couples_test_train), drop_last=True,
                                                   num_workers=0)  # 64,128,len(couples_test_train)

    print(len(trainloader), len(validationloader))
    print(len(trainloader), len(validationloader))

    # We save our sets in pickle files
    torch.save(train_D, 'pickle_files/train_D', pickle_module=pkl)
    torch.save(valid_D, 'pickle_files/valid_D', pickle_module=pkl)
    torch.save(train_L, 'pickle_files/train_L', pickle_module=pkl)
    torch.save(valid_L, 'pickle_files/valid_L', pickle_module=pkl)

    return trainloader, validationloader, couples_train, yt, couples_test_train, yv


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


def creating_couples_after_splitting(train_D, valid_D, y):
    couples_train = []
    couples_test_train = []
    for i, g1_idx in enumerate(train_D):
        for j, g2_idx in enumerate(train_D):
            n = g1_idx
            m = g2_idx
            couples_train.append([n, m])
    yt = np.ones(len(couples_train))
    for k in couples_train:
        if (y[k[0]] != y[k[1]]):
            yt[k] = -1.0
    for i, g1_idx in enumerate(valid_D):
        for j, g2_idx in enumerate(train_D):
            n = g1_idx
            m = g2_idx
            couples_test_train.append([n, m])

    yv = np.ones(len(couples_test_train))
    for k in couples_test_train:
        if (y[k[0]] != y[k[1]]):
            yv[k] = -1.0

    return torch.tensor(couples_train), yt, torch.tensor(couples_test_train), yv