import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def initialize_dataset_split(graphs, y,
                             train_size=0.6, valid_size=0.2, test_size=0.2,
                             shuffle=True):
    '''
    Split dataset indices into train, valid, test. Returns the
    three datasets with indices of graphs in Gs
    '''
    graph_idx = torch.arange(0, len(graphs), dtype=torch.int64)
    [train_graph, remain_graph,
     train_label, remain_label] = train_test_split(graph_idx, y,
                                                   train_size=train_size,
                                                   test_size=test_size + valid_size,
                                                   shuffle=shuffle,
                                                   stratify=y)
    adjusted_ratio_valid = valid_size/(1-train_size)
    # pour le second split. ratios réajustés.
    adjusted_ratio_test = test_size/(1-train_size)

    [valid_graph, test_graph,
     valid_label, test_label] = train_test_split(remain_graph, remain_label,
                                                 test_size=adjusted_ratio_test,
                                                 train_size=adjusted_ratio_valid,
                                                 shuffle=shuffle,
                                                 stratify=remain_label)
    dataset_train = (train_graph, train_label)
    dataset_valid = (valid_graph, valid_label)
    dataset_test = (test_graph, test_label)
    return dataset_train, dataset_valid, dataset_test


def build_pairs_of_graphs_for_classification(graph_indices, y, avoid_pair_of_negative=True):
    '''
    Associe des index couples de    graphes à leur similarité de
    classes ! réservé à la classif !
    '''
    couples_train = []
    paired_y = []
    for idx_i, y_i in zip(graph_indices, y):
        # on rajoute les paires de graphes similaires (intéressant ?)
        couples_train.append([idx_i, idx_i])
        paired_y.append(1)  # forcément meme classe
        for idx_j, y_j in zip(graph_indices, y):
            if (idx_i < idx_j):  # on rajoute qu'une fois un couple
                if not (avoid_pair_of_negative and y_i == 0 and y_j == 0):
                    couples_train.append([idx_i, idx_j])
                    paired_y.append(1 if (y_i == y_j) else -1)
    return torch.tensor(couples_train), torch.tensor(paired_y)


def generate_dataloader(graph_indices, graph_label):
    dataset = TensorDataset(graph_indices, graph_label)
    loader = DataLoader(dataset, batch_size=len(dataset), drop_last=False)
    return loader


def from_indices_to_dataloader(graph_indices, graph_label, avoid_pair_of_negative=True):
    data, y = build_pairs_of_graphs_for_classification(
        graph_indices, graph_label, avoid_pair_of_negative)
    return generate_dataloader(data, y)


def initialize_dataset(graphs, y, avoid_pair_of_negative=True, train_size=0.6, valid_size=0.2, test_size=0.2, shuffle=True):
    '''
    Returns three torch dataLoader for train, valid and test according to ratios
    '''
    dataset_train, dataset_valid, dataset_test = initialize_dataset_split(graphs, y,
                                                                          train_size=train_size, valid_size=valid_size, test_size=test_size,
                                                                          shuffle=shuffle)
    loader_train = from_indices_to_dataloader(*dataset_train)
    loader_valid = from_indices_to_dataloader(*dataset_valid)
    loader_test = from_indices_to_dataloader(*dataset_test)
    return loader_train, loader_valid, loader_test
