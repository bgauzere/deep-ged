import torch
from torch.utils import data
from torch import triu_indices

from sklearn.model_selection import train_test_split


# Génération des loader's pour les différents dataSet (entrainement et validation)
def gen_loader(graph_list, graph_label, params):
    graph_idx = torch.arange(0, len(graph_list), dtype=torch.int64)

    [train_graph, valid_graph, train_label, valid_label] = train_test_split(graph_idx, graph_label, test_size=0.20,
                                                                            train_size=0.80, shuffle=True,
                                                                            stratify=graph_label)

    dataset_train = DataSet(train_graph, train_label)
    dataset_valid = DataSet(valid_graph, valid_label)

    generator_train = data.DataLoader(dataset_train, **params)
    generator_valid = data.DataLoader(dataset_valid, **params)

    return generator_train, generator_valid


# Objet dataset qui est géré sous forme de map
class DataSet(data.Dataset):
    def __init__(self, graph_idx, graph_label):
        self.graph_idx = graph_idx
        self.graph_label = graph_label

        self.graph_set_size = len(self.graph_idx)
        self.data_set_size = int((self.graph_set_size - 1) * (self.graph_set_size) / 2)

        # On fait en sorte de prendre des couples différents ie les couples n-m et m-n sont considérés comme identiques
        self.indice = torch.triu_indices(self.graph_set_size, self.graph_set_size, offset=1)

    def __getitem__(self, key):
        # Label des couples de graphes
        label = -1
        couple = self.indice[:, key]
        if self.graph_label[couple[0]] == self.graph_label[couple[1]]:
            label = 1
        return (couple, label)

    def __len__(self):
        return self.data_set_size