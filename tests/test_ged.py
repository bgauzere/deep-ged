from data_manager.label_manager import compute_extended_labels
import pytest
from layers.layer import Net
from gklearn.utils.graphfiles import loadDataset
import torch
from graph_torch import ged_torch

Gs, y = loadDataset('../DeepGED/MAO/dataset.ds')
for g in Gs:
    compute_extended_labels(g)

model = Net(Gs)


@pytest.fixture(scope="function")
def a_graph():
    return 12


def test_ged_0(a_graph):
    """
    test si la ged d'un graphe avec lui meme est 0
    """
    cns, cndl, ces, cedl = model.from_weighs_to_costs()
    n = model.card[a_graph]
    C = model.construct_cost_matrix(a_graph, a_graph, cns, ces, cndl, cedl)
    v = torch.flatten(torch.eye(n+1))

    c = torch.diag(C)
    D = C - torch.eye(C.shape[0], ) * c
    ged = (.5 * v.T @ D @ v + c.T @ v)

    assert (ged == 0)


def test_all_ged():
    for a_graph in range(len(Gs)):
        cns, cndl, ces, cedl = model.from_weighs_to_costs()
        n = model.card[a_graph]
        C = model.construct_cost_matrix(a_graph, a_graph, cns, ces, cndl, cedl)
        v = torch.flatten(torch.eye(n + 1))

        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], ) * c
        ged = (.5 * v.T @ D @ v + c.T @ v)
        assert(ged == 0)
