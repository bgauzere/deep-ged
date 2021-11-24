from deepged.data_manager.label_manager import compute_extended_labels, build_node_dictionnary
from deepged.utils import from_networkx_to_tensor
import pytest
from deepged.model import GedLayer
from gklearn.utils.graphfiles import loadDataset
import torch
import os


path_dataset = os.getenv('MAO_DATASET_PATH')
Gs, y = loadDataset(path_dataset)


for g in Gs:
    compute_extended_labels(g, label_node="label")

rings_andor_fw = "sans_rings_sans_fw"
node_label = "extended_label"  # -> parametre
edge_label = "bond_type"  # parametre
node_labels, nb_edge_labels = build_node_dictionnary(
    Gs, node_label, edge_label)
nb_labels = len(node_labels)

model = GedLayer(nb_labels, nb_edge_labels, node_labels, rings_andor_fw, normalize=True,
                 node_label=node_label)


@pytest.fixture(scope="function")
def a_graph():
    return Gs[12]


def test_ged_0(a_graph):
    """
    test si la ged d'un graphe avec lui meme est 0
    """
    cns, cndl, ces, cedl = model.from_weights_to_costs()
    A, l = from_networkx_to_tensor(a_graph, model.dict_nodes, model.node_label, model.device)
    n = a_graph.order()
    C = model.construct_cost_matrix(A, A, [n, n], [l, l], cns, ces, cndl, cedl)
    v = torch.flatten(torch.eye(n+1))

    c = torch.diag(C)
    D = C - torch.eye(C.shape[0], ) * c
    ged = (.5 * v.T @ D @ v + c.T @ v)

    assert (ged == 0)


def test_all_ged():
    for a_graph in Gs:
        cns, cndl, ces, cedl = model.from_weights_to_costs()
        A, l = from_networkx_to_tensor(a_graph, model.dict_nodes, model.node_label, model.device)
        n = a_graph.order()
        C = model.construct_cost_matrix(A, A, [n,n], [l,l], cns, ces, cndl, cedl)
        v = torch.flatten(torch.eye(n + 1))

        c = torch.diag(C)
        D = C - torch.eye(C.shape[0], ) * c
        ged = (.5 * v.T @ D @ v + c.T @ v)
        assert(ged == 0)
