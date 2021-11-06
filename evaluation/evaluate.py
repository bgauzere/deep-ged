from gklearn.utils.graphfiles import loadDataset
from data_manager import DataSet
import math
import numpy as np
import networkx as nx

def classification(test_data, test_label, graphList, costs):
    """
    :param test_data: pytorch dataloader. Tuples of the form data[0] = pair of graph
    :param test_label: True class of each label
    :param graphList: The list of graphs
    :param costs: costs for GED. Can be the fixed one or the learned ones.
    :return:
    """

    print(len(test_data))
    n_graph = int(math.sqrt(len(test_data)*2)) # Compute the number of graph in the set
    ged_matrix = np.zeros((n_graph, n_graph))
    for data in test_data:
        g1 = graphList[data[0][0][0]]
        g2 = graphList[data[0][0][1]]
        print(type(g1))
        print(data[0][0][1])
        ged =  nx.graph_edit_distance(g1, g2,
                                      node_subst_cost=costs[0],
                                      node_del_cost=costs[1],
                                      node_ins_cost=costs[1],
                                      edge_subst_cost=costs[2],
                                      edge_ins_cost=costs[3],
                                      edge_del_cost=costs[3])


        ged_matrix[data[0][0][0], data[0][0][1]] = ged


def compute_ged(g1, g2, costs):
    C = ged_torch.construct_cost_matrix(g1, g2, costs[0], costs[1], costs[2], costs[3])


if __name__ == "__main__":
    Gs, y = loadDataset('../DeepGED/MAO/dataset.ds')
    batch_size = 1

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    trainloader, validationloader = DataSet.gen_loader(Gs, y, params)
    costs = [1, 1, 2, 2]
    classification(validationloader, y, Gs, costs)
