from deepged.ged import Ged
from deepged.inference import evaluate_D
import matplotlib.gridspec as gridspec
import os
import sys
import pickle as pkl
import torch
import GPUtil
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import argparse
from datetime import datetime
from gklearn.utils.graphfiles import loadDataset
from gklearn.dataset import TUDataset_META
from gklearn.dataset import Dataset

from deepged.learning import learn_costs_for_classification
from deepged.label_manager import compute_extended_labels, build_node_dictionnary
from deepged.model import GedLayer
from deepged.dataset import dataset_split
import numpy as np
matplotlib.use('TkAgg')


def create_save_directory():
    '''
    Create a new directory to save data about current run
    '''
    default_directory = "save_runs"
    if(not os.path.isdir(default_directory)):
        os.mkdir(default_directory)
    time_stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path = os.path.join(default_directory, time_stamp)
    os.mkdir(path)
    return path


def plot_labels(Gs, node_label, edge_label, dict_nodes, set_edge_label):
    print(dict_nodes)
    print(set_edge_label)
    nb_node = np.zeros(len(dict_nodes))
    nb_edge = np.zeros(len(set_edge_label))
    for G in Gs:
        A = nx.to_scipy_sparse_matrix(
            G, dtype=int, weight=edge_label).todense()
        lab = [dict_nodes[G.nodes[v][node_label]] for v in nx.nodes(G)]
        for i in lab:
            nb_node[i] += 1
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    nb_edge[A[i, j]-1] += 1
    print(nb_node)
    print(nb_edge)
    plt.subplot(211)
    # , [i for i in range(len(nb_node))])
    plt.bar([i for i in range(len(nb_node))], nb_node)
    plt.title("node_labels")
    plt.subplot(212)
    # , [i for i in range(len(nb_node))])
    plt.bar([i for i in range(len(nb_edge))], nb_edge)
    plt.title("edge labels")
    plt.show()


def visualize(cost_ins_del, cost_node_sub, cost_edge_sub,
              loss_train, loss_valid,
              directory, verbosity=True):
    """
    Plot l'évolution des couts ainsi que la loss
    """
    p = plt.rcParams
    p["figure.figsize"] = 7, 7
    p["font.sans-serif"] = ["Roboto Condensed"]
    p["font.weight"] = "light"
    p["ytick.minor.visible"] = True
    p["xtick.minor.visible"] = True
    p["axes.grid"] = True
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    fig = plt.figure(constrained_layout=True, figsize=(18, 6))  #
    nrows, ncols = 2, 4
    gspec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    ax = plt.subplot(gspec[:, :2])
    color = 'tab:red'
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Train set loss value", color=color)
    ax.plot(loss_train, label="Loss on train set", color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax_2 = ax.twinx()
    color = 'tab:blue'
    ax_2.plot(loss_valid, label="Loss on validation set", color=color)
    ax_2.tick_params(axis='y', labelcolor=color)
    ax_2.set_ylabel("Validation set loss value", color=color)
    # ax_2.legend()
    ax.set_title("Losses", family="Roboto", weight=500)

    ax = plt.subplot(gspec[0, 2])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Costs")
    ax.plot(cost_ins_del[:, 0], label="node ins/del cost ")
    ax.plot(cost_ins_del[:, 1], label="edge ins/del cost ")
    ax.legend()
    ax.set_title("Ins/Del costs", family="Roboto", weight=500)

    ax = plt.subplot(gspec[0, 3])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Costs")
    ax.plot(cost_edge_sub)
    ax.set_title("Edge sub costs", family="Roboto", weight=500)

    ax = plt.subplot(gspec[1, 2:])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Costs")
    ax.plot(cost_node_sub)
    ax.set_title("Node sub costs", family="Roboto", weight=500)

    if (verbosity):
        plt.show()
    if directory is not None:
        fig.savefig(os.path.join(directory, "plot.pdf"))


def save_data(directory, loss_valid, loss_train,
              cost_ins_del, cost_edge_sub, cost_node_sub,
              args, D_train, D_test, clf, perfs):
    """
    Sauvegarde l'ensemble du learning aisni que les poids optimisés
    TODO : Code a factoriser !
    """

    torch.save(loss_valid, os.path.join(
        directory, "loss_valid"), pickle_module=pkl)
    torch.save(loss_train, os.path.join(
        directory, "loss_train"), pickle_module=pkl)

    # We save the costs into pickle files
    torch.save(cost_ins_del, os.path.join(
        directory, "cost_ins_del"), pickle_module=pkl)
    torch.save(cost_edge_sub, os.path.join(
        directory, "cost_edge_sub"), pickle_module=pkl)
    torch.save(cost_node_sub, os.path.join(
        directory, "cost_node_sub"), pickle_module=pkl)

    with open(os.path.join(directory, "arguments.txt"), "w") as f:
        f.write(args)

    pickle_filename = os.path.join(directory, "data_prediction.pkl")
    pkl.dump([D_train, D_test, clf, perfs], open(pickle_filename, "wb"))


def load_dataset(dataset_path):
    '''
    Returns NetworkX graphs and its targets according to given path.
    Special path includes names described at https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
    '''
    if dataset_path in TUDataset_META.keys():
        ds = Dataset(dataset_path)
        return ds.graphs, ds.targets
    else:
        return loadDataset(dataset_path)


def parse_arguments_main():
    '''
    Check all arguments given to the executable file to configure learning
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the dataset. Special values are dataset names included in https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets', type=str)
    parser.add_argument(
        "-v", '--verbosity', help="Print differents informations on the model", action="store_true", default=False)
    parser.add_argument("-d",
                        '--device', help="Device to use : CPU/GPU", choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('-n', '--normalize',
                        help='Enable normalization', action='store_true', default=True)

    parser.add_argument("-c",
                        '--calculation', help='Select the calculation method : Rings only (0) / Frank Wolfe only (1) / both (2) / none (3) / random (4) / default (5)', type=int, choices=[0, 1, 2, 3, 4, 5], default=3)
    parser.add_argument("-ln", '--label_node', help='Labels for the nodes. Depends on the dataset file',
                        nargs='?', type=str, default='extended_label')
    parser.add_argument("-le", '--label_edge', help='Labels for the edges. Depends on the dataset file',
                        nargs='?', type=str, default='bond_type')
    parser.add_argument('--nb_epochs', help="Nb of epochs",
                        type=int, default=50)
    parser.add_argument('--constraint', help="Policy concerning constraints",
                        choices=['no_constraint', 'add_to_loss', 'projection'], type=str, default='no_constraint')
    parser.add_argument('--extended_label', help="if set, use of extended labels. Label used is label_node",
                        action='store_true', default=False)
    parser.add_argument(
        "--size_batch", help="Number of pairs of Graphs, for each batch. Default : 1 batch per epoch", type=int, default=None)
    parser.add_argument(
        "--size_train", help="Proportion of dataset used for training. Remaining data is used for testset. Default : 0.7", type=float, default=0.7)
    args = parser.parse_args()
    return args


def run(args):

    # Configuraiton du modele
    dico_device = {"cpu": 'cpu', 'gpu': 'cuda:0'}
    dico_calc = {0: 'rings_sans_fw', 1: 'sans_rings_avec_fw',
                 2: 'rings_avec_fw', 3: 'sans_rings_sans_fw'}
    rings_andor_fw = dico_calc[args.calculation]
    device = dico_device[args.device]
    nb_epochs = args.nb_epochs
    path_dataset = args.path
    size_batch = args.size_batch
    train_size = args.size_train
    constraint = args.constraint

    # Init dataset
    if (args.verbosity):
        print(f"Paramètres: {args}")

    Gs, y = load_dataset(path_dataset)

    node_label = args.label_node
    edge_label = args.label_edge
    # Utile pour rings ? du coup on a un cout pour chaque extended_label
    if(args.extended_label):
        for g in Gs:
            compute_extended_labels(
                g, label_node=node_label, label_edge=edge_label)
        node_label = 'extended_label'

    node_labels, nb_edge_labels = build_node_dictionnary(
        Gs, node_label, edge_label)
    nb_labels = len(node_labels)

    model = GedLayer(nb_labels, nb_edge_labels, node_labels, rings_andor_fw,
                     normalize=args.normalize,
                     node_label=node_label,
                     device=dico_device[args.device])

    # Getting the GPU status :
    if(args.verbosity and args.device == 'gpu'):
        GPUtil.showUtilization()

    # Preparation of dataset
    # TODO -> mettre dans une fonction + dataclasses
    train_set, test_set = dataset_split(
        Gs, y, train_size=train_size, test_size=1-train_size, shuffle=True)
    indices_train, labels_train = train_set
    indices_test, labels_test = test_set
    graphs_train = [Gs[i] for i in indices_train]
    graphs_test = [Gs[i] for i in indices_test]
    y_train = [y[i] for i in indices_train]
    y_test = [y[i] for i in indices_test]

    if(args.calculation <= 3):
        # Learn with rings_andor_fw configuration
        cost_ins_del, cost_node_sub, \
            cost_edge_sub, loss_valid, \
            loss_train = learn_costs_for_classification(
                model, graphs_train, nb_epochs, device, labels_train, rings_andor_fw,
                verbosity=args.verbosity,
                size_batch=size_batch, constraint=constraint)
        # print(cost_ins_del, cost_node_sub, cost_edge_sub)
        best_epoch = np.argmin(loss_valid)
        if(args.verbosity):
            print(f"best_epoch : {best_epoch}")
        costs = [cost_node_sub[best_epoch, :], cost_ins_del[best_epoch, 0].reshape(-1, 1),
                 cost_edge_sub[best_epoch, :], cost_ins_del[best_epoch, 0].reshape(-1, 1)]
        if(args.verbosity):
            print(loss_train, loss_valid)

    elif(args.calculation >= 4):
        # Poids random initiaux ou par défaut
        node_costs, node_ins_del, edge_costs, edge_ins_del = model.from_weights_to_costs()
        k = 0
        node_sub = np.empty(
            (1, int(node_costs.shape[0] * (node_costs.shape[0] - 1) / 2)))
        edge_sub = np.empty(
            (1, int(edge_costs.shape[0] * (edge_costs.shape[0] - 1) / 2)))
        for p in range(node_costs.shape[0]):
            for q in range(p + 1, node_costs.shape[0]):
                node_sub[0][k] = node_costs[p][q]
                k = k + 1
        k = 0
        for p in range(edge_costs.shape[0]):
            for q in range(p + 1, edge_costs.shape[0]):
                edge_sub[0][k] = edge_costs[p][q]
                k = k + 1
        cost_ins_del = np.empty((1, 2))
        cost_ins_del[0, 0] = node_ins_del
        cost_ins_del[0, 1] = edge_ins_del

        if(rings_andor_fw == 4):
            # Poids random initiaux
            costs = [node_sub[0, :], cost_ins_del[0, 0].reshape(-1, 1),
                     edge_sub[0, :], cost_ins_del[0, 0].reshape(-1, 1)]

        elif(rings_andor_fw == 5):
            # Poids par défault
            node_sub[-1, :] = 1
            cost_ins_del[-1, 0] = 2
            edge_sub[-1, :] = 1
            cost_ins_del[-1, 1] = 3

            costs = [node_sub[-1, :], cost_ins_del[-1, 0].reshape(-1, 1),
                     edge_sub[-1, :], cost_ins_del[-1, 1].reshape(-1, 1)]

    ged = Ged(costs, node_labels, nb_edge_labels, node_label)
    # compute ged between train and test
    D_train = ged.compute_distance_between_sets(
        graphs_train, graphs_train, args.verbosity)
    D_test = ged.compute_distance_between_sets(
        graphs_test, graphs_train, args.verbosity)
    perf_train, perf_test, clf = evaluate_D(
        D_train, y_train, D_test, y_test, mode='classif')
    # We save everything
    # Sauvegarde du modele
    directory = create_save_directory()

    visualize(cost_ins_del, cost_node_sub, cost_edge_sub,
              loss_train, loss_valid,
              directory, args.verbosity)

    save_data(directory, loss_valid, loss_train, cost_ins_del, cost_edge_sub,
              cost_node_sub, repr(args), D_train, D_test, clf, [perf_train, perf_test])

    return perf_train, perf_test


if __name__ == "__main__":
    args = parse_arguments_main()
    perf_train, perf_test = run(args)
    print(perf_train, perf_test)
