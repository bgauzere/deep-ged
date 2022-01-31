import os
import sys
import pickle as pkl
import torch
import GPUtil
import matplotlib.pyplot as plt
import matplotlib
import argparse
from datetime import datetime
from gklearn.utils.graphfiles import loadDataset

from deepged.learning import GEDclassification
from deepged.data_manager.label_manager import compute_extended_labels, build_node_dictionnary
from deepged.model import GedLayer
from deepged.utils import from_networkx_to_tensor
matplotlib.use('TkAgg')


def visualize(cost_ins_del, cost_node_sub, cost_edge_sub,  loss_train, loss_valid):
    """
    Plot l'évolution des couts ainsi que la loss
    """
    # plt.figure(fig
    # Plotting Node/Edge insertion/deletion costs
    plt.figure(0)
    plt.plot(InsDel[0:nb_iter, 0], label="node")
    plt.plot(InsDel[0:nb_iter, 1], label="edge")
    plt.title('Node/Edge insertion/deletion costs')
    plt.legend()

    # Plotting Node Substitutions
    # costs
    plt.figure(1)
    for k in range(nodeSub.shape[1]):
        plt.plot(nodeSub[0:nb_iter, k])
    plt.title('Node Substitutions costs')

    # Plotting Edge Substitutions costs
    plt.figure(2)
    for k in range(edgeSub.shape[1]):
        plt.plot(edgeSub[0:nb_iter, k])
    plt.title('Edge Substitutions costs')

    # Plotting the evolution of the train loss
    plt.figure(3)
    plt.title('Evolution of the train loss')
    plt.plot(loss_train_plt)

    # Plotting the evolution of the validation loss
    plt.figure(4)
    plt.plot(loss_valid_plt)
    plt.title('Evolution of the valid loss')

    plt.show()
    plt.close()


def save_data(loss_valid_plt, loss_train_plt,
              ins_del, edge_sub, node_sub,  args, directory=None):
    """
    Sauvegarde l'ensemble du learning aisni que les poids optimisés
    """
    if (directory is None):
        default_directory = "save_runs"
        if(not os.path.isdir(default_directory)):
            os.mkdir(default_directory)
        time_stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        path = os.path.join(default_directory, time_stamp)
        os.mkdir(path)
        directory = path
    else:
        if (not os.path.isdir(directory)):
            raise FileNotFoundError
    breakpoint()

    torch.save(loss_valid_plt, os.path.join(
        directory, "loss_valid_plt"), pickle_module=pkl)
    torch.save(loss_train_plt, os.path.join(
        directory, "loss_train_plt"), pickle_module=pkl)

    # We save the costs into pickle files
    torch.save(InsDel, os.path.join(
        directory, "cost_ins_del"), pickle_module=pkl)
    torch.save(edge_sub, os.path.join(
        directory, "cost_edge_sub"), pickle_module=pkl)
    torch.save(node_sub, os.path.join(
        directory, "cost_node_sub"), pickle_module=pkl)
    with open(os.path.join(directory, "arguments.txt"), "w") as f:
        f.write(args)


if __name__ == "__main__":

    dico_device = {"cpu": 'cpu', 'gpu': 'cuda:0'}
    dico_calc = {0: 'rings_sans_fw', 1: 'sans_rings_avec_fw',
                 2: 'rings_avec_fw', 3: 'sans_rings_sans_fw'}
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the dataset', type=str)
    parser.add_argument(
        "-v", '--verbosity', help="Print differents informations on the model", action="store_true", default=False)
    parser.add_argument("-d",
                        '--device', help="Device to use : CPU/GPU", choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('-n', '--normalize',
                        help='Enable normalization', action='store_true', default=True)

    parser.add_argument("-c",
                        '--calculation', help='Select the calculation method : Rings only (0) / Frank Wolfe only (1) / both (2) / none (3) ', type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("-ln", '--label_node', help='Labels for the nodes. Depends on the dataset file',
                        nargs='?', type=str, default='extended_label')
    parser.add_argument("-le", '--label_edge', help='Labels for the edges. Depends on the dataset file',
                        nargs='?', type=str, default='bond_type')
    args = parser.parse_args()

    # Configuraiton du modele
    rings_andor_fw = dico_calc[args.calculation]
    device = dico_device[args.device]
    nb_epochs = 50

    # Init dataset
    path_dataset = args.path
    if (args.verbosity):
        print(f"Paramètres: {args}")

    Gs, y = loadDataset(path_dataset)

    # Utile pour rings ? du coup on a un cout pour chaque extended_label
    for g in Gs:
        compute_extended_labels(g, label_node="label")

    node_label = args.label_node
    edge_label = args.label_edge
    node_labels, nb_edge_labels = build_node_dictionnary(
        Gs, node_label, edge_label)
    nb_labels = len(node_labels)
    model = GedLayer(nb_labels, nb_edge_labels, node_labels, rings_andor_fw, normalize=args.normalize,
                     node_label=node_label)

    # Getting the GPU status :
    if(args.verbosity):
        GPUtil.showUtilization()

    cost_ins_del, cost_node_sub, cost_edge_sub, loss_valid, loss_train = GEDclassification(
        model, Gs, nb_epochs, device, y, rings_andor_fw, verbosity=args.verbosity)

    if(args.verbosity):
        print(loss_train, loss_valid)
        visualize(cost_ins_del, cost_node_sub,
                  cost_edge_sub, loss_train, loss_valid)
    # We save the losses into pickle files
    save_data(loss_valid_plt, loss_train_plt, InsDel, edgeSub,
              nodeSub, repr(args))
