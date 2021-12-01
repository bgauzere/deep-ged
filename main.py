import os
import sys
import pickle as pkl
import torch
import GPUtil
import matplotlib.pyplot as plt
import matplotlib
import argparse

from gklearn.utils.graphfiles import loadDataset

from deepged.learning import GEDclassification
from deepged.data_manager.label_manager import compute_extended_labels, build_node_dictionnary
from deepged.model import GedLayer
from deepged.utils import from_networkx_to_tensor
matplotlib.use('TkAgg')


def visualize(InsDel, nb_iter, nodeSub, edgeSub,  loss_valid_plt):
    """
    Plot l'évolution des couts ainsi que la loss
    """
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
    plt.plot(loss_train_plt)
    plt.title('Evolution of the train loss')

    # Plotting the evolution of the validation loss
    plt.figure(4)
    plt.plot(loss_valid_plt)
    plt.title('Evolution of the valid loss')

    plt.show()
    plt.close()


def save_data(loss_valid_plt, loss_train_plt, InsDel, edgeSub,
              nodeSub, rings_andor_fw):
    """
    Sauvegarde l'ensemble du learning aisni que les poids optimisés
    """
    torch.save(loss_valid_plt, 'pickle_files/'+rings_andor_fw +
               '/loss_valid_plt', pickle_module=pkl)
    torch.save(loss_train_plt, 'pickle_files/'+rings_andor_fw +
               '/loss_train_plt', pickle_module=pkl)

    # We save the costs into pickle files
    torch.save(InsDel, 'pickle_files/'+rings_andor_fw +
               '/InsDel', pickle_module=pkl)
    torch.save(edgeSub, 'pickle_files/'+rings_andor_fw +
               '/edgeSub', pickle_module=pkl)
    torch.save(nodeSub, 'pickle_files/'+rings_andor_fw +
               '/nodeSub', pickle_module=pkl)


if __name__ == "__main__":
    dicoDevice = {"cpu": 'cpu', 'gpu': 'cuda:0'}
    dicoCalc = {0: 'rings_sans_fw', 1: 'sans_rings_avec_fw',
                2: 'rings_avec_fw', 3: 'sans_rings_sans_fw'}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", '--verbosity', help="Print differents informations on the model", action="store_true")
    parser.add_argument(
        'device', help="Device to use : CPU/GPU", choices=['cpu', 'gpu'])
    parser.add_argument('-n', '--normalize',
                        help='Enable normalization', action='store_true')
    parser.add_argument('path', help='Path to the dataset', type=str)
    parser.add_argument(
        'calculation', help='Select the calculation method : Rings only (0) / Frank Wolfe only (1) / both (2) / none (3) ', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('labelNode', help='Labels for the nodes. Depends on the dataset file',
                        nargs='?', type=str, default='label')
    parser.add_argument('labelEdge', help='Labels for the edges. Depends on the dataset file',
                        nargs='?', type=str, default='bond_type')
    args = parser.parse_args()

    # Configuraiton du modele
    rings_andor_fw = dicoCalc[args.calculation]
    device = dicoDevice[args.device]
    nb_epochs = 100
    # Init dataset
    path_dataset = args.path

    Gs, y = loadDataset(path_dataset)
    # Gs = Gs[:24]
    # y = y[:24]
    # Utile pour rings ? du coup on a un coup pour chaque extended_label

    for g in Gs:
        compute_extended_labels(g, label_node="label")

    node_label = args.labelNode
    edge_label = args.labelEdge
    node_labels, nb_edge_labels = build_node_dictionnary(
        Gs, node_label, edge_label)
    nb_labels = len(node_labels)
    model = GedLayer(nb_labels, nb_edge_labels, rings_andor_fw, normalize=args.normalize,
                     node_label=node_label)
    model.to(device)

    nb_epochs = 50
    InsDel, nodeSub, edgeSub, loss_valid_plt, loss_train_plt = GEDclassification(
        model, Gs, A, card, labels, nb_epochs, device, y, rings_andor_fw)

    print(loss_train_plt, loss_valid_plt)
    visualize(InsDel, nb_epochs, nodeSub, edgeSub, loss_valid_plt)
    # We save the losses into pickle files
    save_data(loss_valid_plt, loss_train_plt, InsDel, edgeSub,
              nodeSub, rings_andor_fw)
