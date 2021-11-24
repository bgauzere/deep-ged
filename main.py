import os
import sys
import pickle as pkl
import torch
import GPUtil
import matplotlib.pyplot as plt
import matplotlib

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
    verbose = True  # -> parametre
    # Configuraiton du modele
    rings_andor_fw = "sans_rings_sans_fw"  # -> parametre
    device = 'cpu'  # -> parametre
    normalize = True  # -> parametre
    nb_epochs = 15
    # Init dataset
    path_dataset = os.getenv('MAO_DATASET_PATH')  # -> parametre
    Gs, y = loadDataset(path_dataset)
    # Utile pour rings ? du coup on a un cout pour chaque extended_label

    for g in Gs:
        compute_extended_labels(g, label_node="label")

    node_label = "label"  # -> parametre
    edge_label = "bond_type"  # parametre
    node_labels, nb_edge_labels = build_node_dictionnary(
        Gs, node_label, edge_label)
    nb_labels = len(node_labels)

    # Getting the GPU status :
    GPUtil.showUtilization()

    model = GedLayer(nb_labels, nb_edge_labels, node_labels, rings_andor_fw, normalize=True,
                     node_label=node_label)
    # model.to(device)

    InsDel, nodeSub, edgeSub, loss_valid_plt, loss_train_plt = GEDclassification(
        model, Gs, nb_epochs, device, y, rings_andor_fw)

    print(loss_train_plt, loss_valid_plt)
    visualize(InsDel, nb_epochs, nodeSub, edgeSub, loss_valid_plt)
    # We save the losses into pickle files
    save_data(loss_valid_plt, loss_train_plt, InsDel, edgeSub,
              nodeSub, rings_andor_fw)
