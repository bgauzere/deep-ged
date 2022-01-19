import os
import sys
import pickle as pkl
import torch
import GPUtil
import matplotlib.pyplot as plt
import matplotlib
import argparse

from gklearn.utils.graphfiles import loadDataset
from sklearn.model_selection import train_test_split


from deepged.learning import GEDclassification
from deepged.data_manager.label_manager import compute_extended_labels, build_node_dictionnary
from deepged.model import GedLayer
from deepged.utils import from_networkx_to_tensor
from model_regression import RegressGedLayer


matplotlib.use('TkAgg')



if __name__ == "__main__":
    dico_device = {"cpu": 'cpu', 'gpu': 'cuda:0'}
    dico_calc = {0: 'rings_sans_fw', 1: 'sans_rings_avec_fw',
                 2: 'rings_avec_fw', 3: 'sans_rings_sans_fw'}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", '--verbosity', help="Print differents informations on the model", action="store_true")
    parser.add_argument(
        'device', help="Device to use : CPU/GPU", choices=['cpu', 'gpu'])
    parser.add_argument('-n', '--normalize',
                        help='Enable normalization', action='store_true')
    # parser.add_argument('path', help='Path to the dataset', type=str)
    parser.add_argument(
        'calculation', help='Select the calculation method : Rings only (0) / Frank Wolfe only (1) / both (2) / none (3) ', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('labelNode', help='Labels for the nodes. Depends on the dataset file',
                        nargs='?', type=str, default='label')
    parser.add_argument('labelEdge', help='Labels for the edges. Depends on the dataset file',
                        nargs='?', type=str, default='bond_type')
    args = parser.parse_args()


    Gs = []
    y= []
    for i in range(10):
        graphSet,ySet = loadDataset("data/Acyclic/trainset_"+str(i)+".ds")

        Gs = Gs + graphSet
        y = y + ySet


    Gs = Gs[:100]
    y = y[:100]

    # Configuraiton du modele
    rings_andor_fw = dico_calc[args.calculation]
    device = dico_device[args.device]
    nb_epochs = 5
    # Init dataset
    # path_dataset = args.path


    [train_graph, valid_graph, train_label, valid_label] = train_test_split(Gs, y, test_size=0.10, shuffle=True)









    for g in Gs:
        compute_extended_labels(g, label_node="label")

    node_label = args.labelNode
    edge_label = args.labelEdge
    node_labels, nb_edge_labels = build_node_dictionnary(
        Gs, node_label, edge_label)
    nb_labels = len(node_labels)


    model = RegressGedLayer(train_graph ,torch.tensor(train_label), 3,nb_labels, nb_edge_labels, node_labels, rings_andor_fw, normalize=args.normalize,
                     node_label=node_label)






    graph_idx = torch.arange(0, len(valid_graph), dtype=torch.int64)
    dataSet = torch.utils.data.TensorDataset(torch.tensor(graph_idx), torch.tensor(valid_label))
    trainloader = torch.utils.data.DataLoader(dataSet, batch_size=8, shuffle=True, drop_last=True,
                                              num_workers=0)  # 128, len(couples_train)







    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1000):
        for graphs_idx, temp in trainloader:
            optimizer.zero_grad()

            pred_val = torch.zeros(len(graphs_idx))
             # forward






            with torch.set_grad_enabled(True):
                for i in range(len(graphs_idx)):

                    pred_val[i] =  model(valid_graph[graphs_idx[i]])

            loss  = criterion(pred_val ,temp)
            print(torch.nn.functional.l1_loss(pred_val ,temp))
            loss.backward()

            optimizer.step()
            # print(loss)









    # # Getting the GPU status :
    # if(args.verbosity):
    #     GPUtil.showUtilization()
    # InsDel, nodeSub, edgeSub, loss_valid_plt, loss_train_plt = GEDclassification(
    #     model, Gs, nb_epochs, device, y, rings_andor_fw, verbosity=args.verbosity)






    # if(args.verbosity):
    #     print(loss_train_plt, loss_valid_plt)
    #     visualize(InsDel, nb_epochs, nodeSub, edgeSub, loss_valid_plt)
    # # We save the losses into pickle files
    # save_data(loss_valid_plt, loss_train_plt, InsDel, edgeSub,
    #           nodeSub, rings_andor_fw)
