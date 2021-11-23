import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm

from deepged.triangular_losses import TriangularConstraint as triangular_constraint
from deepged.data_manager.data_split import splitting


def normalize(ged):
    '''
    Normalise la GED entre 0 et 1 pour la hinge Loss
    '''
    max = torch.max(ged)
    min = torch.min(ged)
    ged = (ged - min) / (max - min)
    return ged


def save_costs(node_ins_del, edge_ins_del, node_costs, edge_costs, rings_andor_fw, identifier):
    '''
    Sauvegarde l'ensemble des couts sous forme de pickle files

    '''
    node_ins_del_init = node_ins_del
    edge_ins_del_init = edge_ins_del
    node_sub_init = node_costs
    edge_sub_init = edge_costs
    torch.save(node_ins_del_init, 'pickle_files/' +
               rings_andor_fw + '/node_ins_del_' + identifier, pickle_module=pkl)
    torch.save(edge_ins_del_init, 'pickle_files/' +
               rings_andor_fw + '/edge_ins_del_' + identifier, pickle_module=pkl)
    torch.save(node_sub_init, 'pickle_files/' +
               rings_andor_fw + '/node_sub_' + identifier, pickle_module=pkl)
    torch.save(edge_sub_init, 'pickle_files/' +
               rings_andor_fw + '/edge_sub_' + identifier, pickle_module=pkl)


def forward_data_model(loader, model, Gs, device):
    '''
    Effectue une passe forward d'un loader (train, valid ou test) et renvoie
    :param loader: le loader utilisé
    :param model: le modèle utilisé
    :param Gs: l'ensemble des graphes sous forme de liste
    :param device:device utilisé (cpu ou gpu)
    :return: l'ensemble des prédictions, ainsi que les true_labels
    '''

    for data, labels in loader:
        ged_pred = torch.zeros(len(data))
        # Zero gradients, perform a backward pass, and update the weights.
        # Forward pass: Compute predicted y by passing data to the model
        for k in tqdm(range(len(data))):
            g1_idx, g2_idx = data[k]
            ged_pred[k] = model((Gs[g1_idx], Gs[g2_idx])).to(device)

        ged_pred = normalize(ged_pred)
        # Computing and printing loss

    return ged_pred, labels.to(device)


def GEDclassification(model, Gs, A, card, labels, nb_epochs, device, y, rings_andor_fw):
    """
    Run 100 epochs pour fiter les couts de la ged

    TODO : function trop longue, à factoriser
    """

    trainloader, validationloader, test_loader = splitting(Gs, y, saving_path=rings_andor_fw, already_divided=True)

    criterion = torch.nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
    criterion_tri = triangular_constraint()
    optimizer = torch.optim.Adam(model.parameters())  # , lr=1e-3

    node_costs, nodeInsDel, edge_costs, edge_ins_del = model.from_weights_to_costs()
    # TODO ; a documenter et mettre dansu ne fonction
    ins_del = np.empty((nb_epochs, 2))
    node_sub = np.empty((nb_epochs, int(node_costs.shape[0] * (node_costs.shape[0] - 1) / 2)))
    edge_sub = np.empty((nb_epochs, int(edge_costs.shape[0] * (edge_costs.shape[0] - 1) / 2)))

    loss_train = np.empty(nb_epochs)
    loss_valid = np.empty(nb_epochs)
    min_valid_loss = np.inf
    iter_min_valid_loss = 0

    for epoch in range(nb_epochs):
        current_train_loss = 0.0
        current_valid_loss = 0.0

        # Learning step

        ged_pred, train_labels = forward_data_model(trainloader, model, Gs, device)
        loss = criterion(ged_pred, train_labels).to(device)
        node_costs, node_ins_del, edge_costs, edge_ins_del = model.from_weights_to_costs()
        triangular_inequality = criterion_tri(node_costs, node_ins_del, edge_costs, edge_ins_del)
        loss = loss * (1 + triangular_inequality)
        loss.to(device)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss.item of the train = ', epoch, loss.item())
        current_train_loss = current_train_loss + loss.item()  # * train_data.size(0)
            # Fin for Batch

        # Getting the training loss
        loss_train[epoch] = current_train_loss / len(trainloader)

        # Getting the costs of the first iteration, to compare later
        if epoch == 0:
            # TODO : faire une structure pour les couts également
            save_costs(node_ins_del, edge_ins_del,
                             node_costs, edge_costs, rings_andor_fw, "init")

        # Getting some information every 100 iterations, to follow the evolution
        # if epoch % 100 == 99 or epoch == 0:
        if True:
            print('Distances: ', ged_pred)
            # print('Loss Triangular:', triangular_inequality.item())
            print('node_costs : \n', node_costs)
            print('node_ins_del:', node_ins_del.item())
            print('edge_costs : \n', edge_costs)
            print('edge_ins_del:', edge_ins_del.item())

        print(
            f'Iteration {epoch + 1} \t\t Training Loss: {loss_train[epoch]}')

        # We delete to liberate some memory
        del ged_pred, current_train_loss, loss
        torch.cuda.empty_cache()

        # The validation part :
        ged_pred, valid_labels = forward_data_model(validationloader, model, Gs, device)

        loss = criterion(ged_pred, valid_labels).to(device)
        loss.to(device)
        print('loss.item of the valid = ', epoch, loss.item())
        current_valid_loss = current_valid_loss + loss.item()

        # Getting the validation loss
        loss_valid[epoch] = current_valid_loss / len(validationloader)
        # Getting edges and nodes Insertion/Deletion costs
        ins_del[epoch][0] = node_ins_del.item()
        ins_del[epoch][1] = edge_ins_del.item()
        # TODO : a pythoniser
        k = 0
        for p in range(node_costs.shape[0]):
            for q in range(p + 1, node_costs.shape[0]):
                node_sub[epoch][k] = node_costs[p][q]
                k = k + 1
        k = 0
        for p in range(edge_costs.shape[0]):
            for q in range(p + 1, edge_costs.shape[0]):
                edge_sub[epoch][k] = edge_costs[p][q]
                k = k + 1

        print(
            f'Iteration {epoch + 1} \t\t Validation Loss: {loss_valid[epoch]}')
        if min_valid_loss > loss_valid[epoch]:
            print(
                f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss_valid[epoch]:.6f})')
            min_valid_loss = loss_valid[epoch]
            iter_min_valid_loss = epoch
            node_sub_min = node_costs
            edge_sub_min = edge_costs
            node_ins_del_min = node_ins_del
            edge_ins_del_min = edge_ins_del

        # We delete to liberate some memory
        del current_valid_loss, loss
        # training.plot.plot("pickle_files/", rings_andor_fw)
        torch.cuda.empty_cache()

    print('iter and min_valid_loss = ', iter_min_valid_loss, min_valid_loss)
    print(' Min cost for node_ins_del = ', node_ins_del_min)
    print(' Min cost for edge_ins_del = ', edge_ins_del_min)
    print(' Min cost for node_sub = ', node_sub_min)
    print(' Min cost for edge_sub = ', edge_sub_min)
    # Saving the minimum costs into pickle files

    save_costs(node_ins_del_min, edge_ins_del_min,
               node_sub_min, edge_sub_min, rings_andor_fw, "min")

    return ins_del, node_sub, edge_sub,  loss_valid, loss_train
