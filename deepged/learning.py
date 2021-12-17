import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime


from deepged.triangular_losses import TriangularConstraint as triangular_constraint
from deepged.data_manager.data_split import splitting


def normalize(ged):
    '''
    Normalise la GED entre 0 et 1 pour la hinge Loss
    '''
    max_ged = torch.max(ged)
    min_ged = torch.min(ged)
    ged = (ged - min_ged) / (max_ged - min_ged)
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
            ged_pred[k] = model((Gs[g1_idx], Gs[g2_idx]))

    ged_pred = normalize(ged_pred)

    return ged_pred, labels


def tensorboardExport(writer, epoch, train_loss, valid_loss, node_ins_del, edge_ins_del, node_costs, edge_costs):
    # Sauvegarde de la loss dans le tensorboard
    writer.add_scalar('Evolution of the loss/train', train_loss, epoch)
    # Sauvegarde de la loss dans le tensorboard
    writer.add_scalar('Evolution of the loss/validation',  valid_loss, epoch)

    # Sauvegarde des couts d'insertion/ deletion dans le tensorboard
    writer.add_scalars('Costs evolution/Node.Edge insertion.deletion costs', {'node': node_ins_del,
                                                                              'edge': edge_ins_del}, epoch)

    data = {}
    k = 0
    for p in range(edge_costs.shape[0]):
        for q in range(p + 1, edge_costs.shape[0]):
            data["poids_"+str(k)] = edge_costs[p][q]
            k += 1
    writer.add_scalars('Costs evolution/Edge Substitutions costs', data, epoch)

    data = {}
    k = 0
    for p in range(node_costs.shape[0]):
        for q in range(p + 1, node_costs.shape[0]):
            data["poids_"+str(k)] = node_costs[p][q]
            k += 1
    writer.add_scalars('Costs evolution/Node Substitutions costs', data, epoch)

    writer.flush()


def GEDclassification(model, Gs, nb_epochs, device, y, rings_andor_fw, verbosity=True):
    """
    Run nb_epochs epochs pour fiter les couts de la ged

    TODO : function trop longue, à factoriser
    """

    trainloader, validationloader, test_loader = splitting(
        Gs, y, saving_path=rings_andor_fw, already_divided=False)

    now = datetime.now()
    writer = SummaryWriter("runs/data_" + now.strftime("%d-%m_%H-%M-%S"))

    #criterion = torch.nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
    criterion = torch.nn.HingeEmbeddingLoss()
    criterion_tri = triangular_constraint()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # , lr=1e-3

    node_costs, nodeInsDel, edge_costs, edge_ins_del = model.from_weights_to_costs()
    # TODO ; a documenter et mettre dansu ne fonction
    ins_del = np.empty((nb_epochs, 2))
    node_sub = np.empty(
        (nb_epochs, int(node_costs.shape[0] * (node_costs.shape[0] - 1) / 2)))
    edge_sub = np.empty(
        (nb_epochs, int(edge_costs.shape[0] * (edge_costs.shape[0] - 1) / 2)))

    loss_train = np.empty(nb_epochs)
    loss_valid = np.empty(nb_epochs)
    min_valid_loss = np.inf
    iter_min_valid_loss = 0

    for epoch in range(nb_epochs):
        current_train_loss = 0.0
        current_valid_loss = 0.0
        # Learning step
        ged_pred, train_labels = forward_data_model(
            trainloader, model, Gs, device)
        loss = criterion(ged_pred, train_labels)
        node_costs, node_ins_del, edge_costs, edge_ins_del = model.from_weights_to_costs()

        triangular_inequality = criterion_tri(
            node_costs, node_ins_del, edge_costs, edge_ins_del)
        loss = loss * (1 + triangular_inequality)

        # Mise à jour des couts
        ins_del[epoch][0] = node_ins_del.item()
        ins_del[epoch][1] = edge_ins_del.item()

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

        loss.backward()
        if(verbosity):
            print('grad of node weighs', model.node_weights.grad)
            print('grad of edge weighs', model.edge_weights.grad)
        optimizer.step()
        optimizer.zero_grad()

        current_train_loss = loss.item()
        loss_train[epoch] = current_train_loss

        # Fin for Batch
        if epoch == 0:
            # TODO : faire une structure pour les couts également
            save_costs(node_ins_del, edge_ins_del,
                       node_costs, edge_costs, rings_andor_fw, "init")

        # Getting some information every 100 iterations, to follow the evolution
        # The validation part :
        with torch.no_grad():
            ged_pred, valid_labels = forward_data_model(
                validationloader, model, Gs, device)
            current_valid_loss = criterion(ged_pred, valid_labels).item()
            loss_valid[epoch] = current_valid_loss

        if (verbosity):
            print(
                f'Iteration {epoch + 1} \t\t Training Loss: {loss_train[epoch]}')
            print(f"loss.item of the valid={current_valid_loss}")

        tensorboardExport(writer, epoch, current_train_loss, current_valid_loss,
                          node_ins_del.item(), edge_ins_del.item(), node_costs, edge_costs)

        tensorboardExport(writer, epoch, current_train_loss, current_valid_loss,
                          node_ins_del.item(), edge_ins_del.item(), node_costs, edge_costs)
        # Fermeture du tensorboard
        writer.close()
        # Enregistrement du min
    return ins_del, node_sub, edge_sub,  loss_valid, loss_train
