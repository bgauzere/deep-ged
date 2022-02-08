import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from deepged.triangular_losses import TriangularConstraint
# from deepged.data_manager.data_split import splitting
from deepged.dataset import initialize_dataset


def normalize(ged):
    '''
    Normalise la GED entre 0 et 1 pour la hinge Loss
    '''
    max_ged = torch.max(ged)
    min_ged = torch.min(ged)
    ged = (ged - min_ged) / (max_ged - min_ged)
    return ged


def forward_data_model(data, model, Gs, device):
    '''Effectue une passe forward d'un loader (train, valid ou test)
    et renvoie l'ensemble des ged et des labels si meme classe ou
    non

    :param loader: le loader utilisé
    :param model: le modèle utilisé
    :param Gs: l'ensemble des graphes sous forme de
    liste
    :param device:device utilisé (cpu ou gpu)
    :return: l'ensemble des prédictions, ainsi
    que les true_labels
    '''
    ged_pred = torch.zeros(len(data))
    # Forward pass: Compute predicted y by passing data to the model
    for k in tqdm(range(len(data))):
        g1_idx, g2_idx = data[k]
        ged_pred[k] = model((Gs[g1_idx], Gs[g2_idx]))

    ged_pred = normalize(ged_pred)

    return ged_pred


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


def GEDclassification(model, Gs, nb_epochs, device, y, rings_andor_fw,
                      verbosity=True, learning_rate=0.01,
                      size_batch=None):
    """ Run nb_epochs epochs pour fiter les couts de la ged
    TODO : function trop longue, à factoriser
    """
    train_loader, valid_loader, test_loader = initialize_dataset(
        Gs, y, size_batch_train=size_batch)

    criterion = torch.nn.HingeEmbeddingLoss(reduction='sum')
    criterion_tri = TriangularConstraint()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    node_costs, node_ins_del, edge_costs, edge_ins_del = model.from_weights_to_costs()

    # TODO ; a documenter et mettre dansu ne fonction
    # Pour sauvegarde :
    now = datetime.now()
    writer = SummaryWriter("runs/data_" + now.strftime("%d-%m_%H-%M-%S"))

    ins_del = np.empty((nb_epochs, 2))
    node_sub = np.empty(
        (nb_epochs, int(node_costs.shape[0] * (node_costs.shape[0] - 1) / 2)))
    edge_sub = np.empty(
        (nb_epochs, int(edge_costs.shape[0] * (edge_costs.shape[0] - 1) / 2)))

    loss_train = np.empty(nb_epochs)
    loss_valid = np.empty(nb_epochs)

    for epoch in range(nb_epochs):
        current_train_loss = 0.0
        nb_train = 0
        for data, labels in train_loader:
            # Learning step
            nb_train += len(data)
            # TODO : du coup train_labels devrait être inutile
            ged_pred = forward_data_model(
                data, model, Gs, device)
            loss = criterion(ged_pred, labels)
            triangular_inequality = criterion_tri(
                node_costs, node_ins_del, edge_costs, edge_ins_del)
            loss = loss * (1 + triangular_inequality)
            current_train_loss += loss
            node_costs, node_ins_del, edge_costs, edge_ins_del = model.from_weights_to_costs()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Fin for Batch
        # if(verbosity):
        #     print('grad of node weighs', model.node_weights.grad)
        #     print('grad of edge weighs', model.edge_weights.grad)

        # Mise à jour des couts
        ins_del[epoch][0] = node_ins_del.item()
        ins_del[epoch][1] = edge_ins_del.item()

        # Sauvegarde des couts
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

        current_train_loss = current_train_loss.item()
        loss_train[epoch] = current_train_loss

        # The validation part :
        current_valid_loss = 0.0
        nb_valid = 0
        for data, labels in valid_loader:
            nb_valid += len(data)
            with torch.no_grad():
                ged_pred = forward_data_model(
                    data, model, Gs, device)
                loss = criterion(ged_pred, labels).item()
                loss += criterion_tri(node_costs, node_ins_del,
                                      edge_costs, edge_ins_del)
                loss = loss * (1 + triangular_inequality)

                current_valid_loss += loss
        loss_valid[epoch] = current_valid_loss

        if (verbosity):
            print(
                f"Iteration {epoch + 1} \t\t Training Loss: {current_train_loss} - {current_train_loss/nb_train}")
            print(
                f"loss.item of the valid={current_valid_loss} - {current_valid_loss/nb_valid}")
        # Sauvegarde
        tensorboardExport(writer, epoch, current_train_loss, current_valid_loss,
                          node_ins_del.item(), edge_ins_del.item(), node_costs, edge_costs)

        tensorboardExport(writer, epoch, current_train_loss, current_valid_loss,
                          node_ins_del.item(), edge_ins_del.item(), node_costs, edge_costs)
        # Fermeture du tensorboard
        writer.close()
    return ins_del, node_sub, edge_sub,  loss_valid, loss_train
