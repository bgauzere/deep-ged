import numpy as np
import torch
from losses.triangular_losses import TriangularConstraint as triangular_constraint
from data_manager.data_split import splitting
import pickle as pkl
from tqdm import tqdm

def classification(model, Gs, nb_iter, device, y):
    trainloader, validationloader, couples_train, yt, couples_test_train, yv = splitting(Gs, y)
    criterion = torch.nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
    criterionTri = triangular_constraint()
    optimizer = torch.optim.Adam(model.parameters())  # , lr=1e-3

    train_input = couples_train.to(device)
    # valid_input=couples_test_train.to(device)

    target = yt.to(device)
    InsDel = np.empty((nb_iter, 2))
    node_costs, nodeInsDel, edge_costs, edgeInsDel = model.from_weighs_to_costs()
    nodeSub = np.empty((nb_iter, int(node_costs.shape[0] * (node_costs.shape[0] - 1) / 2)))
    edgeSub = np.empty((nb_iter, int(edge_costs.shape[0] * (edge_costs.shape[0] - 1) / 2)))
    loss_plt = np.empty(nb_iter)
    loss_train_plt = np.empty(nb_iter)
    loss_valid_plt = np.empty(nb_iter)
    min_valid_loss = np.inf
    iter_min_valid_loss = 0

    for t in tqdm(range(nb_iter)):
        print(t, torch.cuda.memory_allocated())
        train_loss = 0.0
        valid_loss = 0.0
        tmp = np.inf

        # The training part :
        for train_data, train_labels in trainloader:
            print(type(train_data))
            print(len(train_data))
            print(train_data)
            print(train_labels)
            # print(t, torch.cuda.memory_allocated())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # inputt=train_data.to(device)

            # Forward pass: Compute predicted y by passing data to the model
            y_pred = model(train_data).to(device)

            # Computing and printing loss
            train_labels = train_labels.to(device)
            loss = criterion(y_pred, train_labels).to(device)
            node_costs, nodeInsDel, edge_costs, edgeInsDel = model.from_weighs_to_costs()
            triangularInq = criterionTri(node_costs, nodeInsDel, edge_costs, edgeInsDel)
            loss = loss * (1 + triangularInq)
            loss.to(device)
            loss.backward()
            loss = loss.detach()

            optimizer.step()
            print('loss.item of the train = ', t, loss.item())
            train_loss = + loss.item()  # * train_data.size(0)
            if (loss.item() < tmp): tmp = loss.item()

        # Getting the training loss
        loss_plt[t] = loss.item()
        loss_train_plt[t] = train_loss / len(trainloader)
        # loss_plt[t]=tmp

        # Getting the costs of the first iteration, to compare later
        if t == 0:
            nodeInsDelInit = nodeInsDel
            edgeInsDelInit = edgeInsDel
            nodeSubInit = node_costs
            edgeSubInit = edge_costs
            torch.save(nodeInsDelInit, 'pickle_files/nodeInsDelInit', pickle_module=pkl)
            torch.save(edgeInsDelInit, 'pickle_files/edgeInsDelInit', pickle_module=pkl)
            torch.save(nodeSubInit, 'pickle_files/nodeSubInit', pickle_module=pkl)
            torch.save(edgeSubInit, 'pickle_files/edgeSubInit', pickle_module=pkl)

            # Getting some information every 100 iterations, to follow the evolution
        if t % 100 == 99 or t == 0:
            print('ged=', y_pred * target)  # train_labels
            print('Distances: ', y_pred)
            print('Loss Triangular:', triangularInq.item())
            print('node_costs : \n', node_costs)
            print('nodeInsDel:', nodeInsDel.item())
            print('edge_costs : \n', edge_costs)
            print('edgeInsDel:', edgeInsDel.item())

        print(f'Iteration {t + 1} \t\t Training Loss: {train_loss / len(trainloader)}')

        # We delete to liberate some memory
        del y_pred, train_loss, loss
        torch.cuda.empty_cache()

        # The validation part :
        for valid_data, valid_labels in validationloader:
            inputt = valid_data.to(device)
            y_pred = model(inputt).to(device)
            # Compute and print loss
            valid_labels = valid_labels.to(device)
            loss = criterion(y_pred, valid_labels).to(device)
            loss.to(device)
            print('loss.item of the valid = ', t, loss.item())
            valid_loss = valid_loss + loss.item()  # * valid_data.size(0)

        # Getting the validation loss
        loss_valid_plt[t] = valid_loss / len(validationloader)

        # Getting edges and nodes Insertion/Deletion costs
        InsDel[t][0] = nodeInsDel.item()
        InsDel[t][1] = edgeInsDel.item()

        k = 0
        for p in range(node_costs.shape[0]):
            for q in range(p + 1, node_costs.shape[0]):
                nodeSub[t][k] = node_costs[p][q]
                k = k + 1
        k = 0
        for p in range(edge_costs.shape[0]):
            for q in range(p + 1, edge_costs.shape[0]):
                edgeSub[t][k] = edge_costs[p][q]
                k = k + 1

        print(f'Iteration {t + 1} \t\t Validation Loss: {valid_loss / len(validationloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
            min_valid_loss = valid_loss
            iter_min_valid_loss = t
            nodeSub_min = node_costs
            edgeSub_min = edge_costs
            nodeInsDel_min = nodeInsDel
            edgeInsDel_min = edgeInsDel

        # We delete to liberate some memory
        del valid_loss, loss
        torch.cuda.empty_cache()

    print('iter and min_valid_loss = ', iter_min_valid_loss, min_valid_loss)
    print(' Min cost for nodeInsDel = ', nodeInsDel_min)
    print(' Min cost for edgeInsDel = ', edgeInsDel_min)
    print(' Min cost for nodeSub = ', nodeSub_min)
    print(' Min cost for edgeSub = ', edgeSub_min)
    # Saving the minimum costs into pickle files
    torch.save(nodeInsDel_min, 'pickle_files/nodeInsDel_min', pickle_module=pkl)
    torch.save(edgeInsDel_min, 'pickle_files/edgeInsDel_min', pickle_module=pkl)
    torch.save(nodeSub_min, 'pickle_files/nodeSub_min', pickle_module=pkl)
    torch.save(edgeSub_min, 'pickle_files/edgeSub_min', pickle_module=pkl)
    return InsDel, nodeSub, edgeSub, loss_plt, loss_valid_plt, loss_train_plt