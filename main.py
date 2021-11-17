from data_manager.label_manager import *
from gklearn.utils.graphfiles import loadDataset
import torch
import GPUtil
from layers.layer import Net, GedLayer
import matplotlib.pyplot as plt
import matplotlib
from graph_torch.ged_torch import build_node_dictionnary
matplotlib.use('TkAgg')
from training.train import classification, GEDclassification
import pickle as pkl
# Loading the dataset :

Gs, y = loadDataset('DeepGED/MAO/dataset.ds')
# Getting the GPU status :
GPUtil.showUtilization()

print("Length of Gs = ", len(Gs))
# print('edge max label',max(max([[G[e[0]][e[1]]['bond_type'] for e in G.edges()] for G in Gs])))

for g in Gs:
    compute_extended_labels(g)

rings_andor_fw = "sans_rings_avec_fw"

device = 'cpu'
# node_dict, nb_edge_labels = build_node_dictionnary(Gs, "extended_label", "bond_type")
# model = GedLayer(rings_andor_fw, node_label='extended_label', node_label_dict=node_dict, nb_edge_label=nb_edge_labels)
model = Net(Gs,normalize=True,node_label='extended_label')
model.to(device)


print(len(Gs))
nb_iter=100

InsDel, nodeSub,edgeSub,loss_plt,loss_valid_plt,loss_train_plt=classification(model,Gs,nb_iter, device, y, rings_andor_fw)

# Plotting Node/Edge insertion/deletion costs
plt.figure(0)
plt.plot(InsDel[0:nb_iter,0],label="node")
plt.plot(InsDel[0:nb_iter,1],label="edge")
plt.title('Node/Edge insertion/deletion costs')
plt.legend()

# Plotting Node Substitutions
# costs
plt.figure(1)
for k in range(nodeSub.shape[1]):
    plt.plot(nodeSub[0:nb_iter,k])
plt.title('Node Substitutions costs')

# Plotting Edge Substitutions costs
plt.figure(2)
for k in range(edgeSub.shape[1]):
    plt.plot(edgeSub[0:nb_iter,k])
plt.title('Edge Substitutions costs')

# Plotting the evolution of the train loss
plt.figure(3)
plt.plot(loss_plt)
plt.title('Evolution of the train loss (loss_plt)')

# Plotting the evolution of the validation loss
plt.figure(4)
plt.plot(loss_valid_plt)
plt.title('Evolution of the valid loss')

plt.show()
plt.close()

# We save the losses into pickle files
torch.save(loss_plt, 'pickle_files/'+rings_andor_fw+'/loss_plt', pickle_module=pkl)
torch.save(loss_valid_plt, 'pickle_files/'+rings_andor_fw+'/loss_valid_plt', pickle_module=pkl)
torch.save(loss_train_plt, 'pickle_files/'+rings_andor_fw+'/loss_train_plt', pickle_module=pkl)

# We save the costs into pickle files
torch.save(InsDel,'pickle_files/'+rings_andor_fw+'/InsDel', pickle_module=pkl)
torch.save(edgeSub,'pickle_files/'+rings_andor_fw+'/edgeSub', pickle_module=pkl)
torch.save(nodeSub,'pickle_files/'+rings_andor_fw+'/nodeSub', pickle_module=pkl)