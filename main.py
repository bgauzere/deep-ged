from data_manager.label_manager import *
from gklearn.utils.graphfiles import loadDataset
import torch
import GPUtil
from layers.layer import Net
import matplotlib.pyplot as plt
from training.train import classification
# Loading the dataset :

Gs, y = loadDataset('DeepGED/MAO/dataset.ds')
# Getting the GPU status :
GPUtil.showUtilization()

print("Length of Gs = ", len(Gs))
# print('edge max label',max(max([[G[e[0]][e[1]]['bond_type'] for e in G.edges()] for G in Gs])))

for g in Gs:
    compute_extended_labels(g)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = Net(Gs, normalize=False, node_label='extended_label')
model.to(device)

print(len(Gs))
nb_iter=100

InsDel, nodeSub,edgeSub,loss_plt,loss_valid_plt,loss_train_plt=classification(model,Gs,nb_iter, device, y)

# Plotting Node/Edge insertion/deletion costs
plt.figure(0)
plt.plot(InsDel[0:nb_iter,0],label="node")
plt.plot(InsDel[0:nb_iter,1],label="edge")
plt.title('Node/Edge insertion/deletion costs')
plt.legend()

# Plotting Node Substitutions costs
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