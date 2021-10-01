from __future__ import print_function
from gklearn.utils.graphfiles import loadDataset
import torch
import numpy as np

Gs,y = loadDataset('/home/luc/TRAVAIL/DeepGED/MAO/dataset.ds')

def construct_cost_matrix(G1,G2,cost):
    n = G1.order()
    m = G2.order()
    
    #C2=torch.zeros(((n+1)*(m+1),(n+1)*(m+1)))    
    C2=cost[3]*torch.tensor(np.array([ [  k!=l and G1.has_edge(k//(m+1),l//(m+1))^G2.has_edge(k%(m+1),l%(m+1)) for k in range((n+1)*(m+1))] for l in range((n+1)*(m+1))]))        

    D=torch.zeros((n+1)*(m+1))
    D[n*(m+1):]=costs[1]
    D[n*(m+1)+m]=0
    D[np.array([i*(m+1)+m for i in range(n)])]=costs[1]
    D[[k for k in range(n*(m+1)) if k%(m+1) != m]]=costs[0]*torch.tensor([(G1.nodes[k//(m+1)]!=G2.nodes[k%(m+1)]) for k in range(n*(m+1)) if k%(m+1) != m] )

    C2[range(len(C2)),range(len(C2))]=D
    
    
    #C2=C2.masked_fill(mask,costs['edgeDel'])
    
    return C2

x=torch.rand(4,requires_grad=True) # 0 node subs, 1 nodeIns/Del, 2 : edgeSubs, 3 edgeIns/Del
x=torch.exp(x)
costs=x/x.sum()
print(costs)

G1=Gs[0]
G2=Gs[1]
n=G1.order()
m=G2.order()

C=construct_cost_matrix(G1,G2,costs)


import svd

c=torch.diag(C)
D=C-torch.eye(C.shape[0])*c
x0=svd.eps_assigment_from_mapping(torch.exp(-.5*c.view(n+1,m+1)),10).view((n+1)*(m+1),1) # a améliorer.
x=svd.franck_wolfe(x0,D,c,2,15,n,m)
x.register_hook(print)

ged=(.5*x.T@D@x+c.T@x)
ged.backward()

#x.register_hook(print)


print('ged=',ged.item())
x0=x0.view(n+1,m+1)
x=x.view(n+1,m+1)
#print('x0=',x0)
#print('c=',torch.exp(-0.5*c.view(n+1,m+1)))
#print('x=',x)