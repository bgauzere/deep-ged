import torch
import torch.nn as nn

def train_set_for_kernel_ridge(train_graphs,y,train_size):

    couples=torch.triu_indices(train_size,train_size,offset=1)
    nb_elt=int(train_size*(train_size-1)/2)
    print(nb_elt)
    data=torch.empty((nb_elt,2),dtype=torch.int)
    data[0:nb_elt,0]=train_graphs[couples[0]]
    data[0:nb_elt,1]=train_graphs[couples[1]]
    yt=torch.tensor(y)[train_graphs]

    return data,yt

def train_set_for_knnregression(train_graphs,y,train_size,nb_test):
    yt=torch.tensor(y)[train_graphs]
    nb_train=train_size-nb_test
    nb_elt=nb_test*nb_train
    data=torch.empty((nb_elt,2),dtype=torch.int)
    k=0
    for test in train_graphs[-nb_test:]:
        data[k*nb_train:(k+1)*nb_train,0]=test
        data[k*nb_train:(k+1)*nb_train,1]=train_graphs[0:nb_train]
        k=k+1
    return data,yt


class RegressFromGED(nn.Module):
    def __init__(self, y,normalize=None,eps=10**(-6),nb_test=5):
        super(RegressFromGED, self).__init__() 
        
        self.y=y
        self.normalize = normalize
        self.eps = eps        
        self.nb_test=nb_test
        self.device=torch.device("cuda:0")
        self.coef_dist=nn.Parameter(torch.tensor(1.0,requires_grad=True,device=self.device))
        self.regu=nn.Parameter(torch.tensor(1.0,requires_grad=True,device=self.device))
        
    def forward(self, ged):
        # Warning, the ged are supposed normalized between 0 and 1. 
        x=torch.tensor(ged.size(),dtype=torch.float)
        nb_graph=((1+torch.sqrt(1+8*x))/2.0).int()
        D=torch.zeros(nb_graph,nb_graph,device=ged.device)
        indices=torch.triu_indices(nb_graph.item(),nb_graph.item(),offset=1,device=ged.device)
        D[indices[0],indices[1]]=ged
        D=D+D.T
        alpha=self.coef_dist*self.coef_dist
        regu=self.regu*self.regu
        
        if self.normalize=='exp':
            K=torch.exp(-alpha*D/D.mean(dim=[0,1]))
        else:
            K=1-.5*alpha*D
        train_size=K.shape[0]-self.nb_test
        K_train=K[0:train_size,0:train_size]
        K_test=K[0:train_size,train_size:]
       # print('train_size=',train_size,', nb tests: ',self.nb_test)
       # print('K=',K)
        #print('K train : ',K_train.size())
        #print('K test : ',K_test.size())
        #print('y=',y.size())
        #print('y train : ',y[0:train_size])
        #print('D=',D)
#        print('K=',K)
#        U,mu,V=torch.svd(K_train,some=True)   
#        if (mu>0.0).all():
        return self.y[0:train_size].T@torch.inverse(K_train+regu*torch.eye(train_size,device=ged.device))@K_test
            
        
        keep=(mu>self.limit).int()
        print('mu=',mu)
        print('keep=',keep)
        y_pred=( (keep*(1/(mu+self.eps)))*((V.T@y)*(V.T@K))).sum(dim=1)
        return y_pred

class KnnRegressFromGED(nn.Module):
    def __init__(self, y,k,nb_test=5,device=torch.device("cpu"),weights='uniform'):
        super(KnnRegressFromGED, self).__init__() 
        
        self.y=y
        self.k=k
        self.nb_test=nb_test
        self.device=device
        self.weights=weights
        self.coef_dist=nn.Parameter(torch.tensor(.1,requires_grad=True,device=self.device))
    
    def forward(self, ged):
        alpha=self.coef_dist*self.coef_dist
           
        #ged is supposed to be a 1D array of size (train_size-nb_test X nb_test)
        train_size=ged.shape[0]//self.nb_test+self.nb_test
        x=ged.view((self.nb_test,train_size-self.nb_test))
        val,ind=torch.topk(x,self.k,dim=1,largest=False)
        if self.weights=='uniform':
            return torch.sum(self.y[ind],1)/self.k
        sim=torch.exp(-alpha*val)
        return torch.sum(sim*self.y[ind],1)/torch.sum(sim,1)
