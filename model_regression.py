from deepged.model import GedLayer
import torch
import torch.nn as nn




class RegressGedLayer(nn.Module):
    def __init__(self, Gs_train ,y_train, k , nb_labels, nb_edge_labels, dict_nodes, rings_andor_fw='sans_rings_sans_fw',
                 normalize=False, node_label="label",verbose=True):
        super(RegressGedLayer, self).__init__()


        self.gedLayer  =   GedLayer(nb_labels, nb_edge_labels, dict_nodes, rings_andor_fw, normalize=normalize,
                         node_label=node_label)


        self.knnLayer = KnnRegressFromGED(y_train, k)
        self.y_train = y_train
        self.Gs_train = Gs_train


    def forward(self , graph ):
        ged_Array = torch.zeros(len(self.Gs_train))
        for i in range(len(self.Gs_train)):

            ged =  self.gedLayer((graph,self.Gs_train[i] ))
            ged_Array[i] = ged

        # print(torch.tensor(ged_Array))
        return self.knnLayer(ged_Array)
        # return ged_Array








class KnnRegressFromGED(nn.Module):
    def __init__(self, y, k, nb_test=5, device=torch.device("cpu"), weights='uniform'):
        super(KnnRegressFromGED, self).__init__()

        self.y = y
        self.k = k
        self.nb_test = nb_test
        self.device = device
        self.weights = weights
        # self.coef_dist = nn.Parameter(torch.tensor(
        #     0.85, requires_grad=True, device=self.device))

    def normalize_val(self, val):
        n = val.shape[0]
        m = val.shape[1]
        alpha = self.coef_dist*self.coef_dist
        ones_m = torch.ones((1, m), device=val.device)

        min = val[:, 0].view(n, 1)@ones_m
        v2 = val-min
        normal = (1.0/(alpha*torch.std(v2, dim=1))).view(n, 1)@ones_m
        print('alpha=', alpha)

        return v2*normal

    # def forward(self, ged):
    #
    #     # ged is supposed to be a 1D array of size (train_size-nb_test X nb_test)
    #     # train_size = 10
    #     # x = ged.view((self.nb_test, train_size-self.nb_test))
    #     get = torch.nn.functional.normalize(ged , dim=0)
    #
    #
    #     val, ind = torch.topk(ged, self.k, dim=0, largest=False)
    #     val_y, ind_y = torch.topk(self.y, self.k, dim=0, largest=False)
    #
    #
    #     val_y_inv, ind_y_inv = torch.topk(self.y, len(self.y) - self.k, dim=0, largest=True)
    #
    #
    #
    #     sum = torch.sum(ged[ind_y_inv] - 1/(len(ind_y_inv)) ) + torch.sum(ged[ind_y])
    #     # if self.weights == 'uniform':
    #     # print(val)
    #     # print(ind)
    #
    #     return torch.sum(self.y[ind], 0)/self.k, sum



    def forward(self, ged):


        ged = torch.nn.functional.normalize(ged , dim=0)


        val, ind = torch.topk(ged, self.k, dim=0, largest=False)


        sim = 1.0/(val+1.0)
        # print(sim)
#        m=torch.nn.Softmax(dim=1)
#        sim=m(-val)
#        print('y=',self.y[ind])
#        print('val=',val)
#        print('sim=',sim)
#        print('res=',torch.sum(sim*self.y[ind],1)/torch.sum(sim,1))
        
        return torch.sum(sim*self.y[ind], 0)/torch.sum(sim, 0)
        # return torch.sum(self.y[ind], 0)/self.k














# #        sim=torch.exp(-alpha*val)
#         val = self.normalize_val(val)
#         sim = 1.0/(val+1.0)
# #        m=torch.nn.Softmax(dim=1)
# #        sim=m(-val)
# #        print('y=',self.y[ind])
# #        print('val=',val)
# #        print('sim=',sim)
# #        print('res=',torch.sum(sim*self.y[ind],1)/torch.sum(sim,1))
#         return torch.sum(sim*self.y[ind], 1)/torch.sum(sim, 1)



class RegressFromGED(nn.Module):
    def __init__(self, y, normalize=None, eps=10**(-6), nb_test=5):
        super(RegressFromGED, self).__init__()

        self.y = y
        self.normalize = normalize
        self.eps = eps
        self.nb_test = nb_test
        self.device = torch.device("cuda:0")
        self.coef_dist = nn.Parameter(torch.tensor(
            1.0, requires_grad=True, device=self.device))
        self.regu = nn.Parameter(torch.tensor(
            1.0, requires_grad=True, device=self.device))

    def forward(self, ged):
        # Warning, the ged are supposed normalized between 0 and 1.
        x = torch.tensor(ged.size(), dtype=torch.float)
        nb_graph = ((1+torch.sqrt(1+8*x))/2.0).int()
        D = torch.zeros(nb_graph, nb_graph, device=ged.device)
        indices = torch.triu_indices(
            nb_graph.item(), nb_graph.item(), offset=1, device=ged.device)
        D[indices[0], indices[1]] = ged
        D = D+D.T
        alpha = self.coef_dist*self.coef_dist
        regu = self.regu*self.regu

        if self.normalize == 'exp':
            K = torch.exp(-alpha*D/D.mean(dim=[0, 1]))
        else:
            K = 1-.5*alpha*D
        train_size = K.shape[0]-self.nb_test
        K_train = K[0:train_size, 0:train_size]
        K_test = K[0:train_size, train_size:]
       # print('train_size=',train_size,', nb tests: ',self.nb_test)
       # print('K=',K)
        #print('K train : ',K_train.size())
        #print('K test : ',K_test.size())
        # print('y=',y.size())
        #print('y train : ',y[0:train_size])
        # print('D=',D)
#        print('K=',K)
#        U,mu,V=torch.svd(K_train,some=True)
# #        if (mu>0.0).all():
#         return self.y[0:train_size].T@torch.inverse(K_train+regu*torch.eye(train_size, device=ged.device))@K_test

        keep = (mu > self.limit).int()
        print('mu=', mu)
        print('keep=', keep)
        y_pred = ((keep*(1/(mu+self.eps)))*((V.T@y)*(V.T@K))).sum(dim=1)
        return y_pred
