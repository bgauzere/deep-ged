import torch
torch.autograd.set_detect_anomaly(True)

def sinkhorn(S,nb_iter):
        ones_n = torch.ones(S.shape[0],device=S.device)
        ones_m = torch.ones(S.shape[1],device=S.device)
        pdist=torch.nn.PairwiseDistance(p=1)
        converged=False

        Sk = S
        i=1
        while (not converged) and (i <= nb_iter) :
            D=torch.diag(1.0/(Sk@ones_m))
            Sk1 = D@Sk
            D=torch.diag(1.0/(ones_n@Sk1))
            pSk=Sk
            Sk = Sk1@D
            dist=pdist(pSk.view(1,S.shape[0]*S.shape[1]),Sk.view(1,S.shape[0]*S.shape[1]))/S.shape[0]
            converged= dist < 1e-2
            i=i+1

#        print('dist(',i,')=',dist)
        return Sk

def from_lsape_to_lsap(C):
        n=C.shape[0]-1
        m=C.shape[1]-1
        Cred=torch.zeros((n,m),device=C.device,requires_grad=True)
        
        ones_m=torch.ones((1,m+1),device=C.device)
        ones_n=torch.ones((n+1,1),device=C.device)

        Cjeps=ones_n@(C[-1,:]).view(1,m+1)
        #print('Cjeps=',Cjeps)
        Cieps=(C[:,-1]).view(n+1,1)@ones_m
        # print('Cieps=',Cieps)
        CinsRem=(Cieps+Cjeps)[0:n,0:m]
        delta=(C[0:n,0:m]<=CinsRem)
#        print('delta=',delta.long())
        Cred=delta.long()*C[0:n,0:m]+(~delta).long()*CinsRem

        if n<m:
            Cred=Cred-Cjeps[0:n,0:m]
        if n>m:
            Cred=Cred-Cieps[0:n,0:m]
        Cred=Cred-torch.min(Cred)


        return Cred,delta

def lsape_recover_sol_from_lsap(Xred,delta):
        n=delta.shape[0]
        m=delta.shape[1]
        dev=Xred.device

        X=torch.empty((n+1,m+1),device=Xred.device,requires_grad=True)
        delta_X=delta.long()*Xred[0:n,0:m]
        with torch.no_grad():
                X[0:n,0:m]=delta_X
                X[0:n,-1]=torch.ones(n,device=dev)-delta_X@torch.ones(m,device=dev)
                X[-1,0:m]=torch.ones(m,device=dev)-torch.ones(n,device=dev)@delta_X

        return X


def from_costs_to_similarities(C_lsap,k):
        n=C_lsap.shape[0]
        m=C_lsap.shape[1]
        # on rajoute des lignes/colonnes pour avoir une matrice carré (indispensable au SinkHorn).        
        p=max(n,m)        
        squared_cost=(torch.max(C_lsap)+1)*torch.ones((p,p),device=C_lsap.device,requires_grad=True)        
#        squared_cost[:,:]=torch.max(C_lsap)+1
        squared_cost[0:n,0:m]=C_lsap.clone()

#        squared_cost=squared_cost+1e-3
#        print('matrice de coût augmentée pour obtenir une taille carrée')
#       print(Cp.numpy())
        Sim=torch.exp(-k*squared_cost)
#        Sim[-1,-1]=0.0 # enlevé pour le calcul du gradient à vérifier si cela ne gène pas. 
            
        return Sim
def lsape(C,k,nb_iter):
        Cred,delta=from_lsape_to_lsap(C)
        S=from_costs_to_similarities(Cred,k)
        s=sinkhorn(S,nb_iter)
        x=lsape_recover_sol_from_lsap(s,delta).view(C.shape[0]*C.shape[1],1)

        return x

def franck_wolfe(x0,D,c,offset,kmax,n,m):
    k=0
    converged=False
    x=x0
    T=3.0
    dT=.5
    nb_iter=15
    ones_m=torch.ones((1,m+1),device=D.device)
    ones_n=torch.ones((n+1,1),device=D.device)
    while (not converged) and (k<=kmax):
        Cp=(x.T@D+c).view(n+1,m+1)
        #minL,_=Cp.min(dim=1)
        #minL[-1]=0.0
        #Cp=Cp-(minL.view(n+1,1)@ones_m)
        #minC,_=Cp.min(dim=0)
        #minC[-1]=0.0
        #Cp=Cp-ones_n@minC.view(1,m+1)
        Cost=(10*Cp)/(torch.max(Cp)+1e-2)
        b=lsape(Cost,T,nb_iter)
#        print('Cost=',Cost)
#       print('b=',b.view(n+1,m+1))
#        print('x=',x.view(n+1,m+1))

        #nb_iter=max(1,nb_iter-2*dT)
#        b=eps_assigment_from_mapping(torch.exp(-T*Cp),nb_iter).view((n+1)*(m+1),1)
        alpha=x.T@D@(b-x)+c.T@(b-x)
        beta=.5*(b-x).T@D@(b-x)

        if alpha >0: # security check if b is not a local minima (does not occur with real hungarian)                
#            print('alpha positif(',k,')',alpha.item(),'beta=',beta.item())
            return x
        
        #if .5*x.T@D@x+c.T@x > .5*b.T@D@b+c.T@b:
            #print('last value:',.5*b.T@D@b+c.T@b)
         #   xp=b
        

        if beta <=0:
                t=1.0
        else:
                t=min(-alpha/(2*beta),1.0)
#        dirac_betaPos=(torch.sign(beta)+1.0)/2.0
#        t=(1.0-dirac_betaPos)+dirac_betaPos*min(-alpha/(2*beta),1)
#        if torch.isnan(t) :
#               print('t value is nan: alpha=',alpha,'beta=',beta)
#                assert False, "Aborting... Please solve the pb"
                
#        print('alpha=',alpha.item(),'beta:',beta.item(),'t=',t.item())
        x=x+t*(b-x)
        k=k+1
        converged= (-alpha < 1e-5)
#        print('cost(',k,')=',(.5*x.T@D@x+c.T@x).item(),'-alpha=',-alpha.item(),'t=',t.item())
#        print('converge=',converged)
        T=T+dT
        

    return x