import numpy as np

def cost_matrix(N):
    res=np.zeros((N,N))
    for i in range(len(res)):
        #print(i)
        for j in range(len(res)):
            res[i][j]=(i+1)*(j+1)

    for i in range(len(res)):
        res[len(res)-1][i]=200
    for i in range(len(res)):
        res[i][len(res)-1]=200
    res[len(res)-1][len(res)-1]=0
    print (res)
    return res


def sub2ind(array_shape, rows, cols):
    return rows*array_shape + cols

def ecvec2mtx( rho, varrho ):
    n=len(rho)
    print(varrho.shape[1])
    m=varrho.shape[1]

    #n = len(rho,1)
    #m = len(varrho,2)
    P = np.zeros((n,m))
    idx_n=[i for i in range(n)]
    idx = sub2ind(m, idx_n, rho) #idx_n.T
    P.flat[[idx]] = 1 #[0]
    idx_m=[i for i in range(m)]
    idx = sub2ind(m, varrho,idx_m)
    P.flat[[idx]]=1
    return P



def min_k(arr, k):
    ind = np.argpartition(arr, k)[:k]
    res=arr[ind]
    
    return res,ind

def mink_loop(W,k):
    ind_final=np.zeros((len(W),k))
    res_final=np.zeros((len(W),k))
    for i in range(len(W)):
        res,ind=min_k(W[i], k)
        ind_final[i]=ind
        res_final[i]=res
    return res_final,ind_final

def auction_fwd_min_lsape(W,delta,rho,varrho,u,v):
    n=len(W)
    m=len(W[0])
    aidx = np.arange(n)
    #kmax = 1000
    k = 0
    feas = [bool(i==-1) for i in rho]
    feas[len(feas)-1] = bool(0)
    pursue = 1
    bij = np.zeros(n)
    print("bij = ",bij)
    nbunass = sum(feas)
    
    while (nbunass > 0 and pursue):

        #sub2ind: vectoriser 2 indices
        
        # Bid phase
        ifeas = aidx[feas]
        print("v : ",v)
        print("W[ifeas]-v : ",W[ifeas]-v)
        p,ji = mink_loop(W[ifeas]-v,2) #1st and 2nd max
        print("p : ",p)
        print("ji : ",ji)
        ji_first_col=[i[0] for i in ji]
        print("ifeas.T : ",ifeas.T)
        ji_ind = sub2ind(len(W[0]),ifeas.T,ji_first_col)
        print("ji_ind : ",ji_ind)
        p_second_col=[i[1] for i in p]
        print("p_second_col : ",p_second_col)
        print("delta = ",delta)
        print("W.flat[[ji_ind]] = ",W.flat[[ji_ind]][0])
        bij[ifeas] = (W.flat[[ji_ind]][0] - p_second_col) - delta
        print("bij[ifeas] = ",bij[ifeas])
        
        # Assignment phase
        for j in range (m-1):
            print("ji_first_col = ",ji_first_col)
            print( ji_first_col==j)
            test = [i for i in ji_first_col]
            #print(test)
            testt=[test[i]==j for i in range(len(test))]
            #print(testt)
            Pj=[ifeas[i] for i in range(len(ifeas)) if testt[i]==True]

            print("Pj = ",Pj)
            
            print("min(bij[Pj]) : ",min_k(bij[Pj], 1))
            if (len(Pj) != 0):
                v_j,ij = min_k(bij[Pj], 1)
                print("v = ",v)
                v[0][j]=v_j[0]
                print("v = ",v)
                print("v[j] : ",v[j])
                print("là : ",Pj[ij[0]])
                print(" v_j = ", v_j[0])
                print("ICI : ", W[Pj[ij[0]]][j] - v_j[0])
                u[Pj[ij[0]]][0] = W[Pj[ij[0]]][j] - v_j[0]
                print("u[Pj[ij[0]]] : ",u[Pj[ij[0]]][0])
                print("varrho = ",varrho)
                if (varrho[0][j] > -1):
                    rho[varrho[0][j]] = -1
                
                rho[Pj[ij[0]]] = j
                varrho[0][j] = Pj[ij[0]]
                pursue = 0
        
        # dummy node m
        test = [i for i in ji_first_col]
        testt=[test[i]==m for i in range(len(test))]
        Pj=[ifeas[i] for i in range(len(ifeas)) if testt[i]==True]

        if (len(Pj) != 0):
            u[Pj][0] = W[Pj][m][0]
            rho[Pj][0] = m
            pursue = 0
        
        feas = [bool(i==-1) for i in rho]
        feas[len(feas)-1] = bool(0)
        nbunass = sum(feas)
        k = k + 1

    return rho,varrho,u,v,k,nbunass


def auction_min_lsape(W):
    n = len(W)
    m= len(W[0])
    X = np.zeros(len(W))
    rho = -(np.ones((n,1)))
    varrho = -(np.ones((1,m)))
    #kmax = 1000
    u = np.zeros((n,1))
    delta = 1/(m+1)
    v=np.zeros((1,m))
    k=0
    nbunass1 = 1
    nbunass2 = 1
    #p = W[len(W)-1]-delta   #Extract last row
    #p[len(p)-1] = 0

    #while (nbunass1 > 0 or nbunass2 > 0):
    rho,varrho,u,v,k1,nbunass1 = auction_fwd_min_lsape(W,delta,rho,varrho,u,v)      #forward auction
    #varrho,rho,v,u,k2,nbunass2 = auction_fwd_min_lsape(W.T,delta,varrho.T,rho.T,v.T,u.T)  #reverse auction
    k = k + k1 #+ k2
    '''
    rho = rho.T
    varrho = varrho.T
    u = u.T
    v = v.T
    '''
    X = ecvec2mtx(rho[:len(rho)-2],varrho[:len(varrho)-2])
    
    return X,rho,varrho,u,v,k





'''
def forward_auction(W,u,v,pi,pi_1,delta):
    n=len(W)-1
    m=len(W[0])-1
    U = [i for i in range(n) if pi[i]==0]
    P=np.zeros(m+1)
    while (U.size != 0): #w=couts
        for i in U:
            j=
            p=min(w[i]-v)
            b=w[i][j]-p+delta
            P.append(i)
        for j in range(m):
            if (P[j]!=0):
                i=
                v_j=
                u_i=w[i][j]-v[j]
                k=pi_1[j]
                if(k>0):
                    pi[k]=0
                    U.append(k)
                pi[i]=j
                pi_1[j]=i
                U.remove(i)
                P[j]=0
        for i in 
'''

if __name__ == "__main__":
    N=10
    W=cost_matrix(N)
    X,rho,varrho,u,v,k=auction_min_lsape(W)
    print(X)
