'''
TODO : faire le ménage
'''
import torch
from torch.autograd import Function


def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    print('1er col de grad V:', grad_V[:, 0])
    print('grad_V=', grad_V)
    S = torch.eye(N) * S.reshape((N, 1))  # .cpu(S.get_device()) après le eye

    inner = K.T * (V.T @ grad_V)
    # print('K=',K,'max K=',K.max())
    # print('gradV=',grad_V)

    inner = (inner + inner.T) / 2.0
    # print('inner=',inner)
    return 2 * U @ S @ inner @ V.T


def svd_grad_K(S):
    import math
    import numpy as np

    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    Kp = diff * plus
    eps = torch.ones((N, N)) * 10 ** (-6)
    max_diff = torch.max(torch.abs(Kp), eps)
    sign_diff = torch.sign(diff) + eps
    K_neg = sign_diff * max_diff
    K_neg[torch.arange(N), torch.arange(N)] = 1
    ones = torch.ones((N, N))  # .cpu(S.get_device())
    rm_diag = ones - torch.eye(N)  # .cpu(S.get_device())

    K = (1 / K_neg) * rm_diag

    # TODO Look into it
    # eps = torch.ones((N, N)) * 10**(-4)
    ##eps = eps.cpu(S.get_device())
    # max_diff = torch.max(torch.abs(diff), eps)
    # sign_diff = torch.sign(diff)

    # K_neg = sign_diff * max_diff
    # print('S=',S)
    # print('K_neg=',K_neg)
    # print('plus=',plus)
    # print('diff=',diff)
    # gaurd the matrix inversion
    # K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-5)
    # K_neg = 1 / K_neg
    # K_pos = 1 / plus
    # print('max K_neg',K_neg.max().item(),'mas K_pos=',K_pos.max().item())
    if (not np.isfinite(K.max().item())):
        print('min diff', torch.abs(max_diff).min().item(), 'inv=', 1.0 / torch.abs(max_diff).min().item(),
              'min |K_neg|', torch.abs(K_neg).min().item())
    # ones = torch.ones((N, N))#.cpu(S.get_device())
    # rm_diag = ones - torch.eye(N)#.cpu(S.get_device())
    # K = K_neg * K_pos * rm_diag

    return K


class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.
        try:
            U, S, V = torch.svd(input, some=True)
        except:
            U, S, V = torch.svd(input + 1e-2 * input.mean()
                                * torch.rand(input.shape[0], input.shape[1]))
            import ipdb
            ipdb.set_trace()

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors

        grad_input = compute_grad_V(U, S, V, grad_V)
        # print('grad de U et S:',grad_U.max(),grad_S.max())
        return grad_input


customsvd = CustomSVD.apply


def compute_grad_MA(U, S, V, grad_V):
    N = S.shape[0]

    K = svd_grad_K_MA(S)

    inner_vec = K * (V.T @ grad_V)
    inner = torch.zeros(N, N).to(S.device)
    inner[:, 0] = S * inner_vec
    inner[0, :] = S[0] * inner_vec.T

    # print('K=',K,'max K=',K.max())
    # print('gradV=',grad_V)

    # inner = (inner + inner.T) / 2.0
    # print('inner=',inner)
    return U @ inner @ V.T


def svd_grad_K_MA(S):
    import numpy as np
    N = S.shape[0]
    K = S[0] * S[0] - S * S

    if (K[1] != 0):
        K[0] = 1
        K = 1 / K
        K[0] = 0
        return K

    print('Warning  svd_grad_K_MA: 2 maximal eigen values')
    eps = torch.ones(N, device=S.device) * 10 ** (-6)
    max_diff = torch.max(torch.abs(K), eps)
    sign_diff = torch.sign(K) + eps
    K_neg = sign_diff * max_diff
    K_neg[0] = 1
    K = (1 / K_neg)
    K[0] = 0

    if (not np.isfinite(K.max().item())):
        print('min diff', torch.abs(max_diff).min().item(), 'inv=', 1.0 / torch.abs(max_diff).min().item(),
              'min |K_neg|', torch.abs(K_neg).min().item())
    # ones = torch.ones((N, N))#.cpu(S.get_device())
    # rm_diag = ones - torch.eye(N)#.cpu(S.get_device())
    # K = K_neg * K_pos * rm_diag

    return K


class CustomMajorAxis(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.

        if torch.norm(torch.diag(input), p=2) / torch.norm(input, p='fro') > .99:
            print('Warning Custom Major Axis: Matrix considered as diagonal')
            val, ind = torch.diag(input).sort(descending=True)
            V = torch.eye(input.shape[0], device=input.device)
            U = V
        else:
            try:
                U, S, V = torch.svd(input, some=True)
            except:
                print(
                    'Error CustomMajorAxis, SVD pb detected trying to recover. M=', input)
                print('norm of diagonal/norm of matrix',
                      100.0 * torch.norm(torch.diag(input), p=2) / torch.norm(input, p='fro'))
                U, S, V = torch.svd(
                    input + 1e-2 * input.mean() * torch.rand(input.shape[0], input.shape[1], device=input.device))
                import ipdb
                ipdb.set_trace()

        val, ind = torch.diag(V.T @ input @ V).sort(descending=True)
        Sp = val
        Up = torch.zeros_like(U)
        Up = V.T[ind]
        Vp = Up.T
        ctx.save_for_backward(Up, Sp, Vp)
        # ctx.save_for_backward(U, S, V)
        return Vp[:, 0]

    @staticmethod
    def backward(ctx, grad_V):
        U, S, V = ctx.saved_tensors

        grad_input = compute_grad_MA(U, S, V, grad_V)

        return grad_input


def iterated_power(M, inv=False):
    # we estimate an upper bound of the max absolute value
    w = torch.ones(M.shape[0], device=M.device)
    for i in range(5):
        mw = M @ w
        w = mw / torch.norm(mw, 2)
    # we shift all eigenvalues so that the max one is positive (hopefully)
    l1 = torch.norm(mw, 2)
    N = l1 * torch.eye(M.shape[0], device=M.device) + M
    w = torch.ones(M.shape[0], device=M.device)
    for i in range(20):
        mw = N @ w
        w = mw / torch.norm(mw, 2)

    l = torch.norm(mw, 2) - l1

    # If inv is true we affinate with the inverse method.
    if inv:
        M_inv = torch.inverse(
            M - (l + 1) * torch.eye(M.shape[0], device=M.device))
        for i in range(10):
            mw = M_inv @ w
            w = mw / torch.norm(mw, 2)

    return w


def eps_assigment_from_mapping(S, nb_iter):
    ones_n = torch.ones(S.shape[0], device=S.device)
    ones_m = torch.ones(S.shape[1], device=S.device)

    last_line = S[-1, :] @ ones_m
    if last_line > S.shape[1]:
        S[-1, :] = S.shape[1] * S[-1, :] / last_line
    last_column = S[:, -1] @ ones_n
    if last_column > S.shape[0]:
        S[:, -1] = S[:, -1] * S.shape[0] / last_column

    Sk = S
    i = 0
    converged = False
    while i <= nb_iter and not converged:
        D = torch.diag(1.0 / (Sk @ ones_m))
        D[D.shape[0] - 1, D.shape[1] - 1] = 1.0
        Sk1 = D @ Sk
        norm_col = torch.linalg.norm(
            (ones_n @ Sk1 - ones_m)[0:-1], ord=float('inf'))
        D = torch.diag(1.0 / (ones_n @ Sk1))
        D[D.shape[0] - 1, D.shape[1] - 1] = 1.0
        Sk = Sk1 @ D
        norm_line = torch.linalg.norm(
            (Sk @ ones_m - ones_n)[0:-1], ord=float('inf'))
        converged = (norm_col <= 1e-2) and (norm_line <= 1e-2)
        i = i + 1
    return Sk


def eps_assign2(S, nb_iter):
    '''
    correspond au sinkhorn demontré par luc
    àpriviligier
    '''
    ones_n = torch.ones(S.shape[0], device=S.device)
    ones_m = torch.ones(S.shape[1], device=S.device)
    c = ones_m
    converged = False
    i = 0
    while i <= nb_iter and not converged:
        rp = 1.0 / (S @ c)
        rp[-1] = 1.0
        if i >= 1:
            norm_r = torch.linalg.norm(
                r / rp - torch.ones_like(r / rp), ord=float('inf'))
        r = rp
        cp = 1.0 / (S.T @ r)
        cp[-1] = 1.0
        norm_c = torch.linalg.norm(
            c / cp - torch.ones_like(c / cp), ord=float('inf'))
        c = cp
        if i >= 1:
            converged = (norm_r <= 1e-2) and (norm_c <= 1e-2)
        i += 1
    #        print('r=',r)
    #        print('c=',c)
    return torch.diag(r) @ S @ torch.diag(c)


def franck_wolfe(x0, D, c, offset, kmax, n, m):
    k = 0
    converged = False
    x = x0
    T = 1.0
    dT = .5
    nb_iter = 40
    ones_m = torch.ones((1, m + 1), device=D.device)
    ones_n = torch.ones((n + 1, 1), device=D.device)
    while (not converged) and (k <= kmax):
        Cp = (x.T @ D + c).view(n + 1, m + 1)
        minL, _ = Cp.min(dim=1)
        minL[-1] = 0.0
        Cp.sub_(minL.view(n + 1, 1) @ ones_m)
        T = min(10, 7.0 * torch.log(torch.tensor(10.0)) / torch.max(Cp))
        #        minC,_=Cp.min(dim=0)
        #        minC[-1]=0.0
        #        Cp=Cp-ones_n@minC.view(1,m+1)
        #        Cp=Cp/torch.max(Cp)

        b = eps_assign2(torch.exp(-T * Cp), nb_iter).view((n + 1) * (m + 1), 1)
        alpha = x.T @ D @ (b - x) + c.T @ (b - x)
        #        print('M=',torch.exp(-T*Cp))
        #        print('b=',b.view(n+1,m+1))
        #        t=offset/(offset+k)
        #        x=x+t*(b-x)

        # security check if b is not a local minima (does not occur with real hungarian)
        if alpha > 0:
            # print('alpha positif(', k, ')', alpha.item())
            """
            if k==0:
                print('T=',T)
                print('Sim=',torch.exp(-T*Cp))
                print('Cp=',Cp)
                print('Cp original= ',(x.T@D+c).view(n+1,m+1))
            """
            return x

        beta = .5 * (b - x).T @ D @ (b - x)
        if beta <= 0:
            t = 1.0
        else:
            t = min(-alpha / (2 * beta), 1.0)
        # if .5*x.T@D@x+c.T@x > .5*b.T@D@b+c.T@b:
        # print('last value:',.5*b.T@D@b+c.T@b)
        #   xp=b

        #        dirac_betaPos=(torch.sign(beta)+1.0)/2.0
        #        t=(1.0-dirac_betaPos)+dirac_betaPos*min(-alpha/(2*beta),1)
        #        #        print('alpha=',alpha,'beta:',beta,'t=',t)
        # if beta <=0:
        #    xp=b
        # else:
        x = x + t * (b - x)
        k = k + 1
        converged = (-alpha < 1e-5)
        #        print('cost(',k,')=',.5*x.T@D@x+c.T@x,'-alpha=',-alpha)
        #        print('converge=',converged)
        T = T + dT

    return x
