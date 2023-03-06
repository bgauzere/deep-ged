"""
Utilitaries for sinkhorn

We assume that all matrices are torch matrices
"""
import torch


def from_cost_to_similarity(C):
    M = torch.max(C)+1.0
    S = 2*M*torch.ones((C.shape[0], C.shape[1]))
    S[-1, :] = M
    S[:, -1] = M
    return simplify_matrix(S-C)


def cost_to_sim(C):
    """
    Transforms a cost matrix to a similarity matrix
    """
    n, m = C.shape
    c = torch.max(C)+1.0
    matrix_sim = 2.0*c*torch.ones((n, m))
    matrix_sim[-1, :] = c
    matrix_sim[:, -1] = c
    return simplify_matrix(matrix_sim - C)


def sim_to_cost(S):
    """
    """
    pass


def simplify_matrix(S):
    """
    S : similarity matrix to normalize

    correspond au début de la page 5 d'ICPR  (simplifing the matrix)
    """
    def compute_threshold(S, n, m):
        ones_n = torch.ones((n, 1))
        ones_m = torch.ones((1, m))

        # chaque elt i,j a pour seuil S[n+1,j] + S[i,m+1]
        C = ones_n@(S[-1, 0:m].reshape(1, m))
        C += S[0:n, -1].reshape(n, 1)@ones_m
        return C

    # n,m : taille de la matrice originale
    n = S.shape[0]-1
    m = S.shape[1]-1

    C = compute_threshold(S, n, m)

    eps = 1e-4  # the low value
    S2 = S.clone()
    # mets eps si S[i,j] < C pour les substitution
    S2[0:n, 0:m] = torch.where(
        S[0:n, 0:m] < C, eps*torch.ones((n, m)), S[0:n, 0:m])

    return S2
