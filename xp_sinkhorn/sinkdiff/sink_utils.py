import numpy as np


def simplify_matrix(S):
    """
    S : similarity matrix to normalize

    correspond au début de la page 5 d'ICPR  (simplifing the matrix)
    """
    def compute_threshold(S, n, m):
        ones_n = np.ones((n, 1))
        ones_m = np.ones((1, m))

        # chaque elt i,j a pour seuil S[n+1,j] + S[i,m+1]
        C = ones_n@(S[-1, 0:m].reshape(1, m))
        C += S[0:n, -1].reshape(n, 1)@ones_m
        return C

    # n,m : taille de la matrice originale
    n = S.shape[0]-1
    m = S.shape[1]-1

    C = compute_threshold(S, n, m)

    eps = 1e-4  # the low value
    S2 = S.copy()
    # mets eps si S[i,j] < C pour les substitution
    S2[0:n, 0:m] = np.where(S[0:n, 0:m] < C, eps*np.ones((n, m)), S[0:n, 0:m])

    return S2
