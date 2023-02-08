"""
Refactoring du code de Luc xp.py
"""
import numpy as np
import torch
import time
from itertools import product
import matplotlib.pyplot as plt
import librariesImport
import gedlibpy

from sinkdiff.sinkdiff import sinkhorn_d1d2
from sinkdiff.sink_utils import simplify_matrix


def random_matrix(n, m, h):
    """
    Genere une matrice random de taille n  x m conformément à l'équation 4 d'ICPR 2022
    """
    S = np.random.rand(n+1, m+1)+np.ones((n+1, m+1))
    S[-1, :] = h*np.random.rand(1, m+1)
    S[:, -1] = h*np.random.rand(1, n+1)
    S[-1, -1] = 0.0

    return S


def random_matrix_pertubation(n, m, h, h2, p, q):
    """
    Genere une matrice  de taille n \times m avec une double pertubation

    """
    S = random_matrix(n, m, h)
    # TODO : que font ces lignes ?
    line = np.random.permutation(n)[0:p]  # on choisit p lines au hasard
    col = np.random.permutation(m)[0:q]  # on choisit q colonnes au hasard
    for idx in list(product(line, col)):  # on parcourt l'ensemble des couples (line,col)
        # on pertube certaines entrées par h2
        S[idx] = h2*S[idx]

    return S


def sinkhorn_random_matrix(n):
    return np.random.rand(n, n)+np.ones((n, n))


def test_size2(n, m, h, h2, p, q, nb_iter):
    """fonction principale pour xp

    Parameters

    ----------

    n : int

    taille matrice 1

    m : int

    taille matrice 2

    h : float

    pertubation initiale pour toutes les insertions/suppressions

    h2 : float

    deuxieme pertubation sur une selection aléatoire de lignes et colonnes

    p : int

    nombre de lignes impactées par la deuxieme pertubation

    q :

    nombre de colonnes impactées par la deuxieme pertubation

    nb_iter :
    """

    optim = 0
    eps_ass2 = 0
    for _ in range(nb_iter):
        similarity = random_matrix_pertubation(n, m, h, h2, p, q)
        similarity_normal = torch.from_numpy(
            simplify_matrix(similarity)).float().to("cpu")

        # Test du sinhorn
        assignement, _ = sinkhorn_d1d2(
            similarity_normal, 100)  # 100 à mettre en param
        # WARNING : assignement n'est pas binaire
        # on enregistre le s
        eps_ass2 += (assignement.to("cpu")*similarity).sum()

        # Test LSAPE
        # Conversion similarity <-> dissimilarity
        # Correspond page 3 ICPR 2022
        c = torch.max(similarity_normal).item()  # h+2.0 sans la perturbation
        matrix_optim = 2.0*c*np.ones((n+1, m+1)) - similarity
        matrix_optim[-1, 0:m] -= c
        matrix_optim[0:n, -1] -= c
        matrix_optim[-1, -1] = 0.0
        result = gedlibpy.hungarian_LSAPE(matrix_optim)
        offset = c*(n+m)
        hungarian = offset - (np.sum(result[2])+np.sum(result[3]))

        # trivial : tout supprimer et tout insérer
        trivial = similarity[-1, 0:m].sum()+similarity[0:n, -1].sum()
        # on veut maximiser on est d'accord.
        if (trivial > hungarian):
            optim += trivial
        else:
            optim += hungarian
    # on retourne la valeur moyenne. Il faudrait pas mieux retourner les résultats sous forme de liste pour comparer xp a xp ?
    # Ici, on va faire la différence des moyennes, et non pas la moyenne des différences...
    return eps_ass2/nb_iter, optim/nb_iter


if __name__ == '__main__':
    perfs = {}
    n = 75  # taill des graphes
    h2_values = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256]
    p_values = list(range(0, 11, 1))
    for h2 in h2_values:
        print('h2=', h2)
        # out_D1D2, out_optim = test_size(n, n, 1.0, 100)
        # print(0.0, 100.0*((out_optim-out_D1D2)/out_optim).item())
        perfs[h2] = []
        for p in p_values:
            out_D1D2, out_optim = test_size2(n, n, 1.0, h2, p, p, 100)
            perf = 100.0*((out_optim-out_D1D2)/out_optim).item()
            perfs[h2].append(perf)
            print(p, perf)

    fig, ax = plt.subplots()
    for h2, perf in perfs.items():
        plt.plot(p_values, perf, label=h2)
    ax.legend()
    plt.show()
