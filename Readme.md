                              Labels                   Labels étendus             Plongement
  ---------------- ---------------------------- ---------------------------- ---------------------
   Classification   DeepGraphWithNNTorch.ipynb   DeepGraphWithNNTorch.ipynb        gcn.ipynb
     Regression     DeepGraphRegression.ipynb    DeepGraphRegression.ipynb    gcnRegression.ipynb

Optimisation {#optimisation .unnumbered}
============

Dans la classe Net (forward) et en préambule de cette classe.

Franck wolfe :

:   S=self.mapping\_from\_cost(C,n,m)

Puissance itérés ou SVD :

:   S=self.mapping\_from\_similarity(C,n,m)

    Puissances itérés :

    :   from svd import iterated\_power as compute\_major\_axis (en
        préambule)

    SVD :

    :   compute\_major\_axis=svd.CustomMajorAxis.apply (en préambule)

Fichiers {#fichiers .unnumbered}
========

-   svd.py : contient les classes fonctions pour obtenir une matrice
    d'assignement à partir d'une matrice de coût ou de similarité.
    Comprends: la SVD, puissance itérés, Franck Wolfe,

-   regression.py : Fonction de regressions avec deux classes: Une
    classe qui utilise la kernel ridge regression (nécessite une matrice
    de distances complète) et une classe qui utilise une Knn regression
    (ne nécessite que la distance des données à prédire aux données de
    train).

-   triangular\_loss.py Deux classes qui pénalisent la violation de
    l'inégalité triangulaire. Une classe qui considère avoir une matrice
    de coût de substitution entre noeuds (pour les labels, cf tableau ci
    dessus) et une autre classe qui ne considère qu'un coût de
    substitution qui est multiplié à la distance euclidienne entre les
    plongements de noeuds.
