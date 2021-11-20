# Configuration
* avec pipenv :
`pipenv install`
* avec requirements.txt :
` pip install -r requirements.txt`
## Configuration Dataset
 * `export MAO_DATASET_PATH=/path/to/dataset.ds`

# Structure du projet 
## .
├── data : Répertoire avec les dataset de molécules à tester
│   ├── Acyclic
│   └── MAO
├── deepged : package d'apprentissage des couts de la GED
│   ├── data_manager : sous package à mieux ranger
│   │   ├── DataSet.py
│   │   ├── data_split.py
│   │   ├── label_manager.py
│   ├── extended_label.py : Calcule les labels étendus des noeuds. Permet de prendre en
compte plus de contexte
│   ├── GedLayer.py : Réseau de neurones principal pour apprendre les couts
│   ├── rings.py : Calcules les rings pour chaque paire de noeuds afin de mieux les comparer 
│   ├── svd.py  : a renommer, contient toutes les fonctions de calculs de mappings
│   ├── triangular_losses.py : TODO 
│   └── utils.py 
├── evaluate.py : Donné un jeu de couts, évalue la performance
├── legacy : vieux fichiers qu'on a peur de supprimer mais qu'on supprimera un jour
├── main.py : fichier principal pour lancer l'apprentissage des couts
├── notebooks : dossier avec tout les notebooks pour faire des tets. Contient du vieux code
├── pickle_files : Dossier avec les sauvegardes des poids et éxécutions
├── Pipfile 
├── Pipfile.lock
├── Readme.md
├── regression.py : à intégrer dans main.py à terme.
├── requirements.txt
├── tests : Dossier contenant les tests unitaires
└── training : redondance et utilité par rapport au contenu de deepged à évaluer
    ├── gedtrain.py
    ├── plot.py
    └── train.py

# Méthodes d'optimisation

TODO

# TODO 

Dans les commentaires et docstrings, renseignez d'éventuels trucs à faire et/ou à corriger
avec les mots clés "TODO" et "QUESTION". Pour les lister: 
`grep -r TODO --include "*py"`

* training/gedtrain.py:    TODO : function trop longue, à factoriser
* evaluate.py:TODO : Ne marche pas
*evaluate.py:TODO : A modifier pour prendre en compte les matrices d'adjacence ?
*deepged/svd.py:    # TODO Look into it
*deepged/GedLayer.py:        # TODO : a virer autre part ?
*deepged/GedLayer.py:        TODO
*deepged/GedLayer.py:        Doc TODO
*deepged/GedLayer.py:        Doc TODO
*deepged/GedLayer.py:        TODO : a verifier
*deepged/GedLayer.py:        TODO : nom à changer ?
*deepged/GedLayer.py:        TODO : a factoriser avec toutes les fonctions de calcul de mapping
*deepged/GedLayer.py:        TODO : Utile ? pas vu dans le grep. à virer probablement
*regression.py:TODO : à reprendre pour intégrer au main
*legacy/graph_torch/svd.py:    # TODO Look into it


# Tests
Pour lancer les tests, installer pytest et faire python -m pytest depuis la racine
