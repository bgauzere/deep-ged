# Configuration
* avec pipenv :
`pipenv install`
* avec requirements.txt :
` pip install -r requirements.txt`

# Structure du projet 
<pre>
├── data : Répertoire avec les dataset de molécules à tester
│   ├── Acyclic
│   └── MAO
├── deepged : package d'apprentissage des couts de la GED
│   ├── data_manager : sous package à mieux ranger
│   │   ├── dataset.py
│   │   ├── data_split.py
│   │   ├── label_manager.py
│   ├── model.py : Réseau de neurones principal pour apprendre les couts
│   ├── rings.py : Calcules les rings pour chaque paire de noeuds afin de mieux les comparer 
│   ├── svd.py  : a renommer, contient toutes les fonctions de calculs de mappings
│   ├── triangular_losses.py : TODO 
│   ├── learning.py : routine pour apprendre les couts 
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
└── tests : Dossier contenant les tests unitaires
</pre>

# Méthodes d'optimisation

TODO

# TODO 
* configure main pour gerer le fw et les rings
* Probleme de RAM -> regler par le refactoring (chez moi oui) ?
* Redondance de code entre main et evaluate.py
* Pourquoi quand on réduit le dataset ça ne marche pas ?
* Doc forward de GedLayer (hugo et sidney)
* evaluate.py : redondance de code

Dans les commentaires et docstrings, renseignez d'éventuels trucs à faire et/ou à corriger
avec les mots clés "TODO" et "QUESTION". Pour les lister: 
`grep -rn TODO --include "*py"`

Egalement, analyser les redondances entre main, evaluate et le package training
*  training/gedtrain.py:    TODO : function trop longue, à factoriser
* evaluate.py:TODO : Ne marche pas
* evaluate.py:TODO : A modifier pour prendre en compte les matrices d'adjacence ?
* deepged/svd.py:    # TODO Look into it
* deepged/GedLayer.py:        # TODO : a virer autre part ?
* deepged/GedLayer.py:        TODO
* deepged/GedLayer.py:        Doc TODO
* deepged/GedLayer.py:        Doc TODO
* deepged/GedLayer.py:        TODO : a verifier
* deepged/GedLayer.py:        TODO : nom à changer ?
* deepged/GedLayer.py:        TODO : a factoriser avec toutes les fonctions de calcul de mapping
* deepged/GedLayer.py:        TODO : Utile ? pas vu dans le grep. à virer probablement
* regression.py:TODO : à reprendre pour intégrer au main
* legacy/graph_torch/svd.py:    # TODO Look into it


# Tests
Pour lancer les tests, installer pytest et faire python -m pytest depuis la racine

# Execution

Pour lancer le projet : `main.py` avec ses arguments de lancement suivant :

* `-v` pour la verbosité
* `-n` pour normaliser les valeurs de la GED
* `device` permet a l'utilisateur de chosir la solution hardware de calcul (i.e : `cpu` ou `gpu`)
* `path` permet de renter le chemin d'accès vers le dataset, plus précisément le fichier .ds
* `approximation` permet de choisir la méthode d'approxiamtion (Frank Wolf ou Rings)
* `labelNode` permet de choisir les labels pour les noeuds dépend du dataSet
* `labelEdge` permet de choisir les labels pour les arrêtes dépend du dataSet

Toutes ces informations son disponible si l'argument `-h` est donné.

# Tensorboard 

Pour lancer le tensorboard : `tensorboard --logdir runs` 
