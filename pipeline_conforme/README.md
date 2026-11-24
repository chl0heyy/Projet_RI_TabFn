# pipeline_conforme

Pipeline de prédiction conforme sur données tabulaires (TabPFN / XGBoost).
1. `chargement.py` : charge les jeux de données depuis OpenML (`pip install openml`).
2. `pipeline.py` : applique le pipeline conforme sur les données chargées et produit les résultats.
3. `test_pipeline.ipynb` : notebook de test ; à lancer dans le venv du projet (`source .venv/bin/activate`).



Analyse du dataset produit:

Analyse des couvertures moyennes
 (tab recap section 2 + boxplot qui représentent bien) pour alpha=0.1 (généralisable pour alpha =0.05 et alpha=0.2):
1. LAC est généralement très proche de la cible.
2. APS et RAPS obtiennent des couvertures plus grandes que demandées.
3. CV+ et Jackknife+ présentent souvent une couverture supérieure à Split.
4. Jackknife+ n'est pas toujours meilleur par rapport à CV+.

Pour la taille des ensemble, lac est meilleur (cela explique le fait qu'il soit proche (voir inferieur) à la cible, le compromis est la).

Etude taille des ensembles: cohérence plus le alpha est petit plus l'ensemble est grand, TabICL semble produire des ensembles systematiquement plus petits que xgboost

Les scores LAC proposent plus de singleton (plus surs ?; plus extremes ?) que les scores Raps et aps (Raps propose plus de singletons que aps)
=> est ce que c'est une incertitude correctment identifiée ?

analyse du compromis couverture vs taille normailsée:
1. les scores lac est relativement compact, petite couverture (entre 0.4 et 0.7) et proche de la ligne des 0.9
2. aps est un intermediaire, bon compromis et comportement invariant malgrés le datatset
3. raps très étalé, de bonnes couvertures mais des tailles d'ensemble prenant tout le dataset... pas top
4. les deux modeles ont des comporteents relativement similaires

la taille brute des ensembles: lac ne varie pas, en revanche aps augmente de 1 à 2 (moyen) quand on passe en multiclasse 

Tabicl obtient un score F1 Macro meilleur que Xgboost (legere meilleur gestion du desequilibre des classes)


Analyse par cluster:

1. cluster 0: dataset avec bcp d'observations mais moins de variables et de classes
cluster 1: moins d'observation mais plus de variables et de classes
le 1 a l'air plus dur a estimer

2. couverture satisfaisante pour le cluster 0, resultats equivalents tabicl et xgboost: pb de cover a 100% pour jackknife... 
cluster 1: pluscontrasté (attention on est sur 4 datasets meme si c'est les plus pbmatiques) la couverture de jackknife est variable selon le score, pour tabicl bcp de valeurs inferieur a 90 notamment pour LAC (le critique)

cluster 0: si on regarde le set_size_norm on a des scores relativements elevés (lac entre 0.5 et 0.6; aps et raps entre 0.5 et 0.7): bcp de classes possibles dans certains modeles (cohérent avec la coverage de 100%)

pour le cluster 1: des ensembles plus petit (on retrouve le compromis sur la coverage d'autant que 4 datasets)

accuracy meilleure sur cluster 1 (aucune verif stat, pas vrai): meilleur resultat en general mais pas de gestion du desequilibre des classes...
score f1 meilleur sur cluster 0: identifie mieux les classes minoritaires
