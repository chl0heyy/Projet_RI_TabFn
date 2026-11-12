"""chargement.py : récupérer les datasets TabArena depuis OpenML."""

import openml


def liste_tabarena(type_tache="Classification"):
    """Tableau des tâches TabArena-v0.1 (suite OpenML 457) : 38 de classification, 13 de régression."""
    suite = openml.study.get_suite(457)
    t = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
    t = t[t["task_type"].str.contains(type_tache)]      # "Supervised Classification" / "Supervised Regression"
    t = t[["name", "tid", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]
    return t.sort_values("NumberOfInstances").reset_index(drop=True)


def charge(nom, tabarena):
    """Nom du dataset -> X (DataFrame), y (Series). Télécharge la 1re fois, puis lit le cache."""
    tid = int(tabarena.loc[tabarena["name"] == nom, "tid"].iloc[0])
    return openml.tasks.get_task(tid).get_X_and_y(dataset_format="dataframe")
