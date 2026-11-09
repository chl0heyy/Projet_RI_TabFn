"""
generation_decoupage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def generation_decoupage(
    fichier_csv="dataset_modifie.csv",
    graine=90,
    colonne_cible="attended",
    proportion_test=0.25,
    proportion_calibration=0.30,
    dossier_sortie=None,
):
    """
    Génère decoupage_graine.csv avec cette réparrtition:

    Répartition :
      TRAIN_TEST = 0 : groupe D, 75 % 
      TRAIN_TEST = 1 : test final, 25 % 
      SPLIT_CONFORMAL = 0 : entraînement, au sein de D
      SPLIT_CONFORMAL = 1 : calibration, au sein de D
      SPLIT_CONFORMAL vide : test final

    Les valeurs manquantes sont remplacées par les médianes calculées sur D uniquement.
    """
    fichier_csv = Path(fichier_csv)

    if not fichier_csv.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {fichier_csv.resolve()}. "
            "Vérifie le chemin vers dataset_modifie.csv."
        )

    if not 0 < proportion_test < 1:
        raise ValueError("proportion_test doit être comprise entre 0 et 1.")
    if not 0 < proportion_calibration < 1:
        raise ValueError("proportion_calibration doit être comprise entre 0 et 1.")

    donnees = pd.read_csv(fichier_csv)

    if colonne_cible not in donnees.columns:
        raise ValueError(
            f"La colonne cible '{colonne_cible}' est absente. "
            f"Colonnes disponibles : {donnees.columns.tolist()}"
        )
    if donnees[colonne_cible].isna().any():
        raise ValueError(
            f"La colonne cible '{colonne_cible}' contient des valeurs manquantes. "
            "Il faut les traiter avant de créer le découpage."
        )

    colonnes_a_exclure = [colonne_cible, "TRAIN_TEST", "SPLIT_CONFORMAL"]
    X_complet = donnees.drop(
        columns=[c for c in colonnes_a_exclure if c in donnees.columns]
    )
    y_brut = donnees[colonne_cible]

    colonnes_non_numeriques = X_complet.select_dtypes(exclude=np.number).columns.tolist()
    if colonnes_non_numeriques:
        raise ValueError(
            "Les variables explicatives doivent être numériques avant le découpage. "
            f"Colonnes non numériques : {colonnes_non_numeriques}. "
            "Vérifie que le One-Hot Encoding a bien été fait dans dataset_modifie.csv."
        )

    encodeur = LabelEncoder()
    y_complet = encodeur.fit_transform(y_brut)
    indices = np.arange(len(donnees))

    # Première séparation : D / test final.
    indices_D, indices_test = train_test_split(
        indices,
        test_size=proportion_test,
        random_state=graine,
        stratify=y_complet,
    )

    # Deuxième séparation : entraînement / calibration dans D.
    indices_train, indices_cal = train_test_split(
        indices_D,
        test_size=proportion_calibration,
        random_state=graine,
        stratify=y_complet[indices_D],
    )

    # Les indicateurs sont ajoutés sans réordonner les lignes du CSV.
    donnees_decoupage = donnees.copy()
    donnees_decoupage["TRAIN_TEST"] = 0
    donnees_decoupage["SPLIT_CONFORMAL"] = pd.Series(
        pd.NA, index=donnees_decoupage.index, dtype="Int64"
    )
    donnees_decoupage.loc[indices_test, "TRAIN_TEST"] = 1
    donnees_decoupage.loc[indices_train, "SPLIT_CONFORMAL"] = 0
    donnees_decoupage.loc[indices_cal, "SPLIT_CONFORMAL"] = 1

    dossier = Path(dossier_sortie) if dossier_sortie is not None else fichier_csv.parent
    dossier.mkdir(parents=True, exist_ok=True)
    fichier_decoupage = dossier / f"decoupage_{graine}.csv"
    donnees_decoupage.to_csv(fichier_decoupage, index=False)

    # Matrices de variables explicatives ; imputation calculée sur D seulement.
    X_D_brut = X_complet.iloc[indices_D].copy()
    X_test_brut = X_complet.iloc[indices_test].copy()
    medianes_D = X_D_brut.median()
    X_D = X_D_brut.fillna(medianes_D).to_numpy(dtype=float)
    X_test = X_test_brut.fillna(medianes_D).to_numpy(dtype=float)

    y_D = y_complet[indices_D]
    y_test = y_complet[indices_test]

    positions_train = np.isin(indices_D, indices_train)
    positions_cal = np.isin(indices_D, indices_cal)

    X_train = X_D[positions_train]
    X_cal = X_D[positions_cal]
    y_train = y_D[positions_train]
    y_cal = y_D[positions_cal]

    print(f"Découpage créé : {fichier_decoupage.resolve()}")
    print(f"Graine : {graine}")
    print(f"Groupe D : {len(indices_D)} lignes | Test final : {len(indices_test)} lignes")
    print(f"Entraînement : {len(indices_train)} lignes | Calibration : {len(indices_cal)} lignes")

    return {
        "donnees_decoupage": donnees_decoupage,
        "fichier_decoupage": fichier_decoupage,
        "X_D": X_D,
        "X_test": X_test,
        "y_D": y_D,
        "y_test": y_test,
        "X_train": X_train,
        "X_cal": X_cal,
        "y_train": y_train,
        "y_cal": y_cal,
        "classes": encodeur.classes_,
        "colonnes_X": X_complet.columns.tolist(),
    }
