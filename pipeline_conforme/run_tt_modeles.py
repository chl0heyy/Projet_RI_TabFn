import gc
import os
import sys
import time
import pandas as pd
import torch
from chargement import liste_tabarena, charge
from pipeline import pipeline
from xgboost import XGBClassifier
from tabicl import TabICLClassifier

# 1. Configuration
TABARENA = liste_tabarena()
total_datasets = len(TABARENA)
METHODES = ["split", "cv+", "jackknife+"]
DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(DOSSIER_SCRIPT, "results_pipeline.csv")
gc.enable()

# 2. Limitation des threads CPU (n_jobs=2 au lieu de -1) pour eviter de saturer la RAM
modeles = [
    XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1, device="cuda", random_state=0, tree_method="hist"),
    TabICLClassifier(device="cuda")
]

# 3. Charger les executions deja faites
deja = set()
if os.path.exists(FICHIER):
    deja = set(map(tuple, pd.read_csv(FICHIER)[["dataset", "modele", "methode"]].drop_duplicates().values))

echecs = []

# 4. Boucle principale
for idx, (_, row) in enumerate(TABARENA.iterrows(), start=1):
    nom = row["name"]
    n_instances = row["NumberOfInstances"]
    
    print(f"\n==================================================", flush=True)
    print(f"Progression : Dataset {idx}/{total_datasets} | Nom : {nom} (Instances : {n_instances})", flush=True)
    print(f"==================================================", flush=True)
    
    try:
        X, y = charge(nom, TABARENA)
    except Exception as e:
        print(f"ECHEC chargement {nom} : {e}", flush=True)
        echecs.append((nom, "chargement", "tous", str(e)))
        continue
        
    for m in modeles:
        nom_modele = type(m).__name__
        
        if nom == "seismic-bumps" and nom_modele == "TabICLClassifier":
            print(f"[{nom_modele}] : seismic-bumps ignore.", flush=True)
            continue

        if nom == "Bioresponse" and nom_modele == "TabICLClassifier":
                    print(f"[{nom_modele}] : Bioresponse ignore.", flush=True)
                    continue
            
        for meth in METHODES:
                
            if (nom, nom_modele, meth) in deja:
                print(f"[{nom_modele} / {meth}] : Deja fait, ignore.", flush=True)
                continue
                
            t0 = time.time()
            try:
                df = pipeline(nom, X, y, m, meth)
                df.to_csv(FICHIER, mode="a", header=not os.path.exists(FICHIER), index=False)
                print(f"SUCCES : {nom_modele} / {meth} en {time.time() - t0:.1f} s", flush=True)
            except Exception as e:
                print(f"ECHEC : {nom_modele} / {meth} -> {e}", flush=True)
                echecs.append((nom, nom_modele, meth, str(e)))
            finally:
                # Liberation forcee de la memoire apres chaque calcul
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Suppression explicite des donnees du dataset termine avant de charger le suivant
    del X, y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nTraitement termine. Nombre d'echecs : {len(echecs)}", flush=True)