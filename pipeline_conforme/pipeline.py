"""pipeline.py : prédiction conforme (classification) pour n'importe quel modèle sklearn.

    from chargement import liste_tabarena, charge
    from pipeline import pipeline
    from xgboost import XGBClassifier

    TABARENA = liste_tabarena()
    X, y = charge("Fitness_Club", TABARENA)
    modele = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=0)
    df = pipeline("Fitness_Club", X, y, modele, "split")       # 3 scores x 3 alphas = 9 lignes

Le modèle peut être n'importe quel objet avec .fit() et .predict_proba() :
XGBClassifier, TabICLClassifier, RandomForestClassifier, LogisticRegression...
(Sur Mac : ne pas utiliser XGBoost et TabICL dans le même kernel, ça plante.)
"""

import math
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

SCORES = ["lac", "aps", "raps"]
ALPHAS = [0.05, 0.1, 0.2]
JACKKNIFE_MAX_N = 2000      # au-delà, jackknife+ (un entraînement par point) est refusé


# 1. Préparation ---------------------------------------------------------------------

def prepare(X, y, graine, test=0.25, calib=0.30, doublons=True):
    """Supprime les lignes en double (si doublons=True), découpe D (75 %) / test (25 %),
    puis D -> train (70 %) / calib (30 %).
    Médiane, modalité la plus fréquente et one-hot sont appris sur D, puis appliqués à D et au test."""
    y = np.asarray(y)
    n_doublons = 0
    if doublons:                                       # même X ET même y ; on garde la 1re
        double = X.assign(_y=y).duplicated().to_numpy()
        X, y, n_doublons = X[~double], y[~double], int(double.sum())
    y = LabelEncoder().fit_transform(y.astype(str))
    strat = lambda v: v if np.bincount(v).min() >= 2 else None

    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    X = X.astype({c: float for c in num} | {c: object for c in cat})

    X_D, X_te, y_D, y_te = train_test_split(X, y, test_size=test, random_state=graine, stratify=strat(y))
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_D, y_D, test_size=calib, random_state=graine, stratify=strat(y_D))

    ct = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)), cat)])
    ct.fit(X_D)
    t = ct.transform
    infos = {"n_samples": len(X), "n_doublons": n_doublons, "n_features": X.shape[1], "n_classes": len(np.unique(y))}
    return t(X_tr), y_tr, t(X_cal), y_cal, t(X_te), y_te, t(X_D), y_D, infos


# 2. Probabilités : la seule étape où le modèle tourne ------------------------------

def fit_proba(modele, X_fit, y_fit, X_pred, K):
    """Copie vierge du modèle, entraînée sur (X_fit, y_fit) -> probas de X_pred, toujours K colonnes."""
    presentes = np.unique(y_fit)                       # une classe peut manquer dans un petit pli
    m = clone(modele).fit(X_fit, np.searchsorted(presentes, y_fit))
    P = np.zeros((len(X_pred), K))
    P[:, presentes] = m.predict_proba(X_pred)
    return P


def calcule_probas(modele, methode, X_tr, y_tr, X_cal, y_cal, X_te, X_D, y_D, K, n_plis):
    """split : 1 modèle sur train. cv+ : n_plis modèles sur D. jackknife+ : 1 modèle par point de D.
    Renvoie P_cal (probas « hors pli » de la calibration), y_cal, P_test (une matrice par modèle),
    pli (le modèle qui n'a pas vu chaque point de calibration), poids (taille relative des plis)."""
    if methode == "split":
        P = fit_proba(modele, X_tr, y_tr, np.vstack([X_cal, X_te]), K)
        return dict(P_cal=P[:len(X_cal)], y_cal=y_cal, P_test=P[None, len(X_cal):],
                    pli=np.zeros(len(X_cal), int), poids=np.ones(1))
    if methode == "cv+":
        kf = (StratifiedKFold if np.bincount(y_D).min() >= n_plis else KFold)(n_plis, shuffle=True, random_state=0)
    elif methode == "jackknife+":
        if len(X_D) > JACKKNIFE_MAX_N:
            raise ValueError(f"jackknife+ refusé : {len(X_D)} points, donc autant d'entraînements")
        n_plis, kf = len(X_D), KFold(len(X_D))
    else:
        raise ValueError(f"methode inconnue : {methode} (split, cv+, jackknife+)")

    P_cal, P_test, pli = np.zeros((len(X_D), K)), np.zeros((n_plis, len(X_te), K)), np.zeros(len(X_D), int)
    for k, (fit, out) in enumerate(kf.split(X_D, y_D)):
        P = fit_proba(modele, X_D[fit], y_D[fit], np.vstack([X_D[out], X_te]), K)
        P_cal[out], P_test[k], pli[out] = P[:len(out)], P[len(out):], k
    return dict(P_cal=P_cal, y_cal=y_D, P_test=P_test, pli=pli, poids=np.bincount(pli, minlength=n_plis) / len(pli))


# 3. Scores et ensembles : rapide, réutilise les probas -----------------------------

def scores_classes(P, score, lam, k_reg):
    """Score de non-conformité de chaque classe (n x K) ; grand = classe peu plausible.
    lac = 1 - p ; aps = somme des probas jusqu'à la classe (triées) ; raps = aps + lam * max(0, rang - k_reg)."""
    if score == "lac":
        return 1 - P
    ordre = np.argsort(-P, axis=1, kind="stable")
    S = np.empty_like(P)
    np.put_along_axis(S, ordre, np.cumsum(np.take_along_axis(P, ordre, 1), 1), 1)
    if score == "aps":
        return S
    rang = np.argsort(ordre, axis=1) + 1                # rang 1 = classe la plus probable
    return S + lam * np.maximum(0, rang - k_reg)


def ensembles_conformes(pr, score, alpha, lam, k_reg):
    """Booléen n_test x K : la classe est-elle dans l'ensemble ?
    split : seuil = ceil((n+1)(1-alpha))-ième plus petit score de calibration.
    cv+ / jackknife+ : classe gardée si #{i : E_i < score_test} < (1-alpha)(n+1)."""
    n = len(pr["y_cal"])
    E = scores_classes(pr["P_cal"], score, lam, k_reg)[np.arange(n), pr["y_cal"]]
    if len(pr["P_test"]) == 1:
        rang = math.ceil((n + 1) * (1 - alpha))
        q = np.inf if rang > n else np.sort(E)[rang - 1]
        return scores_classes(pr["P_test"][0], score, lam, k_reg) <= q
    compte = sum(np.searchsorted(np.sort(E[pr["pli"] == k]), scores_classes(P_k, score, lam, k_reg))
                 for k, P_k in enumerate(pr["P_test"]))
    return compte < (1 - alpha) * (n + 1)


def metriques(ens, pr, y_te):
    taille = ens.sum(1)
    y_pred = np.tensordot(pr["poids"], pr["P_test"], 1).argmax(1)   # moyenne des modèles (cv+)
    m = lambda f: f(y_te, y_pred, average="macro", zero_division=0)
    return dict(coverage=ens[np.arange(len(y_te)), y_te].mean(), avg_set_size=taille.mean(),
                singleton_rate=(taille == 1).mean(), multi_rate=(taille > 1).mean(), empty_rate=(taille == 0).mean(),
                accuracy=accuracy_score(y_te, y_pred), precision_macro=m(precision_score),
                recall_macro=m(recall_score), f1_macro=m(f1_score))


# 4. pipeline ------------------------------------------------------------------------

def pipeline(nom, X, y, modele, methode, graine=90, score=None, alpha=None, lam=None, k_reg=None, n_plis=10,
             doublons=True):
    """Un dataset, un modèle, une méthode -> DataFrame, une ligne par (score, alpha).
    modele : objet avec .fit / .predict_proba (non entraîné) ; score, alpha : valeur, liste ou None (= tous) ;
    lam, k_reg : réglages RAPS (défaut 0.01 et 1), interdits si "raps" n'est pas dans score ;
    doublons : True = supprime les lignes en double avant le découpage (nombre dans la colonne n_doublons)."""
    scores = SCORES if score is None else [score] if isinstance(score, str) else list(score)
    alphas = ALPHAS if alpha is None else [alpha] if np.isscalar(alpha) else list(alpha)
    if set(scores) - set(SCORES):
        raise ValueError(f"score inconnu : {set(scores) - set(SCORES)} (lac, aps, raps)")
    if (lam is not None or k_reg is not None) and "raps" not in scores:
        raise ValueError("lam et k_reg ne servent qu'au score raps")
    lam, k_reg = (0.01 if lam is None else lam), (1 if k_reg is None else k_reg)

    try:
        X_tr, y_tr, X_cal, y_cal, X_te, y_te, X_D, y_D, infos = prepare(X, y, graine, doublons=doublons)
    except Exception as e:
        raise RuntimeError(f"[prepare] {nom} : {e}") from e
    try:
        t0 = time.time()
        pr = calcule_probas(modele, methode, X_tr, y_tr, X_cal, y_cal, X_te, X_D, y_D, infos["n_classes"], n_plis)
        temps = time.time() - t0
    except Exception as e:
        raise RuntimeError(f"[modèle] {nom} / {type(modele).__name__} / {methode} : {e}") from e

    lignes = [dict(dataset=nom, modele=type(modele).__name__, methode=methode, score=s, alpha=a, graine=graine,
                   lam=lam if s == "raps" else None, k_reg=k_reg if s == "raps" else None,
                   **metriques(ensembles_conformes(pr, s, a, lam, k_reg), pr, y_te),
                   **infos, temps_s=temps)
              for s in scores for a in alphas]
    return pd.DataFrame(lignes)
