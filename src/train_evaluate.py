"""
VoltSight - Étape 2 : entraînement et évaluation du BPNN
========================================================

Question : un BPNN peut-il dire si un isolateur est défectueux
à partir d'une découpe d'image de drone ?

Ce script compare :
  - 2 représentations de l'image :
        A. pixels bruts (128 x 32 = 4096 valeurs)
        B. descripteurs HOG (histogrammes de gradients orientés : formes et contours)
  - 3 modèles :
        1. régression logistique (référence la plus simple)
        2. BPNN codé à la main (bpnn.py)
        3. MLPClassifier de scikit-learn (contrôle : notre BPNN doit faire aussi bien)

Méthode :
  - division train / validation / test PAR GROUPE (quasi-doublons ensemble)
    et stratifiée (même proportion de défauts partout) ;
  - petite recherche d'hyperparamètres sur la validation ;
  - seuil de décision choisi sur la validation pour un rappel >= 0,90 ;
  - évaluation finale UNE SEULE FOIS sur le test ;
  - validation croisée 5 plis pour mesurer la stabilité ;
  - test de masquage : le modèle regarde-t-il vraiment le défaut ?

Utilisation :
  python src/train_evaluate.py   (depuis la racine du dépôt)
"""

import argparse
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from bpnn import BPNN

SEED = 42
TARGET_RECALL = 0.90


# ----------------------------------------------------------------------------
# Représentations de l'image
# ----------------------------------------------------------------------------
def feat_pixels(X):
    return X.reshape(len(X), -1).astype(np.float64) / 255.0


def feat_hog(X):
    return np.stack([hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), feature_vector=True) for img in X])


FEATURES = {"pixels": feat_pixels, "hog": feat_hog}


# ----------------------------------------------------------------------------
# Outils d'évaluation
# ----------------------------------------------------------------------------
def threshold_for_recall(y, p, target=TARGET_RECALL):
    """Plus grand seuil qui garantit un rappel >= target (sur la validation)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def threshold_by_cost(y, p, cost_fn=10.0, cost_fp=1.0):
    """Seuil qui minimise le COÛT sur la validation :
        coût = cost_fn x (défauts manqués) + cost_fp x (fausses alertes).
    Règle métier : un défaut manqué (risque de panne) coûte bien plus cher
    qu'une fausse alerte (quelques secondes de vérification humaine).
    À coût égal, on garde le seuil le plus haut (moins d'alertes)."""
    best_cost, best_thr = None, 0.5
    for t in np.unique(p):
        fn = int(((p < t) & (y == 1)).sum()); fp = int(((p >= t) & (y == 0)).sum())
        c = cost_fn * fn + cost_fp * fp
        if best_cost is None or c < best_cost or (c == best_cost and t > best_thr):
            best_cost, best_thr = c, float(t)
    return best_thr


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "seuil": round(thr, 4),
        "exactitude": round((tp + tn) / len(y), 4),
        "precision": round(precision_score(y, yhat, zero_division=0), 4),
        "rappel": round(recall_score(y, yhat), 4),
        "f1": round(f1_score(y, yhat), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn),
    }


def split(y, groups, seed=SEED):
    """train 60 % / val 20 % / test 20 %, stratifié et par groupe."""
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trval, test = next(sgk.split(np.zeros(len(y)), y, groups))
    sgk2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr, va = next(sgk2.split(np.zeros(len(trval)), y[trval], groups[trval]))
    return trval[tr], trval[va], test


# ----------------------------------------------------------------------------
# Modèles
# ----------------------------------------------------------------------------
def make_bpnn(n_in, hidden, lr, l2, dropout, pos_weight):
    return BPNN([n_in, *hidden, 1], activation="relu", lr=lr, momentum=0.9,
                l2=l2, dropout=dropout, pos_weight=pos_weight, seed=SEED)


GRID = {
    "hidden": [(64,), (128, 32), (256, 64)],
    "lr": [0.01],
    "l2": [1e-3, 1e-2],
    "dropout": [0.0, 0.3],
}


def tune_bpnn(Xtr, ytr, Xva, yva, pos_weight):
    """Recherche par grille : on garde la config au meilleur PR-AUC de validation."""
    best = None
    for hidden, lr, l2, dropout in itertools.product(*GRID.values()):
        m = make_bpnn(Xtr.shape[1], hidden, lr, l2, dropout, pos_weight)
        m.fit(Xtr, ytr, Xva, yva, epochs=200, batch_size=64, patience=20, verbose=0)
        score = average_precision_score(yva, m.predict_proba(Xva))
        cfg = dict(hidden=hidden, lr=lr, l2=l2, dropout=dropout)
        if best is None or score > best[0]:
            best = (score, cfg, m)
    return best


# ----------------------------------------------------------------------------
# Test de masquage : le modèle regarde-t-il le défaut ?
# ----------------------------------------------------------------------------
def repair(img, box, rng=None, random_region=False, ring=3, return_src=False):
    """« Répare » le défaut : remplace la boîte par une zone saine du même
    isolateur.

    Méthode (recopie guidée par le contexte) : on cherche dans l'image la
    zone dont le CONTOUR (anneau de `ring` pixels autour de la boîte)
    ressemble le plus au contour du défaut, sans chevaucher le défaut, puis on
    recopie son intérieur. Comme le contour suit l'isolateur, la zone choisie
    est décalée dans la bonne direction, y compris pour un isolateur en
    diagonale (une simple recopie horizontale créerait une cassure en escalier
    qui ressemble elle-même à un défaut).

    Si random_region=True, on applique la même opération à une boîte de même
    taille déplacée au hasard (contrôle). Si return_src=True, retourne aussi
    la boîte saine recopiée (sx1, sy1, sx2, sy2).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    H, W = img.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
    w, h = x2 - x1, y2 - y1
    if random_region:
        shift = int(rng.integers(-40, 40))
        x1 = int(np.clip(x1 + shift, 0, W - w)); x2 = x1 + w
    r = ring
    # fenêtre = boîte + anneau, limitée à l'image
    wx1, wy1 = max(0, x1 - r), max(0, y1 - r)
    wx2, wy2 = min(W, x2 + r), min(H, y2 + r)
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww >= W or wh >= H:
        return (img.copy(), None) if return_src else img.copy()
    f = img.astype(np.float32)
    target = f[wy1:wy2, wx1:wx2]
    mask = np.ones((wh, ww), bool)
    mask[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = False          # anneau seulement
    win = sliding_window_view(f, (wh, ww))                       # (H-wh+1, W-ww+1, wh, ww)
    ssd = ((win - target)[..., mask] ** 2).mean(-1)
    # interdire les sources qui chevauchent le défaut (avec l'anneau)
    oy = np.arange(win.shape[0])[:, None]; ox = np.arange(win.shape[1])[None, :]
    overlap = (np.abs(oy - wy1) < wh) & (np.abs(ox - wx1) < ww)
    ssd[overlap] = np.inf
    if not np.isfinite(ssd).any():
        return (img.copy(), None) if return_src else img.copy()
    sy, sx = np.unravel_index(np.argmin(ssd), ssd.shape)
    out = img.copy()
    by, bx = sy + (y1 - wy1), sx + (x1 - wx1)
    out[y1:y2, x1:x2] = img[by:by + h, bx:bx + w]
    return (out, (bx, by, bx + w, by + h)) if return_src else out


def grow(box, factor):
    """Agrandit une boîte autour de son centre (factor = 1,5 -> 50 % plus grande)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    return (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2)))


def erase_defect(X_img, boxes, owner, pos_x, idx, factor=1.0, patch=64):
    """Zones dont le défaut (boîte agrandie de `factor`) a été effacé."""
    out, healthy = [], []
    for i in idx:
        img, src = repair(X_img[owner[i]], grow(boxes[owner[i]], factor), return_src=True)
        out.append(img[:, pos_x[i]:pos_x[i] + patch])
        healthy.append(None if src is None else (src[0] - pos_x[i], src[1], src[2] - pos_x[i], src[3]))
    return np.stack(out), healthy


def erase_elsewhere(X_img, boxes, owner, pos_x, idx, rng, factor=1.0, patch=64):
    """Zones où l'on efface une région SAINE de la taille du défaut, placée
    au hasard dans la zone sans toucher au défaut (contrôle)."""
    out = []
    for i in idx:
        x1, y1, x2, y2 = grow(boxes[owner[i]], factor)
        w, h = x2 - x1, y2 - y1
        for _ in range(50):
            nx = int(rng.integers(pos_x[i], pos_x[i] + patch - w + 1))
            ny = int(rng.integers(0, patch - h + 1))
            if nx + w <= x1 or nx >= x2 or ny + h <= y1 or ny >= y2:
                break
        out.append(repair(X_img[owner[i]], (nx, ny, nx + w, ny + h))[:, pos_x[i]:pos_x[i] + patch])
    return np.stack(out)



def masking_test(model, feat_fn, scaler, X_img, boxes, rng):
    p0 = model.predict_proba(scaler.transform(feat_fn(X_img)))
    rep = np.stack([repair(im, b) for im, b in zip(X_img, boxes)])
    ctl = np.stack([repair(im, b, rng, random_region=True) for im, b in zip(X_img, boxes)])
    p_rep = model.predict_proba(scaler.transform(feat_fn(rep)))
    p_ctl = model.predict_proba(scaler.transform(feat_fn(ctl)))
    return {
        "proba_moyenne_originale": round(float(p0.mean()), 4),
        "proba_moyenne_defaut_repare": round(float(p_rep.mean()), 4),
        "proba_moyenne_zone_au_hasard_reparee": round(float(p_ctl.mean()), 4),
        "baisse_defaut_repare": round(float((p0 - p_rep).mean()), 4),
        "baisse_controle": round(float((p0 - p_ctl).mean()), 4),
    }, rep


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_all(out, best_name, model, yte, probs, thr, X_img_te, crops_rep):
    # 1. courbes d'apprentissage
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model.history["train_loss"], label="entraînement")
    ax.plot(model.history["val_loss"], label="validation")
    ax.axvline(model.best_epoch - 1, ls="--", c="grey", label=f"meilleure époque ({model.best_epoch})")
    ax.set_xlabel("époque"); ax.set_ylabel("perte (entropie croisée pondérée)")
    ax.set_title(f"BPNN ({best_name}) - courbes d'apprentissage"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/courbes_apprentissage.png", dpi=120); plt.close(fig)

    # 2. ROC et précision-rappel pour tous les modèles
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(yte, p)
        pr, rc, _ = precision_recall_curve(yte, p)
        axs[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.2f})")
        axs[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(yte, p):.2f})")
    axs[0].plot([0, 1], [0, 1], "k:", lw=1)
    axs[0].set(xlabel="taux de fausses alertes", ylabel="rappel (défauts trouvés)", title="Courbe ROC (test)")
    axs[1].axvline(TARGET_RECALL, ls="--", c="grey")
    axs[1].set(xlabel="rappel", ylabel="précision", title="Précision-rappel (test)")
    for a in axs: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/roc_precision_rappel.png", dpi=120); plt.close(fig)

    # 3. matrice de confusion du BPNN retenu
    p = probs[f"BPNN maison ({best_name})"]
    cm = confusion_matrix(yte, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["prédit normal", "prédit défaut"])
    ax.set_yticks([0, 1], ["vrai normal", "vrai défaut"])
    ax.set_title(f"Matrice de confusion (seuil {thr:.2f})")
    fig.tight_layout(); fig.savefig(f"{out}/matrice_confusion.png", dpi=120); plt.close(fig)

    # 4. exemples d'erreurs
    yhat = (p >= thr).astype(int)
    fn = np.where((yte == 1) & (yhat == 0))[0][:4]
    fp = np.where((yte == 0) & (yhat == 1))[0][:4]
    rows = max(1, max(len(fn), len(fp)))
    fig, axs = plt.subplots(rows, 2, figsize=(9, 1.3 * rows + 0.6), squeeze=False)
    for c, (idx, title) in enumerate([(fn, "Défauts manqués"), (fp, "Fausses alertes")]):
        for r in range(rows):
            a = axs[r, c]; a.axis("off")
            if r < len(idx):
                a.imshow(X_img_te[idx[r]], cmap="gray")
                a.set_title(f"p = {p[idx[r]]:.2f}", fontsize=8)
        axs[0, c].text(0.5, 1.35, title, transform=axs[0, c].transAxes, ha="center", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/exemples_erreurs.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cplid_crops.npz")
    ap.add_argument("--out", default="results/etape2_isolateur_entier")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = np.load(args.data)
    X_img, y, groups, dbox = d["X"], d["y"].astype(np.float64), d["groups"], d["defect_box"]
    tr, va, te = split(y.astype(int), groups)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}  "
          f"(défauts : {int(y[tr].sum())} / {int(y[va].sum())} / {int(y[te].sum())})")
    pos_weight = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    print(f"Poids des défauts dans la perte : {pos_weight:.2f}")

    report = {"donnees": {"train": len(tr), "val": len(va), "test": len(te),
                          "defauts_test": int(y[te].sum()),
                          "pos_weight": round(float(pos_weight), 3)},
              "resultats": {}}
    probs_te, best_overall = {}, None

    for fname, ffn in FEATURES.items():
        print(f"\n=== Représentation : {fname} ===")
        F = ffn(X_img)
        scaler = StandardScaler().fit(F[tr])
        Ftr, Fva, Fte = scaler.transform(F[tr]), scaler.transform(F[va]), scaler.transform(F[te])

        # 1. Régression logistique
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(Ftr, y[tr])
        thr = threshold_for_recall(y[va], lr.predict_proba(Fva)[:, 1])
        p = lr.predict_proba(Fte)[:, 1]
        report["resultats"][f"logistique_{fname}"] = metrics(y[te], p, thr)
        probs_te[f"Logistique ({fname})"] = p

        # 2. BPNN maison
        t0 = time.time()
        score, cfg, m = tune_bpnn(Ftr, y[tr], Fva, y[va], pos_weight)
        print(f"  BPNN meilleure config : {cfg} (PR-AUC val {score:.3f}, {time.time() - t0:.0f} s)")
        thr = threshold_for_recall(y[va], m.predict_proba(Fva))
        p = m.predict_proba(Fte)
        res = metrics(y[te], p, thr)
        res["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
        res["meilleure_epoque"] = int(m.best_epoch)
        report["resultats"][f"bpnn_{fname}"] = res
        probs_te[f"BPNN maison ({fname})"] = p

        # 3. MLP scikit-learn (même architecture, contrôle)
        sk = MLPClassifier(hidden_layer_sizes=cfg["hidden"], alpha=cfg["l2"],
                           learning_rate_init=0.001, early_stopping=True,
                           max_iter=500, random_state=SEED).fit(Ftr, y[tr])
        thr_sk = threshold_for_recall(y[va], sk.predict_proba(Fva)[:, 1])
        p = sk.predict_proba(Fte)[:, 1]
        report["resultats"][f"mlp_sklearn_{fname}"] = metrics(y[te], p, thr_sk)
        probs_te[f"MLP sklearn ({fname})"] = p

        for k in ("logistique", "bpnn", "mlp_sklearn"):
            r = report["resultats"][f"{k}_{fname}"]
            print(f"  {k:12s} rappel {r['rappel']:.2f} | précision {r['precision']:.2f} | "
                  f"F1 {r['f1']:.2f} | PR-AUC {r['pr_auc']:.2f}")

        if best_overall is None or score > best_overall["score"]:
            best_overall = dict(score=score, fname=fname, ffn=ffn, cfg=cfg, model=m,
                                scaler=scaler, thr=thr)

    # --- Validation croisée 5 plis pour la meilleure représentation ---------
    b = best_overall
    print(f"\n=== Validation croisée 5 plis : BPNN ({b['fname']}) ===")
    F = b["ffn"](X_img)
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = []
    for k, (trv, tst) in enumerate(sgk.split(F, y.astype(int), groups)):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        i_tr, i_va = next(inner.split(F[trv], y[trv].astype(int), groups[trv]))
        i_tr, i_va = trv[i_tr], trv[i_va]
        sc = StandardScaler().fit(F[i_tr])
        pw = (y[i_tr] == 0).sum() / (y[i_tr] == 1).sum()
        m = make_bpnn(F.shape[1], b["cfg"]["hidden"], b["cfg"]["lr"], b["cfg"]["l2"],
                      b["cfg"]["dropout"], pw)
        m.fit(sc.transform(F[i_tr]), y[i_tr], sc.transform(F[i_va]), y[i_va],
              epochs=200, patience=20, verbose=0)
        thr = threshold_for_recall(y[i_va], m.predict_proba(sc.transform(F[i_va])))
        r = metrics(y[tst], m.predict_proba(sc.transform(F[tst])), thr)
        cv.append(r)
        print(f"  pli {k + 1} : rappel {r['rappel']:.2f} | précision {r['precision']:.2f} | PR-AUC {r['pr_auc']:.2f}")
    report["validation_croisee"] = {
        "representation": b["fname"],
        **{f"{m_}_moyenne": round(float(np.mean([r[m_] for r in cv])), 4)
           for m_ in ("rappel", "precision", "f1", "roc_auc", "pr_auc")},
        **{f"{m_}_ecart_type": round(float(np.std([r[m_] for r in cv])), 4)
           for m_ in ("rappel", "precision", "f1", "roc_auc", "pr_auc")},
    }

    # --- Test de masquage sur les défauts du test --------------------------
    te_def = te[(y[te] == 1) & (dbox[te][:, 0] >= 0)]
    mask_res, _ = masking_test(b["model"], b["ffn"], b["scaler"], X_img[te_def], dbox[te_def], rng)
    report["test_masquage"] = mask_res
    print(f"\nTest de masquage : {mask_res}")

    # --- Figures et rapport ------------------------------------------------
    plot_all(args.out, b["fname"], b["model"], y[te], probs_te, b["thr"], X_img[te], None)
    with open(f"{args.out}/metriques.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats écrits dans {args.out}/")


if __name__ == "__main__":
    main()
