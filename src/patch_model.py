"""
VoltSight - Étape 3 : BPNN « par zone » (correction du raccourci)
=================================================================

Problème découvert à l'étape 2 (test de masquage) :
  Dans le CPLID, TOUS les isolateurs défectueux sont des images
  synthétiques (isolateur découpé puis collé sur un autre fond), et TOUS
  les isolateurs normaux sont de vraies photos. L'étiquette « défaut » est
  donc confondue avec l'origine de l'image. Le réseau peut obtenir un bon
  score en reconnaissant une image « collée », sans regarder le défaut.
  C'est un apprentissage par raccourci (shortcut learning).

Correction :
  On compare des zones qui viennent de LA MÊME image :
    - zone positive : fenêtre de 32 x 32 qui contient le défaut ;
    - zones négatives : fenêtres de 32 x 32 du même isolateur, loin du défaut.
  Le fond, l'éclairage et l'effet « collé » sont identiques des deux côtés :
  la seule différence est le défaut. Le réseau est obligé de l'apprendre.

Vérification : le même test de masquage. Si on « répare » le défaut,
la probabilité doit maintenant chuter.

Utilisation :
  python src/patch_model.py   (zones 32 x 32)
  python src/patch_model.py --data data/cplid_crops_256.npz --patch 64 --out results/etape3_zones_64
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from bpnn import BPNN
from train_evaluate import (FEATURES, SEED, erase_defect, erase_elsewhere, metrics,
                            repair, split, threshold_for_recall)

PATCH = 32
N_NEG = 3      # zones négatives par isolateur
JITTER = 6     # décalage aléatoire de la zone positive (pixels, = PATCH // 5)


def patch_feats(P, kind):
    if kind == "pixels":
        return FEATURES["pixels"](P)
    from skimage.feature import hog
    return np.stack([hog(p, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), feature_vector=True) for p in P])


def extract_patches(X_img, boxes, rng, repaired=False):
    """Retourne zones, étiquettes, index de l'isolateur d'origine."""
    P, y, owner, pos_x = [], [], [], []
    W = X_img.shape[2]
    for i, (img, b) in enumerate(zip(X_img, boxes)):
        x1, _, x2, _ = b
        cx = (x1 + x2) // 2
        src = repair(img, b) if repaired else img
        # positive : contient le défaut (avec un léger décalage)
        s = int(np.clip(cx - PATCH // 2 + rng.integers(-JITTER, JITTER + 1), 0, W - PATCH))
        P.append(src[:, s:s + PATCH]); y.append(1); owner.append(i); pos_x.append(s)
        # négatives : ne chevauchent pas la boîte du défaut
        cand = [s_ for s_ in range(0, W - PATCH + 1, 4) if s_ + PATCH <= x1 or s_ >= x2]
        for s_ in rng.choice(cand, size=min(N_NEG, len(cand)), replace=False):
            P.append(img[:, s_:s_ + PATCH]); y.append(0); owner.append(i); pos_x.append(int(s_))
    return np.stack(P), np.array(y, dtype=np.float64), np.array(owner), np.array(pos_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cplid_crops.npz")
    ap.add_argument("--out", default="results/etape3_zones_32")
    ap.add_argument("--patch", type=int, default=32, help="taille de la zone (= hauteur des découpes)")
    args = ap.parse_args()
    global PATCH, JITTER
    PATCH = args.patch
    JITTER = PATCH // 5
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = np.load(args.data)
    keep = (d["y"] == 1) & (d["defect_box"][:, 0] >= 0)
    X_img, boxes, groups = d["X"][keep], d["defect_box"][keep], d["groups"][keep]

    P, y, owner, pos_x = extract_patches(X_img, boxes, rng)
    g = groups[owner]                      # toutes les zones d'un isolateur ensemble
    tr, va, te = split(y.astype(int), g)
    print(f"Zones : {len(y)} (défaut : {int(y.sum())})  | train {len(tr)} | val {len(va)} | test {len(te)}")

    report = {"zones": {"total": len(y), "defaut": int(y.sum()),
                        "train": len(tr), "val": len(va), "test": len(te)},
              "resultats": {}}
    best = None
    for kind in ("pixels", "hog"):
        F = patch_feats(P, kind)
        sc = StandardScaler().fit(F[tr])
        Ftr, Fva, Fte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
        pw = (y[tr] == 0).sum() / (y[tr] == 1).sum()

        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(Ftr, y[tr])
        thr = threshold_for_recall(y[va], lr.predict_proba(Fva)[:, 1])
        report["resultats"][f"logistique_{kind}"] = metrics(y[te], lr.predict_proba(Fte)[:, 1], thr)

        cand = []
        for hidden in [(64,), (128, 32)]:
            for l2 in [1e-3, 1e-2]:
                m = BPNN([F.shape[1], *hidden, 1], lr=0.01, l2=l2, dropout=0.3,
                         pos_weight=pw, seed=SEED)
                m.fit(Ftr, y[tr], Fva, y[va], epochs=300, patience=25, verbose=0)
                cand.append((average_precision_score(y[va], m.predict_proba(Fva)), hidden, l2, m))
        score, hidden, l2, m = max(cand, key=lambda c: c[0])
        thr = threshold_for_recall(y[va], m.predict_proba(Fva))
        res = metrics(y[te], m.predict_proba(Fte), thr)
        res["config"] = {"hidden": list(hidden), "l2": l2, "dropout": 0.3}
        report["resultats"][f"bpnn_{kind}"] = res
        for k in ("logistique", "bpnn"):
            r = report["resultats"][f"{k}_{kind}"]
            print(f"  {kind:6s} {k:10s} rappel {r['rappel']:.2f} | précision {r['precision']:.2f} | "
                  f"F1 {r['f1']:.2f} | PR-AUC {r['pr_auc']:.2f}")
        if best is None or score > best[0]:
            best = (score, kind, m, sc, thr)

    # --- Test de masquage : on « répare » le défaut dans les zones positives du test
    score, kind, m, sc, thr = best
    te_pos = te[y[te] == 1]
    orig = P[te_pos]
    rep, _ = erase_defect(X_img, boxes, owner, pos_x, te_pos, 1.0, PATCH)
    rep15, _ = erase_defect(X_img, boxes, owner, pos_x, te_pos, 1.5, PATCH)
    rng_t = np.random.default_rng(123)             # mêmes contrôles que cnn_model.py
    ctl = erase_elsewhere(X_img, boxes, owner, pos_x, te_pos, rng_t, 1.0, PATCH)
    ctl15 = erase_elsewhere(X_img, boxes, owner, pos_x, te_pos, rng_t, 1.5, PATCH)
    prob = lambda A: m.predict_proba(sc.transform(patch_feats(A, kind)))
    p0, p1, p15, pc, pc15 = prob(orig), prob(rep), prob(rep15), prob(ctl), prob(ctl15)
    report["test_masquage"] = {
        "representation": kind,
        "proba_moyenne_originale": round(float(p0.mean()), 4),
        "proba_moyenne_defaut_repare": round(float(p1.mean()), 4),
        "baisse_moyenne": round(float((p0 - p1).mean()), 4),
        "baisse_defaut_efface_x1.5": round(float((p0 - p15).mean()), 4),
        "baisse_zone_saine_effacee_x1": round(float((p0 - pc).mean()), 4),
        "baisse_zone_saine_effacee_x1.5": round(float((p0 - pc15).mean()), 4),
        "part_detectee_avant": round(float((p0 >= thr).mean()), 4),
        "part_detectee_apres_reparation": round(float((p1 >= thr).mean()), 4),
        "part_detectee_apres_effacement_x1.5": round(float((p15 >= thr).mean()), 4),
    }
    print(f"\nTest de masquage (zones) : {report['test_masquage']}")

    # --- Figure : exemples avant / après réparation
    k = min(6, len(te_pos))
    fig, axs = plt.subplots(2, k, figsize=(1.6 * k, 3.6), squeeze=False)
    for j in range(k):
        for r, (img, p) in enumerate([(orig[j], p0[j]), (rep[j], p1[j])]):
            axs[r, j].imshow(img, cmap="gray", vmin=0, vmax=255); axs[r, j].axis("off")
            axs[r, j].set_title(f"p = {p:.2f}", fontsize=9)
    axs[0, 0].text(-0.15, 0.5, "défaut", transform=axs[0, 0].transAxes, rotation=90, va="center")
    axs[1, 0].text(-0.15, 0.5, "réparé", transform=axs[1, 0].transAxes, rotation=90, va="center")
    fig.suptitle("BPNN par zone : probabilité de défaut avant et après réparation")
    fig.tight_layout(); fig.savefig(f"{args.out}/zones_masquage.png", dpi=120); plt.close(fig)

    with open(f"{args.out}/metriques_zones.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Résultats écrits dans {args.out}/")


if __name__ == "__main__":
    main()
