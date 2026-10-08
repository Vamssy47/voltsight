"""
VoltSight - Étape 4 : CNN (réseau convolutif) sur la tâche par zones
====================================================================

Objectif : battre le BPNN de l'étape 3 sur EXACTEMENT la même tâche
(mêmes zones 64 x 64, même découpage train / validation / test) et prouver
que le CNN regarde vraiment le défaut.

Pourquoi un CNN devrait faire mieux qu'un BPNN :
  - Le BPNN relie chaque pixel à chaque neurone : un isolateur cassé à gauche
    et le même cassé à droite sont, pour lui, deux motifs différents.
  - Un CNN applique les MÊMES petits filtres (3 x 3) partout dans l'image :
    il apprend « à quoi ressemble un disque manquant » une seule fois et le
    reconnaît n'importe où (invariance par translation). Il a aussi beaucoup
    moins de poids, donc il surapprend moins avec peu de données.

Trois critères de réussite :
  1. PR-AUC sur le test > celle du BPNN (0,67 à 0,70 selon l'environnement) ;
  2. test de masquage : la probabilité CHUTE quand on efface le défaut,
     nettement plus que quand on efface une zone saine de même taille ;
  3. Grad-CAM : la carte de chaleur est plus forte sur le défaut que sur une
     zone saine du même isolateur.

Deux variantes sont entraînées :
  - « standard » : zones originales seulement ;
  - « contrefactuel » : on ajoute à l'entraînement les MÊMES zones avec le
    défaut effacé (étiquette 0). Le réseau voit des paires « avec / sans
    défaut » qui ne diffèrent que par le défaut : il est obligé de s'appuyer
    sur lui. Pour qu'il n'apprenne pas « trace de recopie = normal », on
    ajoute aussi des zones (défectueuses ET normales) où une zone saine
    au hasard a été recopiée, avec leur étiquette d'origine.

Utilisation (GPU conseillé ; sur CPU compter 20 à 40 minutes) :
  python src/cnn_model.py   (depuis la racine du dépôt)
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import patch_model as pm
from train_evaluate import (TARGET_RECALL, erase_defect, erase_elsewhere, grow, metrics,
                            repair, split, threshold_by_cost, threshold_for_recall)

PATCH = 64


# ----------------------------------------------------------------------------
# Le réseau
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """3 blocs convolutifs + moyenne globale + couche de sortie.

    entrée 1 x 64 x 64
      -> [conv 3x3 (16) -> BN -> ReLU -> conv 3x3 (16) -> BN -> ReLU -> maxpool]  32 x 32
      -> [conv 3x3 (32) -> BN -> ReLU -> conv 3x3 (32) -> BN -> ReLU -> maxpool]  16 x 16
      -> [conv 3x3 (64) -> BN -> ReLU -> conv 3x3 (64) -> BN -> ReLU]             16 x 16  <- Grad-CAM ici
      -> moyenne globale -> dropout -> linéaire -> logit
    """

    def __init__(self, c=16, dropout=0.3):
        super().__init__()

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                      nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.b1, self.b2, self.b3 = block(1, c, True), block(c, 2 * c, True), block(2 * c, 4 * c, False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * c, 1))

    def features(self, x):
        return self.b3(self.b2(self.b1(x)))

    def forward(self, x):
        f = self.features(x)
        return self.head(f.mean(dim=(2, 3))).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Augmentations (seulement à l'entraînement)
# ----------------------------------------------------------------------------
def augment(x):
    """x : (N, 1, H, W) dans [0, 1]. Retournements, luminosité, contraste, bruit."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)                                  # miroir gauche-droite
    if torch.rand(1) < 0.5:
        x = x.flip(2)                                  # miroir haut-bas
    n = x.shape[0]
    gain = 1 + 0.3 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)   # contraste ±15 %
    bias = 0.2 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)       # luminosité ±10 %
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * gain + mean + bias
    x = x + 0.02 * torch.randn_like(x)                 # bruit de capteur
    return x.clamp(0, 1)


# ----------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------
def to_tensor(P, device):
    return torch.tensor(P[:, None].astype(np.float32) / 255.0, device=device)


@torch.no_grad()
def predict(model, X, bs=256):
    """Probabilité de défaut. `model` peut être une LISTE de modèles :
    on fait alors la moyenne de leurs probabilités (ensemble)."""
    if isinstance(model, (list, tuple)):
        return np.mean([predict(m, X, bs) for m in model], axis=0)
    model.eval()
    return torch.cat([torch.sigmoid(model(X[i:i + bs])) for i in range(0, len(X), bs)]).cpu().numpy()


def train(Xtr, ytr, Xva, yva, seed, device, epochs=80, patience=15, lr=2e-3, bs=64, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN().to(device)
    pos_weight = torch.tensor((ytr == 0).sum().item() / (ytr == 1).sum().item(), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, wait, hist = -1, None, 0, {"train_loss": [], "val_pr_auc": []}
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(augment(Xtr[idx]))
            loss = F.binary_cross_entropy_with_logits(logits, ytr[idx], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        ap = average_precision_score(yva.cpu().numpy(), predict(model, Xva))
        hist["train_loss"].append(tot / n); hist["val_pr_auc"].append(ap)
        if ap > best + 1e-4:
            best, wait = ap, 0
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        else:
            wait += 1
        if verbose and ep % 10 == 0:
            print(f"    époque {ep:3d} | perte {tot / n:.4f} | PR-AUC val {ap:.3f}")
        if wait >= patience:
            break
    model.load_state_dict(best_state[0])
    model.best_epoch, model.hist = best_state[1], hist
    return model


# ----------------------------------------------------------------------------
# Grad-CAM : où le réseau regarde-t-il ?
# ----------------------------------------------------------------------------
def grad_cam(model, X):
    """Carte d'importance (N, 64, 64) normalisée dans [0, 1].
    Pour un ensemble (liste de modèles) : moyenne des cartes."""
    if isinstance(model, (list, tuple)):
        cams = np.mean([grad_cam(m, X) for m in model], axis=0)
        return cams / (cams.max(axis=(1, 2), keepdims=True) + 1e-8)
    model.eval()
    cams = []
    for i in range(len(X)):
        x = X[i:i + 1].clone().requires_grad_(False)
        f = model.features(x)
        f.retain_grad()
        logit = model.head(f.mean(dim=(2, 3))).squeeze()
        model.zero_grad(); logit.backward()
        w = f.grad.mean(dim=(2, 3), keepdim=True)            # poids = gradient moyen par canal
        cam = F.relu((w * f).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=X.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        cams.append(cam.detach().cpu().numpy())
    return np.stack(cams)


def pointing_game(cams, boxes, tol=2):
    """Part des cartes dont le point le plus chaud tombe dans la boîte du défaut
    (« pointing game »). À comparer à pointing_chance()."""
    hits = []
    for cam, (x1, y1, x2, y2) in zip(cams, boxes):
        r, c = np.unravel_index(cam.argmax(), cam.shape)
        hits.append(x1 - tol <= c <= x2 + tol and y1 - tol <= r <= y2 + tol)
    return float(np.mean(hits))


# ----------------------------------------------------------------------------
# Effacements (masquage) et données contrefactuelles
# ----------------------------------------------------------------------------
def pointing_chance(boxes, size=PATCH, tol=2):
    """Score qu'obtiendrait un point choisi au hasard dans la zone :
    part moyenne de la surface couverte par la boîte (avec la même tolérance)."""
    fr = []
    for x1, y1, x2, y2 in boxes:
        w = min(size, x2 + tol + 1) - max(0, x1 - tol)
        h = min(size, y2 + tol + 1) - max(0, y1 - tol)
        fr.append(max(0, w) * max(0, h) / size ** 2)
    return float(np.mean(fr))


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
COST_RATIOS = (5, 10, 20)


def decision_rules(m, Xva, y_va, Xte, y_te):
    """Rappel / précision sur le test selon la RÈGLE de choix du seuil
    (toujours choisi sur la validation, jamais sur le test)."""
    pv, pt = predict(m, Xva), predict(m, Xte)
    rules = {"cible_rappel_0.90": threshold_for_recall(y_va, pv)}
    for r in COST_RATIOS:
        rules[f"cout_{r}_pour_1"] = threshold_by_cost(y_va, pv, cost_fn=r)
    out = {}
    for name, thr in rules.items():
        mt = metrics(y_te, pt, thr)
        out[name] = {k: mt[k] for k in ("seuil", "rappel", "precision", "f1", "vp", "fp", "fn", "vn")}
    return out


def evaluate(m, thr, Xte, y_te, P_te_pos, erased, controls, cams_input, box_in_patch, device):
    res = metrics(y_te, predict(m, Xte), thr)
    p0 = predict(m, to_tensor(P_te_pos, device))
    res["masquage"] = {"proba_moyenne_originale": round(float(p0.mean()), 4)}
    for name, arr in erased.items():
        p1 = predict(m, to_tensor(arr, device))
        res["masquage"][f"defaut_efface_{name}"] = round(float(p1.mean()), 4)
        res["masquage"][f"baisse_defaut_efface_{name}"] = round(float((p0 - p1).mean()), 4)
        res["masquage"][f"detecte_apres_effacement_{name}"] = round(float((p1 >= thr).mean()), 4)
    for name, arr in controls.items():
        p1 = predict(m, to_tensor(arr, device))
        res["masquage"][f"baisse_zone_saine_effacee_{name}"] = round(float((p0 - p1).mean()), 4)
    res["masquage"]["detecte_avant"] = round(float((p0 >= thr).mean()), 4)
    cams = grad_cam(m, to_tensor(cams_input, device))
    res["gradcam_pointing_game"] = round(pointing_game(cams, box_in_patch), 4)
    return res, p0, cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cplid_crops_256.npz")
    ap.add_argument("--out", default="results/etape4_cnn")
    ap.add_argument("--bpnn-ref", default="results/etape3_zones_64/metriques_zones.json",
                    help="métriques du BPNN par zones 64 x 64 (référence)")
    ap.add_argument("--seeds", type=int, default=3, help="nombre d'entraînements par variante")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--variants", default="standard,contrefactuel")
    ap.add_argument("--reuse", action="store_true",
                    help="recharger les modèles déjà entraînés (cnn_<variante>_graine<i>.pt) au lieu de réentraîner")
    ap.add_argument("--plot-only", action="store_true",
                    help="refaire seulement la synthèse et la figure de comparaison depuis le JSON")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.plot_only:
        with open(f"{args.out}/metriques_cnn.json", encoding="utf-8") as f:
            return finalize(json.load(f), args.out, args.bpnn_ref)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {device}")

    # --- Mêmes zones et même découpage que l'étape 3 (patch_model.py) -------
    pm.PATCH, pm.JITTER = PATCH, PATCH // 5
    rng = np.random.default_rng(pm.SEED)
    d = np.load(args.data)
    keep = (d["y"] == 1) & (d["defect_box"][:, 0] >= 0)
    X_img, boxes, groups = d["X"][keep], d["defect_box"][keep], d["groups"][keep]
    assert X_img.shape[1] == PATCH, "Utilise les découpes 256 x 64 (prepare_data.py --width 256 --height 64)"
    P, y, owner, pos_x = pm.extract_patches(X_img, boxes, rng)
    tr, va, te = split(y.astype(int), groups[owner])
    print(f"Zones : {len(y)} | train {len(tr)} | val {len(va)} | test {len(te)}  (défauts test : {int(y[te].sum())})")

    # --- Jeux de test pour le masquage (images jamais vues) -----------------
    te_pos = te[y[te] == 1]
    rng_t = np.random.default_rng(123)
    erased, controls = {}, {}
    erased["x1"], _ = erase_defect(X_img, boxes, owner, pos_x, te_pos, 1.0, PATCH)
    erased["x1.5"], _ = erase_defect(X_img, boxes, owner, pos_x, te_pos, 1.5, PATCH)
    controls["x1"] = erase_elsewhere(X_img, boxes, owner, pos_x, te_pos, rng_t, 1.0, PATCH)
    controls["x1.5"] = erase_elsewhere(X_img, boxes, owner, pos_x, te_pos, rng_t, 1.5, PATCH)
    box_in_patch = np.array([[boxes[owner[i]][0] - pos_x[i], boxes[owner[i]][1],
                              boxes[owner[i]][2] - pos_x[i], boxes[owner[i]][3]] for i in te_pos])

    # --- Données contrefactuelles (entraînement seulement) ------------------
    rng_c = np.random.default_rng(7)
    tr_pos, tr_neg = tr[y[tr] == 1], tr[y[tr] == 0]
    cf_pos_erased, _ = erase_defect(X_img, boxes, owner, pos_x, tr_pos, 1.5, PATCH)            # étiquette 0
    cf_pos_ctrl = erase_elsewhere(X_img, boxes, owner, pos_x, tr_pos, rng_c, 1.5, PATCH)       # étiquette 1
    neg_sub = rng_c.choice(tr_neg, size=len(tr_pos), replace=False)
    cf_neg_ctrl = erase_elsewhere(X_img, boxes, owner, pos_x, neg_sub, rng_c, 1.5, PATCH)      # étiquette 0
    P_cf = np.concatenate([P[tr], cf_pos_erased, cf_pos_ctrl, cf_neg_ctrl])
    y_cf = np.concatenate([y[tr], np.zeros(len(tr_pos)), np.ones(len(tr_pos)), np.zeros(len(neg_sub))])

    Yva = torch.tensor(y[va], dtype=torch.float32, device=device)
    Xva, Xte = to_tensor(P[va], device), to_tensor(P[te], device)
    train_sets = {
        "standard": (to_tensor(P[tr], device), torch.tensor(y[tr], dtype=torch.float32, device=device)),
        "contrefactuel": (to_tensor(P_cf, device), torch.tensor(y_cf, dtype=torch.float32, device=device)),
    }

    report = {"zones": {"total": len(y), "train": len(tr), "val": len(va), "test": len(te),
                        "train_contrefactuel": int(len(y_cf))},
              "parametres_cnn": n_params(SmallCNN()), "variantes": {}}
    keep_for_fig = {}
    for variant in args.variants.split(","):
        Xtr, Ytr = train_sets[variant]
        runs, kept = [], []
        for s in range(args.seeds):
            print(f"\n=== CNN {variant} - entraînement {s + 1}/{args.seeds} ===")
            t0 = time.time()
            path = f"{args.out}/cnn_{variant}_graine{s}.pt"
            if args.reuse and os.path.exists(path):
                m = SmallCNN().to(device)
                m.load_state_dict(torch.load(path, map_location=device))
                m.best_epoch, m.hist = 0, None
                print("  (modèle déjà entraîné rechargé)")
            else:
                m = train(Xtr, Ytr, Xva, Yva, seed=s, device=device, epochs=args.epochs, verbose=False)
            thr = threshold_for_recall(y[va], predict(m, Xva))
            res, p0, cams = evaluate(m, thr, Xte, y[te], P[te_pos], erased, controls, P[te_pos],
                                     box_in_patch, device)
            res["decision"] = decision_rules(m, Xva, y[va], Xte, y[te])
            res["meilleure_epoque"] = int(m.best_epoch)
            res["duree_s"] = round(time.time() - t0, 1)
            runs.append(res); kept.append((m, thr, p0, cams))
            mk = res["masquage"]
            dc = res["decision"]["cout_10_pour_1"]
            print(f"  test : PR-AUC {res['pr_auc']:.2f} | règle rappel 0,90 : rappel {res['rappel']:.2f} / "
                  f"précision {res['precision']:.2f} | règle coût 10:1 : rappel {dc['rappel']:.2f} / "
                  f"précision {dc['precision']:.2f}")
            print(f"  masquage (boîte x1,5) : baisse si défaut effacé {mk['baisse_defaut_efface_x1.5']:.2f} | "
                  f"si zone saine effacée {mk['baisse_zone_saine_effacee_x1.5']:.2f} | "
                  f"Grad-CAM sur le défaut {res['gradcam_pointing_game']:.0%}")
        flat = lambda r: {**{k: v for k, v in r.items() if isinstance(v, (int, float))}, **r["masquage"],
                          **{f"{rule}_{k}": r["decision"][rule][k] for rule in r["decision"]
                             for k in ("rappel", "precision")}}
        keys = [k for k in flat(runs[0]) if k not in ("vp", "fp", "fn", "vn", "duree_s", "meilleure_epoque")]
        summary = {k: {"moyenne": round(float(np.mean([flat(r)[k] for r in runs])), 4),
                       "ecart_type": round(float(np.std([flat(r)[k] for r in runs])), 4)} for k in keys}
        # --- Ensemble : moyenne des probabilités des modèles des différentes graines
        models = [k[0] for k in kept]
        thr_e = threshold_for_recall(y[va], predict(models, Xva))
        ens, _, _ = evaluate(models, thr_e, Xte, y[te], P[te_pos], erased, controls, P[te_pos],
                             box_in_patch, device)
        ens["decision"] = decision_rules(models, Xva, y[va], Xte, y[te])
        ens["nb_modeles"] = len(models)
        ens["seuils_individuels"] = [r["seuil"] for r in runs]
        dc = ens["decision"]["cout_10_pour_1"]
        print(f"  >>> ENSEMBLE des {len(models)} modèles : PR-AUC {ens['pr_auc']:.2f} | règle rappel 0,90 : "
              f"rappel {ens['rappel']:.2f} / précision {ens['precision']:.2f} | règle coût 10:1 : rappel "
              f"{dc['rappel']:.2f} / précision {dc['precision']:.2f} | baisse si défaut effacé "
              f"{ens['masquage']['baisse_defaut_efface_x1.5']:.2f}")
        report["variantes"][variant] = {"executions": runs, "synthese": summary, "ensemble": ens}
        mid = int(np.argsort([r["pr_auc"] for r in runs])[len(runs) // 2])
        keep_for_fig[variant] = kept[mid]
        for s_, (m_, *_rest) in enumerate(kept):          # tous les modèles, pour l'ensemble
            torch.save(m_.state_dict(), f"{args.out}/cnn_{variant}_graine{s_}.pt")

    report["gradcam_pointing_hasard"] = round(pointing_chance(box_in_patch), 4)
    # --- Figures ------------------------------------------------------------
    for v, (m, thr, p0, cams) in keep_for_fig.items():
        if m.hist is not None:                       # pas de courbes pour un modèle rechargé
            fig, ax1 = plt.subplots(figsize=(6.5, 4))
            ax1.plot(m.hist["train_loss"], c="tab:blue")
            ax1.set_xlabel("époque"); ax1.set_ylabel("perte d'entraînement", color="tab:blue")
            ax2 = ax1.twinx(); ax2.plot(m.hist["val_pr_auc"], c="tab:orange")
            ax2.set_ylabel("PR-AUC validation", color="tab:orange")
            ax1.axvline(m.best_epoch - 1, ls="--", c="grey")
            ax1.set_title(f"CNN {v} - apprentissage (meilleure époque : {m.best_epoch})")
            fig.tight_layout(); fig.savefig(f"{args.out}/cnn_{v}_courbes.png", dpi=120); plt.close(fig)

        p1 = predict(m, to_tensor(erased["x1.5"], device))
        k = min(8, len(te_pos)); order = np.argsort(-p0)[:k]
        fig, axs = plt.subplots(3, k, figsize=(1.7 * k, 5.6), squeeze=False)
        for j, i in enumerate(order):
            x1, y1, x2, y2 = box_in_patch[i]
            axs[0, j].imshow(P[te_pos[i]], cmap="gray", vmin=0, vmax=255)
            axs[0, j].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.2))
            axs[0, j].set_title(f"p = {p0[i]:.2f}", fontsize=9)
            axs[1, j].imshow(P[te_pos[i]], cmap="gray", vmin=0, vmax=255)
            axs[1, j].imshow(cams[i], cmap="jet", alpha=0.45)
            axs[2, j].imshow(erased["x1.5"][i], cmap="gray", vmin=0, vmax=255)
            axs[2, j].set_title(f"p = {p1[i]:.2f}", fontsize=9)
            for r in range(3):
                axs[r, j].axis("off")
        for r, lab in enumerate(["défaut (vert)", "Grad-CAM", "défaut effacé"]):
            axs[r, 0].text(-0.12, 0.5, lab, transform=axs[r, 0].transAxes, rotation=90, va="center", ha="right")
        fig.suptitle(f"CNN {v} : où il regarde, et sa probabilité quand on efface le défaut")
        fig.tight_layout(); fig.savefig(f"{args.out}/cnn_{v}_gradcam_masquage.png", dpi=120); plt.close(fig)

    finalize(report, args.out, args.bpnn_ref)


def finalize(report, out, bpnn_ref_path="results/etape3_zones_64/metriques_zones.json"):
    """Ajoute la référence BPNN, affiche la synthèse, écrit le JSON et la
    figure de comparaison. Appelée à la fin de main() ou seule (--plot-only)."""
    bpnn_ref = None
    if bpnn_ref_path and os.path.exists(bpnn_ref_path):
        with open(bpnn_ref_path, encoding="utf-8") as f:
            r = json.load(f)
        mk = r["test_masquage"]
        bpnn_ref = {k: r["resultats"]["bpnn_hog"][k] for k in ("pr_auc", "rappel", "precision", "f1")}
        bpnn_ref.update({"baisse_defaut_efface_x1": mk["baisse_moyenne"],
                         "baisse_defaut_efface_x1.5": mk.get("baisse_defaut_efface_x1.5"),
                         "baisse_zone_saine_effacee_x1.5": mk.get("baisse_zone_saine_effacee_x1.5")})
    report["reference_bpnn"] = bpnn_ref

    print("\n=== Synthèse (moyenne ± écart-type sur les graines) ===")
    if bpnn_ref:
        print(f"  BPNN+HOG          PR-AUC {bpnn_ref['pr_auc']:.2f}        | rappel {bpnn_ref['rappel']:.2f} | "
              f"précision {bpnn_ref['precision']:.2f} | baisse si défaut effacé (x1,5) "
              f"{bpnn_ref['baisse_defaut_efface_x1.5']:.2f} (zone saine : {bpnn_ref['baisse_zone_saine_effacee_x1.5']:.2f})")
    for v, rep in report["variantes"].items():
        S = rep["synthese"]
        print(f"  CNN {v:13s} PR-AUC {S['pr_auc']['moyenne']:.2f} ± {S['pr_auc']['ecart_type']:.2f} | "
              f"rappel {S['rappel']['moyenne']:.2f} | précision {S['precision']['moyenne']:.2f} | "
              f"baisse si défaut effacé (x1,5) {S['baisse_defaut_efface_x1.5']['moyenne']:.2f} "
              f"(zone saine : {S['baisse_zone_saine_effacee_x1.5']['moyenne']:.2f}) | "
              f"Grad-CAM sur le défaut {S['gradcam_pointing_game']['moyenne']:.0%} "
              f"(hasard : {report['gradcam_pointing_hasard']:.0%})")
        E = rep.get("ensemble")
        if E:
            print(f"  ↳ ensemble ({E['nb_modeles']} modèles) PR-AUC {E['pr_auc']:.2f} | baisse si défaut effacé (x1,5) "
                  f"{E['masquage']['baisse_defaut_efface_x1.5']:.2f} (zone saine : "
                  f"{E['masquage']['baisse_zone_saine_effacee_x1.5']:.2f})")
            for rule, d in E.get("decision", {}).items():
                print(f"      seuil « {rule} » = {d['seuil']:.3f} -> rappel {d['rappel']:.2f} | précision "
                      f"{d['precision']:.2f} | défauts manqués {d['fn']} | fausses alertes {d['fp']}")
    with open(f"{out}/metriques_cnn.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Comparaison : performance et « regarde-t-il le défaut ? » (boîte x1,5)
    names, perf, perf_e, drop, drop_e, ctrl = [], [], [], [], [], []
    if bpnn_ref:
        names.append("BPNN + HOG"); perf.append(bpnn_ref["pr_auc"]); perf_e.append(0)
        drop.append(bpnn_ref["baisse_defaut_efface_x1.5"]); drop_e.append(0)
        ctrl.append(bpnn_ref["baisse_zone_saine_effacee_x1.5"])
    for v, rep in report["variantes"].items():
        S = rep["synthese"]
        names.append(f"CNN {v}"); perf.append(S["pr_auc"]["moyenne"]); perf_e.append(S["pr_auc"]["ecart_type"])
        drop.append(S["baisse_defaut_efface_x1.5"]["moyenne"])
        drop_e.append(S["baisse_defaut_efface_x1.5"]["ecart_type"])
        ctrl.append(S["baisse_zone_saine_effacee_x1.5"]["moyenne"])
    x = np.arange(len(names))
    col = (["#9aa5b1"] if bpnn_ref else []) + ["#4c8fd6", "#0b3d91"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axs[0].bar(x, perf, yerr=perf_e, capsize=4, color=col[:len(names)])
    for xi, v, e in zip(x, perf, perf_e):
        axs[0].text(xi, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axs[0].axhline(0.25, ls=":", c="grey")
    axs[0].text(-0.45, 0.27, "hasard (0,25)", color="grey", fontsize=8)
    axs[0].set_xticks(x, names); axs[0].set_ylim(0, 1.12)
    axs[0].set_title("Performance sur les zones de test (PR-AUC)")
    axs[1].bar(x - 0.18, drop, 0.36, yerr=drop_e, capsize=4, color=col[:len(names)], label="défaut effacé")
    axs[1].bar(x + 0.18, ctrl, 0.36, color="#d0d5db", label="zone saine effacée (contrôle)")
    for xi, v, e in zip(x, drop, drop_e):
        axs[1].text(xi - 0.18, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axs[1].set_xticks(x, names); axs[1].set_ylim(0, 1.0); axs[1].legend(fontsize=8, loc="upper left")
    axs[1].set_title("Regarde-t-il le défaut ? Baisse de probabilité")
    fig.tight_layout(); fig.savefig(f"{out}/bpnn_vs_cnn.png", dpi=120); plt.close(fig)
    # Stabilité : chaque modèle seul vs l'ensemble, pour deux règles de seuil
    ens_v = [v for v, rep in report["variantes"].items() if rep.get("ensemble", {}).get("decision")]
    if ens_v:
        rules = [("cible_rappel_0.90", "règle\nrappel ≥ 0,90"), ("cout_10_pour_1", "règle\ncoût 10:1")]
        groups = [(v, r, lab) for v in ens_v for r, lab in rules]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for ax, metric, title in [(axs[0], "rappel", "Rappel (défauts trouvés)"),
                                  (axs[1], "precision", "Précision (alertes justes)")]:
            for j, (v, r, lab) in enumerate(groups):
                rep = report["variantes"][v]
                vals = [e["decision"][r][metric] for e in rep["executions"]]
                ev = rep["ensemble"]["decision"][r][metric]
                ax.scatter([j - 0.12] * len(vals), vals, color="#9aa5b1", s=40, zorder=3,
                           label="modèle seul (1 graine)" if j == 0 else None)
                ax.scatter([j + 0.12], [ev], color="#0b3d91", marker="D", s=70, zorder=3,
                           label="ensemble des modèles" if j == 0 else None)
                ax.text(j + 0.22, ev, f"{ev:.2f}", va="center", fontsize=9)
            if metric == "rappel":
                ax.axhline(TARGET_RECALL, ls="--", c="grey", lw=1)
            ax.set_xticks(range(len(groups)), [f"CNN {v}\n{lab}" for v, r, lab in groups], fontsize=8)
            ax.set_xlim(-0.5, len(groups) - 0.3); ax.set_title(title)
            for k in range(1, len(ens_v)):
                ax.axvline(k * len(rules) - 0.5, c="#e0e0e0", lw=1)
        axs[0].set_ylim(0.5, 1.03); axs[0].legend(fontsize=8, loc="lower left")
        fig.suptitle("Stabilité de la décision sur le test : modèles seuls contre ensemble, selon la règle de seuil")
        fig.tight_layout(); fig.savefig(f"{out}/stabilite_ensemble.png", dpi=120); plt.close(fig)

    print(f"\nRésultats écrits dans {out}/")


if __name__ == "__main__":
    main()
