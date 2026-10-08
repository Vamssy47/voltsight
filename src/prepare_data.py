"""
VoltSight - Étape 1 : préparation des données pour le BPNN
==========================================================

Jeu de données : CPLID (Chinese Power Line Insulator Dataset)
  https://github.com/InsulatorData/InsulatorDataSet

Ce script :
  1. lit les annotations VOC (boîtes des isolateurs et des défauts) ;
  2. découpe chaque isolateur dans l'image d'origine ;
  3. étiquette la découpe : 1 = défectueux (contient une boîte « defect »), 0 = normal ;
  4. normalise la découpe : niveaux de gris, côté long à l'horizontale, 128 x 32 pixels ;
  5. regroupe les quasi-doublons (hachage perceptuel) pour éviter la fuite
     entre entraînement et test ;
  6. sauvegarde le tout dans data/cplid_crops.npz.

Utilisation :
  python src/prepare_data.py   (depuis la racine du dépôt)
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

CROP_W, CROP_H = 128, 32  # taille finale (largeur x hauteur)


# ----------------------------------------------------------------------------
# Lecture des annotations
# ----------------------------------------------------------------------------
def read_boxes(xml_path):
    """Retourne la liste des boîtes (xmin, ymin, xmax, ymax) d'un fichier VOC."""
    if not os.path.exists(xml_path):
        return []
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        out.append(tuple(int(float(bb.find(k).text))
                         for k in ("xmin", "ymin", "xmax", "ymax")))
    return out


def contains(outer, inner, min_frac=0.5):
    """Vrai si au moins `min_frac` de la boîte `inner` est dans `outer`."""
    ix1, iy1 = max(outer[0], inner[0]), max(outer[1], inner[1])
    ix2, iy2 = min(outer[2], inner[2]), min(outer[3], inner[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return area > 0 and inter / area >= min_frac


# ----------------------------------------------------------------------------
# Découpe et normalisation
# ----------------------------------------------------------------------------
def crop_and_normalize(img, box, defect=None, margin=0.05):
    """Découpe l'isolateur, met le côté long à l'horizontale, redimensionne.

    Si `defect` (boîte dans l'image d'origine) est fourni, retourne aussi sa
    position dans la découpe finale (128 x 32), utile pour vérifier plus tard
    que le modèle regarde bien le défaut.
    """
    x1, y1, x2, y2 = box
    mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
    W, H = img.size
    cx1, cy1, cx2, cy2 = max(0, x1 - mw), max(0, y1 - mh), min(W, x2 + mw), min(H, y2 + mh)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("L")
    w, h = crop.size
    rotated = h > w
    if rotated:                              # isolateur vertical -> horizontal
        crop = crop.rotate(90, expand=True)  # rotation anti-horaire
    sx, sy = CROP_W / crop.width, CROP_H / crop.height
    crop = crop.resize((CROP_W, CROP_H), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.uint8)

    if defect is None:
        return arr, (-1, -1, -1, -1)
    # coordonnées du défaut dans la découpe
    dx1, dy1 = max(defect[0], cx1) - cx1, max(defect[1], cy1) - cy1
    dx2, dy2 = min(defect[2], cx2) - cx1, min(defect[3], cy2) - cy1
    if rotated:  # (x, y) -> (y, w - x) pour une rotation de 90° anti-horaire
        dx1, dy1, dx2, dy2 = dy1, w - dx2, dy2, w - dx1
    return arr, (int(dx1 * sx), int(dy1 * sy), int(np.ceil(dx2 * sx)), int(np.ceil(dy2 * sy)))


def dhash(arr, size=8):
    """Hachage perceptuel (difference hash) : 64 bits."""
    small = np.asarray(Image.fromarray(arr).resize((size + 1, size), Image.BILINEAR),
                       dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def group_near_duplicates(hashes, max_dist=6):
    """Union-find : deux découpes à <= max_dist bits de distance = même groupe."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    H = np.array(hashes, dtype=np.uint8)
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0] + i + 1:
            parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cplid", default="data/InsulatorDataSet")
    ap.add_argument("--out", default="data/cplid_crops.npz")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=32)
    args = ap.parse_args()
    global CROP_W, CROP_H
    CROP_W, CROP_H = args.width, args.height

    X, y, source, src_img, dbox = [], [], [], [], []

    # 1) Isolateurs normaux (vraies photos de drone)
    for xml in sorted(glob.glob(f"{args.cplid}/Normal_Insulators/labels/*.xml")):
        name = os.path.splitext(os.path.basename(xml))[0]
        img = Image.open(f"{args.cplid}/Normal_Insulators/images/{name}.jpg")
        for box in read_boxes(xml):
            arr, db = crop_and_normalize(img, box)
            X.append(arr); y.append(0); dbox.append(db)
            source.append("normal"); src_img.append(f"N_{name}")

    # 2) Isolateurs défectueux (images synthétiques, voir README du CPLID)
    for xml in sorted(glob.glob(f"{args.cplid}/Defective_Insulators/labels/insulator/*.xml")):
        name = os.path.splitext(os.path.basename(xml))[0]
        img = Image.open(f"{args.cplid}/Defective_Insulators/images/{name}.jpg")
        defects = read_boxes(f"{args.cplid}/Defective_Insulators/labels/defect/{name}.xml")
        for box in read_boxes(xml):
            inside = [d for d in defects if contains(box, d)]
            arr, db = crop_and_normalize(img, box, inside[0] if inside else None)
            X.append(arr); y.append(int(bool(inside))); dbox.append(db)
            source.append("defective_set"); src_img.append(f"D_{name}")

    X = np.stack(X); y = np.array(y)
    groups = group_near_duplicates([dhash(a) for a in X])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, groups=groups, defect_box=np.array(dbox),
                        source=np.array(source), src_img=np.array(src_img))

    print(f"Découpes : {len(y)}  | normales : {(y == 0).sum()}  | défectueuses : {(y == 1).sum()}")
    print(f"Groupes de quasi-doublons : {len(np.unique(groups))}")
    print(f"Sauvegardé dans {args.out}")


if __name__ == "__main__":
    main()
