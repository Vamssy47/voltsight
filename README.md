# VoltSight : Inspection de lignes électriques par drone et IA

**Du pixel à la décision :** détecter automatiquement les défauts sur les photos de drone des installations électriques, puis prioriser les réparations.

Projet de portfolio en science des données, inspiré des pratiques d'inspection par drone et par robot d'Hydro-Québec (LineScout, LineDrone, projet CableInspect-AD avec Mila).

**Résultat phare :** sur 10 607 vraies photos de drone (jeu InsPLAD), notre détecteur **YOLO11s atteint un mAP@0.5:0.95 de 0,740**, comparable au meilleur modèle de l'article de référence (DetectoRS, 0,721), tout en étant beaucoup plus léger : environ 3 ms par image sur un GPU T4.

## Où en est le projet

| Phase | Contenu | Statut |
| --- | --- | --- |
| 1. Données | Jeu CPLID (isolateurs photographiés par drone), découpes, division sans fuite de données | ✅ |
| 2. Modèle de référence | BPNN codé à la main en NumPy, rétropropagation vérifiée | ✅ |
| 3. Diagnostic des défauts | Détection d'un raccourci, tâche corrigée, CNN contrefactuel, Grad-CAM | ✅ |
| 4. Décision | Ensemble de 3 modèles, seuil fondé sur le coût d'un défaut manqué | ✅ (amélioration prévue : seuil par validation croisée) |
| 5.1–5.2 Équipements réels | InsPLAD : 10 607 photos de drone, 17 équipements, détection YOLO11s | ✅ mAP 0,740 |
| 5.3 Défauts réels | Rouille, nid d'oiseau, pièce cassée sur les équipements détectés | ⏳ prochaine étape |
| 6. Végétation et capteurs | Segmentation des lignes, anomalies de capteurs | ⏳ |
| 7. Score de risque et tableau de bord | Priorisation des interventions, API, carte | ⏳ |

## Résultats clés : détection des équipements sur de vraies photos de drone (phase 5.2)

Le jeu **InsPLAD** contient de vraies photos de drone de lignes électriques (1920 × 1080). On utilise 7 981 images pour l'entraînement et 2 626 pour le test, soit 28 959 boîtes au total. Le modèle est YOLO11s pré-entraîné sur COCO, réentraîné pendant 30 époques sur un GPU T4 (Google Colab).

| | **YOLO11s (VoltSight)** | DetectoRS (article InsPLAD) |
| --- | --- | --- |
| mAP@0.5:0.95 (« Box AP ») | **0,740** | 0,721 |
| mAP@0.5 | 0,906 | — |
| Précision / rappel | 0,90 / 0,87 | — |
| Vitesse d'inférence | environ 3 à 6 ms par image (T4) | modèle beaucoup plus lourd |

**Ce que le modèle fait bien** (AP50-95 supérieur à 0,90) : la plaque d'identification du pylône (0,99), l'isolateur polymère (0,96), le vari-grip (0,96), l'amortisseur spiralé (0,93), la suspension de paratonnerre (0,92) et l'isolateur en verre (0,90).

**Ses points faibles :**

- **Les petites manilles métalliques sont mal détectées.** Les petites et grandes manilles d'isolateur en verre ont une AP de 0,30 et 0,33, avec un rappel d'environ 0,45 : le modèle en rate environ une sur deux. Ce sont des objets minuscules sur une photo réduite à 640 pixels, et ils sont rares dans les données d'entraînement (environ 100 exemples chacun).
- **Le modèle confond parfois des classes voisines,** comme le joug et la suspension de joug. Les boîtes concernées ont une confiance faible (environ 0,45).

**Note d'honnêteté :**

- Le meilleur modèle a été choisi d'après son score sur ce même ensemble de test. Le biais reste faible : les dernières époques ont des scores presque identiques, entre 0,735 et 0,740.
- Les outils d'évaluation diffèrent un peu de ceux de l'article. La conclusion juste est donc : « performance comparable à l'article, avec un modèle beaucoup plus léger ».
- La classe « sphere » (26 exemples) n'apparaît pas dans l'ensemble de test.

**Pistes d'amélioration :**

- images en 1024 pixels ;
- découpage en tuiles (SAHI) pour les petits objets ;
- seuil de confiance plus élevé contre les doublons.

![Courbe précision-rappel YOLO](results/etape5_yolo/insplad_eval__BoxPR_curve.png)

Notebooks : [`05_insplad_exploration.ipynb`](notebooks/05_insplad_exploration.ipynb) · [`06_yolo_detection.ipynb`](notebooks/06_yolo_detection.ipynb)

## Résultats clés : phase de modélisation sur le jeu CPLID (phases 2 à 4)

| | BPNN + HOG | CNN standard | CNN contrefactuel |
| --- | --- | --- | --- |
| PR-AUC (zones de test) | 0,67 | 1,00 | 1,00 |
| Baisse de probabilité quand on efface le défaut | 0,06 | 0,28 | **0,61** |
| Baisse de probabilité quand on efface une zone saine (contrôle) | 0,04 | 0,02 | 0,03 |

- **Le BPNN avait d'abord une PR-AUC de 0,96, mais il trichait.** Il reconnaissait les images synthétiques, pas les défauts ; un test de masquage avec contrôle l'a révélé.
- **Le CNN contrefactuel s'appuie vraiment sur le défaut.** Il a été entraîné avec des paires « avec / sans défaut ».
- **Seuil de décision :** il est fixé selon un coût métier (un défaut manqué = 10 fausses alertes). Le rappel atteint 0,98 à 1,00 sur deux exécutions indépendantes. Le nombre de fausses alertes varie de 1 à 10 sur 150 : c'est une limite connue, due à la petite taille de la validation.

![Comparaison BPNN / CNN](results/etape4_cnn/bpnn_vs_cnn.png)

👉 Démarche complète, chiffres et limites : **[docs/modelisation.md](docs/modelisation.md)**

## Structure

```
voltsight/
├── src/                 code Python (données, BPNN, CNN, évaluation)
├── results/             métriques, figures et modèles entraînés, par étape
├── notebooks/           notebook Google Colab pour tout relancer
├── docs/                rapport de modélisation
├── data/                données (non versionnées, téléchargées par les commandes)
└── requirements.txt
```

## Lancer

**Dans Google Colab**, le plus simple : ouvre `notebooks/voltsight_colab.ipynb`, active le GPU, puis « Tout exécuter ».

**Sur ton ordinateur :**

```bash
git clone https://github.com/Vamssy47/voltsight.git
cd voltsight
pip install -r requirements.txt
git clone --depth 1 https://github.com/InsulatorData/InsulatorDataSet.git data/InsulatorDataSet
python src/prepare_data.py --width 256 --height 64 --out data/cplid_crops_256.npz
python src/patch_model.py --data data/cplid_crops_256.npz --patch 64 --out results/etape3_zones_64
python src/cnn_model.py
```

## Données et licence

- **InsPLAD** : Vieira e Silva et al., 2023, *International Journal of Remote Sensing*, [Mendeley Data](https://data.mendeley.com/datasets/5n3fjgvfyz/1). Usage de recherche ; les images ne sont pas redistribuées ici.
- **CPLID** (Chinese Power Line Insulator Dataset) : Tao et al., 2018, *IEEE Transactions on Systems, Man, and Cybernetics: Systems*. Licence non précisée par les auteurs ; usage académique uniquement. Les images ne sont pas redistribuées ici.
- **Code :** licence MIT.

## Auteur

Vamoussa Diabaté — baccalauréat en sciences des données et intelligence d'affaires, UQAC.
