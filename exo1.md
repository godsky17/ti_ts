# Épreuve de Traitement d'Image

---

## Question 1 — Principales caractéristiques d'une image

Une image numérique se définit par plusieurs caractéristiques fondamentales :

- **Résolution** : nombre de pixels en largeur × hauteur (ex. 1920×1080). Plus la résolution est élevée, plus l'image est détaillée.
- **Profondeur de couleur (ou dynamique)** : nombre de bits utilisés pour coder la valeur d'un pixel. Sur 8 bits, on a 256 niveaux de gris (0 à 255). Sur 24 bits (RVB), on obtient 16 millions de couleurs.
- **Espace colorimétrique** : façon dont les couleurs sont représentées. Les plus courants sont RGB (rouge, vert, bleu), niveaux de gris, HSV, ou YCbCr.
- **Nombre de canaux** : une image en niveaux de gris a 1 canal ; une image couleur RVB en a 3 ; une image RVBA (avec transparence) en a 4.
- **Type de données** : les pixels peuvent être codés en entiers non signés (`uint8`, `uint16`) ou en flottants (`float32`).
- **Taille en mémoire** : dépend de la résolution, du nombre de canaux et du type. Ex. : 1920×1080×3 octets ≈ 6 Mo.

---

## Question 2 — Chaîne complète de traitement d'image

Voici les étapes successives d'une chaîne de traitement d'image, de l'acquisition à la décision :

```
Scène réelle
     │
     ▼
1. ACQUISITION
   - Capteur (caméra, scanner, IRM…)
   - Numérisation : conversion analogique → numérique
     │
     ▼
2. PRÉ-TRAITEMENT
   - Correction du bruit (filtrage médian, gaussien)
   - Correction de l'illumination
   - Normalisation des niveaux
     │
     ▼
3. AMÉLIORATION / REHAUSSEMENT
   - Contraste, luminosité
   - Égalisation d'histogramme
   - Filtrages (passe-haut, passe-bas)
     │
     ▼
4. SEGMENTATION
   - Séparation des objets d'intérêt du fond
   - Seuillage, détection de contours (Canny, Sobel)
   - Croissance de régions, clustering
     │
     ▼
5. REPRÉSENTATION & DESCRIPTION
   - Extraction de caractéristiques (forme, texture, couleur)
   - Descripteurs : moments, HOG, histogrammes de couleur
     │
     ▼
6. RECONNAISSANCE / CLASSIFICATION
   - Comparaison avec des modèles (KNN, SVM, CNN…)
     │
     ▼
7. DÉCISION / INTERPRÉTATION
   - Résultat final : catégorie, mesure, diagnostic
```

---

## Question 3 — Opérateurs morphologiques

Les opérateurs morphologiques travaillent sur la forme des objets dans une image (binaire ou en niveaux de gris). Ils s'appuient sur un **élément structurant** (SE), qui est un petit masque de forme définie (carré, croix, disque…).

| Opérateur | Description |
|---|---|
| **Érosion** | Réduit la taille des objets blancs ; supprime les petits détails. Un pixel reste blanc seulement si tous les pixels sous le SE sont blancs. |
| **Dilatation** | Agrandit les objets blancs ; comble les petits trous. Un pixel devient blanc si au moins un pixel sous le SE est blanc. |
| **Ouverture** | Érosion suivie d'une dilatation. Supprime les petits objets parasites sans trop déformer les grands. |
| **Fermeture** | Dilatation suivie d'une érosion. Comble les petits trous et relie les contours proches. |
| **Gradient morphologique** | Différence entre dilatation et érosion. Met en évidence les contours. |
| **Top-Hat** (chapeau haut-de-forme) | Différence entre l'image originale et son ouverture. Extrait les petits éléments brillants. |
| **Black-Hat** | Différence entre la fermeture et l'image originale. Extrait les petits éléments sombres. |

---

## Question 4 — Histogramme : variables, étapes et types

### Code de référence (calcul d'histogramme en Python)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open("image.png").convert("L"))  # image en niveaux de gris

hist = np.zeros(256, dtype=int)   # histogramme initialisé à zéro

for i in range(img.shape[0]):         # parcours des lignes
    for j in range(img.shape[1]):     # parcours des colonnes
        niveau = img[i, j]            # valeur du pixel (0 à 255)
        hist[niveau] += 1             # incrémentation du compteur

plt.bar(range(256), hist)
plt.title("Histogramme")
plt.show()
```

### Rôle des variables

| Variable | Rôle |
|---|---|
| `img` | Tableau 2D NumPy contenant les valeurs des pixels (0 à 255). |
| `hist` | Tableau de 256 cases. Chaque case `hist[k]` compte combien de pixels ont la valeur `k`. |
| `i`, `j` | Indices de ligne et de colonne pour parcourir chaque pixel de l'image. |
| `niveau` | Valeur de gris du pixel courant, utilisée comme **indice** dans `hist`. |
| `hist[niveau] += 1` | À chaque pixel, on incrémente le compteur correspondant à sa valeur. |

### Étapes de l'algorithme

1. **Initialisation** : créer un tableau `hist` de 256 zéros.
2. **Parcours** : lire chaque pixel de l'image ligne par ligne.
3. **Comptage** : utiliser la valeur du pixel comme index et incrémenter la case correspondante.
4. **Affichage** : tracer le tableau `hist` sous forme de courbe ou de barres.

### Différence entre les deux types d'histogramme

| Type | Description |
|---|---|
| **Histogramme simple (absolu)** | Compte le nombre de pixels pour chaque niveau de gris. L'axe Y donne un nombre brut de pixels. Dépend de la taille de l'image. |
| **Histogramme normalisé (probabiliste)** | Chaque valeur est divisée par le nombre total de pixels. L'axe Y donne une fréquence (entre 0 et 1). Permet de comparer des images de tailles différentes et sert de base au calcul de la fonction de répartition (CDF) pour l'égalisation. |

---

## Question 5 — Obtenir une image binaire avec les contours fins en blanc

Pour obtenir une image binaire dans laquelle les contours fins sont blancs et le reste est noir, on part d'une image en niveaux de gris et on applique l'algorithme de **Canny**. La première étape consiste à réduire le bruit de l'image en appliquant un filtre gaussien, car sans ce lissage, le détecteur de contours risque de réagir à des variations parasites plutôt qu'aux vraies bordures des objets. Ensuite, Canny calcule le **gradient** de l'image dans les directions horizontale et verticale (via des filtres de type Sobel), ce qui permet d'estimer à chaque pixel l'intensité et la direction du changement de niveau de gris. L'étape clé qui produit des contours fins est la **suppression des non-maxima** : pour chaque pixel, on ne conserve que ceux qui sont un maximum local dans la direction du gradient, ce qui réduit les contours épais à une ligne d'un seul pixel d'épaisseur. Enfin, un **seuillage par hystérésis** avec deux seuils (bas et haut) permet de ne retenir que les contours significatifs et d'éliminer les fragments isolés. Le résultat est un tableau booléen que l'on convertit en image binaire en multipliant par 255 : les pixels de contour deviennent blancs (255) et tout le reste reste noir (0).

---

## Question 6 — Égalisation d'histogramme vs Rehaussement d'image

L'égalisation d'histogramme et le rehaussement d'image sont deux techniques d'amélioration visuelle, mais elles poursuivent des objectifs différents. L'**égalisation d'histogramme** est une transformation automatique et globale : elle calcule la fonction de répartition cumulative (CDF) de l'histogramme de l'image, puis l'utilise pour redistribuer les niveaux de gris de manière à obtenir un histogramme le plus plat possible. Concrètement, si on prend une photo prise dans une pièce sombre, dont les pixels sont tous concentrés entre les niveaux 0 et 80, l'égalisation va étirer cette plage sur toute l'étendue 0–255 : l'image gagne en contraste global de façon entièrement automatique, sans qu'on ait besoin de choisir des paramètres. Le **rehaussement d'image**, en revanche, est une démarche intentionnelle et orientée : il regroupe un ensemble de techniques (étirement de contraste sur une plage ciblée, filtrage passe-haut pour accentuer les détails, correction gamma pour ajuster la luminosité perçue) que l'on choisit et paramètre selon ce qu'on veut mettre en valeur dans l'image. Pour reprendre le même exemple, plutôt que d'étirer l'histogramme entier, on pourrait choisir de n'agir que sur la plage 40–120 pour préserver certaines zones, ou d'appliquer un filtre accentuant les bords pour rendre les textures plus nettes. En résumé, l'égalisation est aveugle et statistique — elle améliore le contraste sans contrôle fin — tandis que le rehaussement est guidé par un objectif visuel ou applicatif précis.
