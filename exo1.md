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

L'objectif est d'extraire les **contours fins** d'une image en niveaux de gris et de produire une image binaire où :
- les contours = **blanc (255)**
- le reste = **noir (0)**

### Étapes

1. **Conversion en niveaux de gris** (si l'image est couleur).

2. **Réduction du bruit** : appliquer un filtre gaussien pour éviter de détecter des faux contours dus au bruit.
   ```python
   from skimage.filters import gaussian
   img_floue = gaussian(img, sigma=1)
   ```

3. **Détection de contours avec l'opérateur de Canny** :
   Canny est l'algorithme le plus adapté pour obtenir des contours fins et précis. Il fonctionne en 4 sous-étapes :
   - Calcul du gradient (Sobel en x et y)
   - Calcul de l'amplitude et de la direction du gradient
   - **Suppression des non-maxima** → amincit les contours à 1 pixel d'épaisseur
   - **Seuillage par hystérésis** → ne conserve que les vrais contours

   ```python
   from skimage.feature import canny
   contours = canny(img, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
   # contours est un tableau booléen : True = contour, False = fond
   ```

4. **Conversion en image binaire** :
   ```python
   import numpy as np
   img_binaire = (contours * 255).astype(np.uint8)
   # pixels de contour → 255 (blanc), reste → 0 (noir)
   ```

### Résultat

On obtient une image binaire dans laquelle les contours sont représentés par des lignes blanches fines d'**un seul pixel d'épaisseur**, sur fond noir.

> 💡 **Pourquoi Canny et pas Sobel directement ?** Sobel donne des contours épais (plusieurs pixels de large). Il faudrait un seuillage supplémentaire. Canny inclut déjà la suppression des non-maxima qui affine automatiquement les contours.

---

## Question 6 — Égalisation d'histogramme vs Rehaussement d'image

### Définitions

| Technique | Objectif | Mécanisme |
|---|---|---|
| **Égalisation d'histogramme** | Améliorer le **contraste global** d'une image en redistribuant uniformément les niveaux de gris. | Utilise la **fonction de répartition cumulative (CDF)** de l'histogramme pour remapper les valeurs de pixels. |
| **Rehaussement d'image** | Améliorer la **qualité visuelle perçue** selon un critère précis (netteté, contraste local, luminosité). | Regroupe plusieurs techniques : étirement de contraste, filtrage passe-haut, gamma, etc. Le résultat est orienté par un **objectif visuel**. |

### Exemple concret

Imaginons une photo prise dans une pièce sombre. Les pixels sont tous concentrés dans les niveaux bas (0–80 sur 255) — l'image est globalement sombre et peu contrastée.

**Avec l'égalisation d'histogramme :**
- La CDF est calculée automatiquement.
- Les niveaux sombres (concentrés) sont étirés sur toute la plage 0–255.
- L'histogramme devient approximativement plat.
- ✅ Résultat : l'image est plus contrastée, mais parfois les zones très lumineuses sont sur-exposées.

**Avec le rehaussement :**
- On peut choisir de faire un **étirement de contraste** sur une plage ciblée (ex. forcer les pixels entre 40 et 120 à occuper 0–255).
- Ou appliquer un **filtre passe-haut** pour accentuer les détails fins (bords, textures).
- Ou corriger le gamma pour ajuster la luminosité perçue.
- ✅ Résultat : plus de contrôle sur le rendu final selon ce qu'on veut mettre en valeur.

### Résumé de la différence

> L'**égalisation** est une transformation **automatique et globale** basée sur la statistique de l'image. Le **rehaussement** est une approche **intentionnelle** qui peut être locale, globale, ou combinée — et qui sert un objectif visuel ou applicatif précis.
