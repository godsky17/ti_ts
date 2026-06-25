import numpy as np
import matplotlib.pyplot as plt

def median_filter(image, size=3):
    """Filtre médian manuel — remplace scipy.ndimage.median_filter."""
    pad = size // 2
    out = np.zeros_like(image)
    img_pad = np.pad(image, pad, mode='edge')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            voisins = img_pad[i:i+size, j:j+size].flatten()
            out[i, j] = np.median(voisins)
    return out

# =============================================================
# 🔧 REMPLACE CES VALEURS PAR TON IMAGE RÉELLE (8x8, 0 à 15)
# =============================================================
I = np.array([
    [ 3,  1,  2,  5,  7,  6,  4,  3],
    [ 0,  2,  4,  6,  8,  7,  5,  2],
    [ 1,  3,  5,  8, 10,  9,  6,  3],
    [ 2,  4,  6,  9, 12, 11,  8,  4],
    [ 3,  5,  8, 11, 13, 12,  9,  5],
    [ 4,  6,  9, 10, 11, 10,  7,  4],
    [ 2,  4,  7,  8,  9,  8,  5,  3],
    [ 1,  3,  5,  6,  7,  6,  4,  2],
], dtype=float)

# Pixel cible pour la Q3 — (ligne, col) en base 1
PIXEL_CIBLE = (3, 4)

# =============================================================


# ─────────────────────────────────────────────────────────────
# QUESTION 1 — Histogramme H(n) et histogramme cumulé C(n)
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  QUESTION 1 — Histogramme H(n) et C(n)")
print("=" * 55)

NB_NIVEAUX = 16  # 4 bits → niveaux 0 à 15
niveaux = np.arange(NB_NIVEAUX)

H = np.zeros(NB_NIVEAUX, dtype=int)
for val in I.flatten():
    H[int(val)] += 1

C = np.cumsum(H)

print(f"\n{'n':>3} | {'H(n)':>5} | {'C(n)':>5}")
print("-" * 20)
for n in niveaux:
    print(f"{n:>3} | {H[n]:>5} | {C[n]:>5}")
print(f"\nTotal pixels : {C[-1]} (attendu : {8*8})")


# ─────────────────────────────────────────────────────────────
# QUESTION 2 — Bruit impulsionnel
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  QUESTION 2 — Ajout du bruit et identification")
print("=" * 55)

I_bruitee = I.copy()
# Coordonnées en base 1 → base 0 pour Python
I_bruitee[2-1, 2-1] = 100   # I(2,2) = 100
I_bruitee[5-1, 5-1] = 0     # I(5,5) = 0

print(f"\nI(2,2) original = {I[1,1]}  →  après bruit = {I_bruitee[1,1]}")
print(f"I(5,5) original = {I[4,4]}  →  après bruit = {I_bruitee[4,4]}")
print("\nType de bruit : IMPULSIONNEL (sel et poivre)")
print("  • I(2,2)=100 → valeur hors plage [0,15] → pixel 'sel' (blanc)")
print("  • I(5,5)=0   → pixel isolé à zéro       → pixel 'poivre' (noir)")
print("\nCorrection par filtre médian (voisinage 3×3) :")

I_corrigee = median_filter(I_bruitee, size=3)
print(f"  I(2,2) corrigé = {I_corrigee[1,1]}")
print(f"  I(5,5) corrigé = {I_corrigee[4,4]}")


# ─────────────────────────────────────────────────────────────
# QUESTION 3 — Filtre H1 et calcul du module en (3,4)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  QUESTION 3 — Filtre H1 et module du gradient")
print("=" * 55)

H1 = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]], dtype=float)

H2 = np.array([[-1,  0,  1],
               [-1,  0,  1],
               [-1,  0,  1]], dtype=float)

print("\nH1 (Prewitt vertical — détecte les contours horizontaux) :")
print(H1.astype(int))
print("\nType  : Filtre de Prewitt (dérivée selon Y)")
print("Utilité : Détecte les variations verticales de luminosité,")
print("          c'est-à-dire les bords horizontaux dans l'image.")

# Pixel cible en base 0
r = PIXEL_CIBLE[0] - 1
c = PIXEL_CIBLE[1] - 1

voisinage = I[r-1:r+2, c-1:c+2]
print(f"\nVoisinage 3×3 centré en {PIXEL_CIBLE} :")
print(voisinage.astype(int))

Gy = np.sum(H1 * voisinage)
Gx = np.sum(H2 * voisinage)
module_exact  = np.sqrt(Gx**2 + Gy**2)
module_approx = abs(Gx) + abs(Gy)

print(f"\nGy = convolution(H1, voisinage) = {Gy:.0f}")
print(f"Gx = convolution(H2, voisinage) = {Gx:.0f}")
print(f"|G| exact    = sqrt({Gx:.0f}² + {Gy:.0f}²) = {module_exact:.4f}")
print(f"|G| approché = |{Gx:.0f}| + |{Gy:.0f}|      = {module_approx:.0f}")


# ─────────────────────────────────────────────────────────────
# TRACÉ — 4 figures dans une même fenêtre
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Exercice Traitement d'Image", fontsize=15, fontweight='bold')

# ── Q1a : H(n) ──────────────────────────────────────────────
ax = axes[0, 0]
ax.bar(niveaux, H, color='steelblue', edgecolor='black', width=0.8)
ax.set_title("Q1 — H(n) : Histogramme simple")
ax.set_xlabel("Niveau de gris n")
ax.set_ylabel("Nombre de pixels")
ax.set_xticks(niveaux)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# ── Q1b : C(n) ──────────────────────────────────────────────
ax = axes[0, 1]
ax.bar(niveaux, C, color='coral', edgecolor='black', width=0.8)
ax.set_title("Q1 — C(n) : Histogramme cumulé")
ax.set_xlabel("Niveau de gris n")
ax.set_ylabel("Nombre cumulé de pixels")
ax.set_xticks(niveaux)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# ── Q2 : Image originale vs bruitée ─────────────────────────
ax = axes[1, 0]
diff = np.zeros_like(I_bruitee)
diff[1, 1] = 1   # pixel bruité
diff[4, 4] = 1   # pixel bruité
im = ax.imshow(I_bruitee, cmap='gray', vmin=0, vmax=15)
# Repère les pixels bruités en rouge
for (pr, pc) in [(1, 1), (4, 4)]:
    rect = plt.Rectangle((pc - 0.5, pr - 0.5), 1, 1,
                          linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
ax.set_title("Q2 — Image bruitée (pixels en rouge = bruit)")
plt.colorbar(im, ax=ax)

# ── Q3 : Voisinage et résultat du filtre ────────────────────
ax = axes[1, 1]
im2 = ax.imshow(voisinage, cmap='gray', vmin=0, vmax=15)
ax.set_title(f"Q3 — Voisinage 3×3 en {PIXEL_CIBLE}\n"
             f"Gy={Gy:.0f}  Gx={Gx:.0f}  |G|={module_exact:.2f}")
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(voisinage[i, j]),
                ha='center', va='center',
                color='white' if voisinage[i, j] < 8 else 'black',
                fontsize=14, fontweight='bold')
# Encadre le pixel central
rect = plt.Rectangle((0.5, 0.5), 1, 1,
                      linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im2, ax=ax)

plt.tight_layout()
plt.savefig("exercice_complet.png", dpi=150)
plt.show()