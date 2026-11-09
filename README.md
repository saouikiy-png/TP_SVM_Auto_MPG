# TP SVM — Approximation de la consommation de carburant (Auto MPG)

[YOUSRA SAOUIKI ] — [IDIA / FSTT] 

## Objectif
Prédire la consommation `mpg` du jeu de données Auto MPG (`seaborn.load_dataset("mpg")`, 392 échantillons après nettoyage) avec des SVM :
1. **Classification (SVC)** : consommation faible / moyenne / élevée.
2. **Régression (SVR linéaire)** : perte ε-insensible.

Chaque modèle est implémenté **depuis zéro (numpy)** puis comparé à scikit-learn.

## Contenu du dépôt
| Fichier                 | Description                            |
|-------------------------|----------------------------------------|
| `TP_SVM_Auto_MPG.ipynb` | Notebook complet (code + explications) |
| `README.md`             | Ce fichier                             |

## Installation et exécution
```bash
pip install numpy pandas seaborn matplotlib scikit-learn jupyter
jupyter notebook TP_SVM_Auto_MPG.ipynb
```
Le dataset est téléchargé automatiquement par seaborn (connexion internet nécessaire). Les graines aléatoires sont fixées : les résultats sont reproductibles.

## Démarche
**Prétraitement** : suppression des 6 lignes sans `horsepower` et de `name` ; one-hot de `cylinders` et `origin` ; standardisation des variables continues (moyenne et écart-type calculés sur le train uniquement, pour éviter la fuite de données) ; découpage 75 % / 25 %.

**Discrétisation de `mpg`** : terciles (seuils ≈ 18.7 et 26.9 mpg) → 3 classes équilibrées. Comparés à des seuils fixes (20 / 30), qui donnent des classes déséquilibrées.

**SVC from scratch** : SMO simplifié, noyaux linéaire, RBF et polynomial, extension multi-classes « un contre un ».

**SVR from scratch** : SVR linéaire, perte ε-insensible, descente de sous-gradient.

## Résultats principaux
| Modèle                        | Perso | scikit-learn |
|-------------------------------|-------|--------------|
| SVC linéaire (accuracy test)  | 0.908 | 0.898        |
| SVC RBF                       | 0.929 | 0.939        |
| SVC polynomial                | 0.898 | 0.908        |
| SVR linéaire (RMSE test, mpg) | 2.864 | 2.865        |

- Les prédictions du SVC perso et de sklearn coïncident à 97-99 %.
- Notre SVR atteint la même valeur d'objectif que sklearn (58.519).
- **Influence de ε** : plus ε augmente, moins il y a de vecteurs de support (294/294 à ε = 0, 18 % à ε = 0.5). L'erreur de test reste stable jusqu'à ε ≈ 0.3 puis se dégrade.
- **Bruit** : face à des valeurs aberrantes, le SVR est nettement plus robuste que les moindres carrés (RMSE ≈ 2.88 contre 3.42). Face à un bruit gaussien, il fait aussi bien, sans mieux.

## Limites
- Test de 98 exemples seulement : 1 erreur = 1 point d'accuracy, les écarts entre noyaux ne sont pas significatifs.
- SMO simplifié (second indice aléatoire) : plus lent que libsvm.
- Hyperparamètres (C, γ, ε) fixés à la main, sans validation croisée.

## Pistes d'amélioration
Grid search de C, γ et ε par validation croisée ; SVR à noyau RBF ; SMO complet avec heuristique de choix du second indice.
