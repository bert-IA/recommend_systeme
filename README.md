# recommend_systeme# Présentation de la Recommandation de Contenu

## Introduction
Cette présentation couvre le développement et le déploiement d'un modèle de recommandation de contenu en utilisant une architecture serverless sur Azure.

## Les Données
- **Utilisateurs** : 322 897
- **Interactions** : 2 988 180
- **Articles** : 46 033
- **Catégories** : 461

## Analyse des Clics des Utilisateurs
- Répartition du nombre de clics par utilisateur.

## Analyse des Catégories
- Répartition du nombre de catégories vues.
- Répartition du nombre d'articles par catégorie.

## La Modélisation
- Sélection des user_id suivant le nombre de clics.
- Création d’une variable ‘rating’.
- Modèles utilisés : Modèle basé sur les embeddings des articles, Modèle ALS, Modèle BPR.
- Comparaison des métriques et analyse des recommandations.

## Preprocessing
- Séparation des utilisateurs en 3 groupes suivant le nombre de clics.
- Echantillonnage et vérification de la représentativité.
- Feature engineering : nombre de clics par utilisateur et par article.

## Les Modèles
### Modèle basé sur les Embeddings
### Modèle ALS
- Matrices de facteurs latents.
- Optimisation alternée et descente de gradient.
- Calcul du score et sélection des articles recommandés.

### Modèle BPR
- Embeddings de facteurs latents.
- Optimisation par rang personnalisé.
- Prédiction des scores et sélection des articles recommandés.

## Fine Tuning
- Cross-validation avec 5 lots.
- Paramètres optimisés : factors, regularization, iterations, alpha_val.

## Les Résultats
- Comparaison des modèles : Embeddings naïf, ALS, BPR.
- Métriques : Recall, HitRate, Match_cat.

## Déploiement de l’API
- Description fonctionnelle et schéma de l’architecture.
- Fonctionnalités : ajout d’utilisateurs et d’articles, réentraînement du modèle, recommandations personnalisées.

## Architecture
- Azure Functions : Scalabilité automatique et modèle serverless.
- Azure Blob Storage : Stockage massif et accès rapide.
- Azure Event Grid : Gestion des événements en temps réel et scalabilité.

## Tests du Déploiement
- Tests de recommandation et d’ajout d’utilisateurs.
- Vérification de la réinitialisation du modèle.

