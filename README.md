Optimisation de la Valeur Client chez Tunisie Telecom
📊 Segmentation Comportementale & Scoring Prédictif d'Appétence DATA
Ce dépôt contient le code source complet et l'architecture technique du Projet de Fin d'Études (PFE) mené au sein de la Direction des Affaires Commerciales et Marketing de Tunisie Telecom, pour l'obtention du Mastère ISISD (Ingénierie des Systèmes Informatiques et Data Science) à la FSEGT - Université de Tunis El Manar.

👥 Équipe & Encadrement
Réalisé par : Melek Habessi

Encadrant universitaire : Mr. Massoud Jemaai

Encadrant professionnel : Mr. Brini Elyes

Année universitaire : 2025-2026

📌 Contexte & Problématique
Le marché des télécommunications en Tunisie est hautement concurrentiel et saturé (taux de pénétration > 100%), rendant l'acquisition de nouveaux clients extrêmement difficile. Le principal levier de croissance réside désormais dans la maximisation de la valeur des clients existants à travers l'usage de la DATA (forfaits internet mobile) et la rétention des abonnés.

Chiffres Clés du Parc Analysé :
31 200 clients actifs analysés (Données de Décembre 2025).

84,7% des clients possèdent déjà un forfait DATA.

15,3% de clients non-DATA à cibler activement.

5,36% de taux de Churn (clients présentant un risque de départ).

Objectifs du Projet :
Explore & Segmenter : Profiler le comportement de consommation des 31 200 clients actifs.

Prédire l'Appétence DATA : Identifier précisément quels clients non-DATA sont les plus susceptibles de souscrire à un forfait.

Détecter le Churn en Amont : Alerter sur les clients à forte probabilité de départ pour mener des actions de rétention.

Industrialiser & Déployer : Mettre à disposition un dashboard interactif complet et un système de scoring de masse automatisé pour les équipes marketing et commerciales.

🛠️ Architecture de la Solution
Le projet suit rigoureusement la méthodologie standard CRISP-DM et s'articule autour d'une architecture à 5 couches :

Sources de Données : 8 fichiers Excel bruts fournis par Tunisie Telecom (Historique recharges, usages, USSD...).

Pipeline ETL : Script preprocessing_pfe_fixed.py de nettoyage, de traitement des anomalies et d'agrégation.

Base Analytique (ABT) : Format final unifié de 31 200 clients × 33 variables, nettoyé et sans doublons.

Couche Modélisation : Calcul de scores RFM (7 segments), clustering non supervisé K-Means (K=4), et modèles supervisés XGBoost.

Déploiement & MLOps : Interface décisionnelle Streamlit de 9 pages et suivi de la performance via MLflow.

🧮 Modélisation & Résultats Algorithmiques
1. Segmentation Marketing (RFM & K-Means)
Analyse RFM : Notation par quintiles (Récence, Fréquence, Montant) pour générer un score de 3 à 15, permettant de mapper le parc sur 7 segments marketing actionnables (ex: Champions pour offres VIP, Clients Fidèles pour upselling DATA, Clients à Risque pour actions de rétention urgentes).

Clustering K-Means : Partitionnement non supervisé optimal identifiant 4 profils comportementaux globaux.

2. Modèles Prédictifs (XGBoost)
Modèle 1 : Appétence DATA

Performance : AUC Cross-Validation (5-Fold) = 0,7180 ± 0,0062.

Variables Clés : Montant total rechargé, Ancienneté du client, Type de terminal (5G/4G/3G/2G).

Note de Conception (Anti-Data Leakage) : Détection et correction d'une fuite de données majeure sur la variable MNT_FORFAIT_TOT. Sa suppression a éliminé le biais (l'AUC est passée de 0,99 à 0,7180), garantissant la robustesse du modèle en production.

Modèle 2 : Détection du Churn

Performance : AUC Cross-Validation (5-Fold) = 0,9504 ± 0,0037.

Gestion de l'Imbalance : Utilisation du paramètre scale_pos_weight = 17.7 pour compenser le déséquilibre des classes (seulement 5,36% de churn).

Variable Clé : La Récence d'activité (corrélation de +0,54 avec la cible).

⚙️ Pipeline d'Exécution (7 Scripts Python)
Le projet est entièrement automatisé et structuré de manière séquentielle :

analyse_descriptive_v2.py : Analyse exploratoire de données (EDA) générant 16 visualisations.

preprocessing_pfe_fixed.py : Pipeline ETL consolidant les 8 sources Excel en une ABT saine (0 doublon, 0 valeur manquante).

rfm_analysis.py : Calcul des métriques RFM et classification des 7 segments.

train_model_v2.py : Entraînement, optimisation et sérialisation du modèle XGBoost DATA.

churn_model.py : Entraînement et sérialisation du modèle XGBoost Churn.

batch_scoring_fixed.py : Scoring de masse du parc complet et génération de 3 fichiers CSV de campagnes ciblées directement exploitables par le CRM.

dashboard_tt_final.py : Lancement de l'interface graphique Streamlit.

🖥️ Application & Dashboard Streamlit (9 Pages)
L'application fournit une interface interactive complète structurée comme suit :

P1 à P4 : Analyses Décisionnelles – Indicateurs de performance (KPIs) globaux, analyses temporelles, cartographies géographiques des revenus et zooms sur les usages (Voix, SMS, DATA).

P5 & P6 : Couche IA & Évaluation – Visualisation des clusters K-Means et des métriques de validation des modèles de Machine Learning.

P7 : 🎯 Prédiction Individuelle – Saisie de l'ID d'un abonné pour afficher son score d'appétence en temps réel sur une jauge animée avec recommandations automatisées.

P8 : Matrice RFM – Cartographie dynamique des 7 segments marketing.

P9 : 🚨 Risque Churn – Listing en temps réel du Top 15 des clients les plus à risque avec option d'export CSV pour l'équipe de fidélisation.

🤖 Gouvernance MLOps (MLflow)
Suivi rigoureux du cycle de vie des modèles via MLflow Tracking :

Stockage automatique des hyperparamètres, des métriques de performance par fold et des fichiers sérialisés (.joblib).

Surveillance de la dérive des performances (Model Drift) avec alertes automatisées si la variance entre les folds dépasse le seuil critique de 2%.
