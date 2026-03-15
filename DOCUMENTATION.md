# Documentation technique — Prédiction du trafic urbain à Paris

**Projet NN2026 — M2 TAL, Université Paris Nanterre**  
**Membres : CAO Yue, XU Hongying — INALCO**


## 1. Objectifs du projet

- **Objectif principal** : prédire le flux de trafic (nombre d’usagers par heure) sur les sites de comptage multimodaux à Paris, à partir des données historiques de comptage et des variables météorologiques.
- **Objectifs secondaires** :
  - Constituer un jeu de données fusionnant trafic (Paris Open Data), météo (Open-Meteo) et variables temporelles/saisonnières.
  - Mettre en place un modèle de type LSTM exploitant des séquences temporelles (par ex. 24 h) pour prédire l’heure suivante.
  - Ajouter un modèle tabulaire de type HistGradientBoostingRegressor (HGB) afin de permettre une prédiction en ligne à partir des derniers comptages observés et des prévisions météo.
  - Proposer une interface de prédiction (Streamlit) permettant de choisir un site, une date et une heure et d’afficher la prédiction ainsi qu’un historique récent.
- **Cadre** : projet final du cours Réseaux de neurones (Loïc Grobol), à livrer sous forme de pipeline reproductible et documenté.


## 2. Données utilisées

### 2.1 Origine et statut juridique

| Source | Description | Statut juridique / licence |
|--------|-------------|----------------------------|
| **Paris Open Data — Comptages multimodaux** | Comptages par site, par créneau horaire, par mode (vélo, véhicule, etc.). Données utilisées : 2024–2026 (extrait JSON). | Données publiques [Open Data Paris](https://opendata.paris.fr), réutilisables sous licence Ouverte / Open Licence. |
| **Open-Meteo** | Données météo horaires (température, humidité, précipitations, vent, nébulosité, etc.) pour Paris (lat/lon fixes). | API gratuite, [sans clé](https://open-meteo.com), usage conforme aux conditions d’utilisation du service. |

Aucune donnée personnelle ; pas de restriction particulière pour un usage académique dans le cadre décrit.

### 2.2 Format des données

- **Entrée trafic** : fichier JSON (tableau d’objets) fourni par l’export Paris Open Data. Champs utilisés : `id_site`, `t` (horodatage), `nb_usagers`, `coordonnees_geo` (lat/lon), `label`, `mode`, etc. Fichier volumineux (~2 Go), d’où un prétraitement en flux.
- **Entrée météo** : aucune donnée locale ; le pipeline d’entraînement appelle l’API Open-Meteo Archive (`https://archive-api.open-meteo.com/v1/archive`) avec latitude/longitude Paris (48.8566, 2.3522) et une plage de dates. En phase de prédiction en ligne, l’application peut aussi interroger l’API Open-Meteo Forecast (`https://api.open-meteo.com/v1/forecast`) pour obtenir les variables horaires futures.
- **Données intermédiaires et finales** : CSV (UTF-8), séparateur virgule. Colonnes cibles : `id_site`, `hour` (ou `t`), `nb_usagers` ; après fusion : variables météo + `hour_of_day`, `day_of_week`, `month`, encodages sin/cos (heure, mois, jour de la semaine).

### 2.3 Traitements opérés

1. **Prétraitement du JSON (preprocess_traffic_json.py)**  
   Lecture en flux (ijson ou fallback manuel) pour éviter de charger tout le JSON en mémoire. Pour chaque enregistrement : normalisation des types (chaînes vides pour les null, entier ≥ 0 pour `nb_usagers`), dépliage de `coordonnees_geo` en colonnes `lat`/`lon`. Suppression des lignes sans horodatage `t` (option par défaut). Filtre optionnel par liste d’`id_site`. Sortie : un CSV avec colonnes fixes.

2. **Agrégation trafic (process_traffic_data.py / load_traffic.py)**  
   Lecture du CSV trafic ; normalisation des noms de colonnes (alias type `date` → `t`, `site_id` → `id_site`, etc.) ; parsing des dates en UTC, arrondi à l’heure ; suppression des lignes sans date valide ou sans `nb_usagers` (ou `nb_usagers` < 0). Agrégation par (`id_site`, `hour`) : somme des `nb_usagers`. Sortie : série temporelle par site et par heure.

3. **Récupération météo (fetch_weather.py)**  
   Appels à l’API Archive Open-Meteo avec paramètres (latitude, longitude, `start_date`, `end_date`, variables horaires). Les réponses sont agrégées en un DataFrame puis sauvegardées en CSV (colonnes `time`, `temperature_2m`, `relative_humidity_2m`, `precipitation`, `weather_code`, `wind_speed_10m`, `cloud_cover`, etc.). Le même script expose aussi une fonction forecast utilisée par l’application pour la prédiction en ligne.

4. **Fusion et variables temporelles (build_dataset.py)**  
   Jointure gauche du trafic et de la météo sur l’heure (alignement temporel). Ajout de variables dérivées : `hour_of_day`, `day_of_week`, `month`, `year`, et encodages cycliques (sin/cos) pour l’heure, le mois et le jour de la semaine. Les créneaux sans météo restent dans le jeu avec des NaN météo ; en entraînement, les features sont complétées par `fillna(0)`.

Aucun traitement de type anonymisation ou agrégation de données personnelles ; les identifiants de sites sont des identifiants publics de compteurs.


## 3. Méthodologie

### 3.1 Répartition du travail et organisation

- **Organisation** : dépôt unique (ex. Git) avec séparation claire : scripts de données (`src/`), données et modèles (`data/`), documentation (fichiers `.md`), interface (`app.py`). Pas de capture d’écran de code dans la documentation.
- **Répartition fonctionnelle** :  
  - Pipeline de données : préprocessing JSON → agrégation trafic → récupération météo → fusion et variables temporelles.  
  - Modélisation : un modèle LSTM (séquences 24 h → prédiction heure suivante), entraîné sur l’ensemble des sites du jeu fusionné.  
  - Interface : une seule app Streamlit pour la prédiction (choix site / date / heure, affichage de la prédiction et d’un historique récent).
- **Outils communs** : Python 3, environnement virtuel, `requirements.txt` pour reproductibilité.
- **Répartition concrète entre les membres** :  
  Le projet a été réalisé de manière collaborative, avec des échanges réguliers sur les orientations méthodologiques et les solutions techniques à adopter. La phase initiale de nettoyage, d’exploration des données et d’identification d’une stratégie réalisable a été menée en commun. Dans ce cadre, **Cao Yue** a d’abord implémenté le pipeline de base. **Xu Hongying** l’a ensuite enrichi et amélioré, a pris en charge l’entraînement du modèle **LSTM** et a développé une première version de l’application web. Dans un second temps, **Cao Yue** a entraîné les modèles **HGB** et **GRU**, puis a complété et consolidé le système final en affinant les résultats et en corrigeant les éléments restés incomplets.  
- **Travail commun** : définition du sujet, choix méthodologiques, expérimentation, analyse des résultats et rédaction de la documentation finale.

### 3.2 Identification et résolution des problèmes

- **Volume du JSON** : impossible de charger tout le fichier en mémoire. Solution : lecture en flux (ijson ou parsing manuel par blocs) et écriture incrémentale en CSV.
- **Alignement trafic–météo** : les comptages sont par site et par heure, la météo est par heure pour une seule position. Solution : jointure sur l’heure (floor à l’heure) ; tous les sites d’une même heure reçoivent les mêmes variables météo.
- **Prédiction sans LSTM** : en l’absence de modèle LSTM (fichiers non présents ou erreur), l’application utilise une baseline (moyenne historique par site, heure et jour de la semaine) pour toujours renvoyer une prédiction.
- **Historique insuffisant pour le LSTM** : le LSTM nécessite les 24 dernières heures avant la date/heure cible. Si un site n’a pas assez d’historique dans le jeu chargé, la prédiction retombe sur la baseline avec un message explicatif.

### 3.3 Étapes du projet

1. Définition des objectifs et choix des sources (Paris Open Data, Open-Meteo).  
2. Mise en place du pipeline : préprocessing JSON → CSV, agrégation, récupération météo, fusion + variables temporelles.  
3. Construction du jeu final (CSV) et vérification des plages de dates et des sites.  
4. Implémentation du LSTM (PyTorch) : séquences 24 h, sortie régression (heure suivante), métriques MAE, RMSE, R², MAPE, accuracy within 10 % / 20 %.  
5. Entraînement sur les données complètes (tous les sites), sauvegarde du modèle et des scalers.  
6. Application Streamlit : chargement du jeu, chargement optionnel du LSTM, calcul de la baseline, affichage de la prédiction et d’un graphique d’historique. Une branche HGB en ligne exploite les derniers comptages récupérés depuis l’API Paris Open Data ainsi que la météo forecast.  
7. Rédaction de la documentation technique et des guides d’exécution.


## 4. Implémentation

### 4.1 Modélisation

- **Problème** : régression — prédire `nb_usagers` (continu) pour un site et une heure donnés.
- **Modèle principal** : LSTM (PyTorch `nn.LSTM`). Entrée : séquence de 24 pas de temps ; à chaque pas : trafic (`nb_usagers`), variables temporelles (heure, mois, jour de la semaine en sin/cos, etc.) et variables météo. Sortie : un scalaire (heure suivante). Architecture : LSTM multicouche (hidden size 64, 2 couches, dropout 0.2) + couche linéaire. Les entrées et la cible sont normalisées (StandardScaler) avant entraînement ; les prédictions sont inversement transformées pour l’affichage.
- **Baseline** : moyenne de `nb_usagers` par (`id_site`, `hour_of_day`, `day_of_week`) sur le jeu chargé ; utilisée si le LSTM est absent ou en secours (historique insuffisant ou erreur).

### 4.2 Modules et API

- **Langage** : Python (3.x).
- **Bibliothèques principales** :  
  - `pandas`, `numpy` : manipulation des données et agrégations.  
  - `requests` : appels à l’API Open-Meteo.  
  - `ijson` : lecture JSON en flux (optionnelle, avec fallback manuel).  
  - `scikit-learn` : `StandardScaler`, utilitaires éventuels.  
  - `torch` : LSTM et entraînement.  
  - `streamlit` : interface web.  
  - `joblib` : sauvegarde/chargement des scalers et (si besoin) d’autres objets.
- **API externes** : Open-Meteo Archive (`https://archive-api.open-meteo.com/v1/archive`) pour les données météo historiques ; Open-Meteo Forecast (`https://api.open-meteo.com/v1/forecast`) pour les prévisions horaires ; Paris Open Data Explore API v2.1 pour récupérer les derniers comptages au moment de la prédiction.

### 4.3 Structure des fichiers et ordre d’exécution

**Arborescence du projet :**

- **Racine** : `app.py` (interface Streamlit), `requirements.txt`, `comptage_2024_2026.json` (données brutes Paris Open Data, à télécharger).
- **src/** :  
  - `preprocess_traffic_json.py` : JSON → CSV trafic nettoyé.  
  - `process_traffic_data.py` : CSV trafic → série temporelle par site/heure (ou `load_traffic.py`, même rôle).  
  - `fetch_weather.py` : API Open-Meteo → CSV météo Paris.  
  - `build_dataset.py` : fusion trafic + météo + variables temporelles.  
  - `train_lstm.py` : entraînement LSTM et sauvegarde du modèle.
- **data/** : tous les fichiers générés (CSV intermédiaires, CSV final, `model_lstm.pt`, scalers, `model_lstm_meta.json`).

**Entrées / sorties par étape :**

| Étape | Script | Entrée | Sortie |
|-------|--------|--------|--------|
| 0 | — | — | Environnement : `pip install -r requirements.txt` |
| 1 | `preprocess_traffic_json.py` | `comptage_2024_2026.json` | `data/traffic_raw_cleaned.csv` |
| 2 | `process_traffic_data.py` | `data/traffic_raw_cleaned.csv` | `data/traffic_timeseries.csv` |
| 3 | `fetch_weather.py` | API (dates en arguments) | `data/weather_paris.csv` |
| 4 | `build_dataset.py` | `data/traffic_timeseries.csv`, `data/weather_paris.csv` | `data/dataset_traffic_weather.csv` |
| 5 | `train_lstm.py` | `data/dataset_traffic_weather.csv` | `data/model_lstm.pt`, `data/scaler_lstm_x.joblib`, `data/scaler_lstm_y.joblib`, `data/model_lstm_meta.json` |
| 5 bis | `train_hgb.py` | `data/dataset_traffic_weather.csv` | `data/model_hgb.joblib`, `data/model_hgb_meta.json`, `data/predictions_hgb.csv` |
| 6 | `app.py` (Streamlit) | `data/dataset_traffic_weather.csv` + optionnellement les fichiers LSTM et HGB | Interface web (prédiction + graphique) |

**Commandes pour exécuter le pipeline (depuis la racine du projet, environnement virtuel activé) :**

1. `pip install -r requirements.txt`
2. `python src/preprocess_traffic_json.py --input comptage_2024_2026.json --output data/traffic_raw_cleaned.csv`
3. `python src/process_traffic_data.py --input data/traffic_raw_cleaned.csv --output data/traffic_timeseries.csv`
4. `python src/fetch_weather.py --start 2024-01-01 --end 2026-02-28 --output data/weather_paris.csv`
5. `python src/build_dataset.py --traffic data/traffic_timeseries.csv --weather data/weather_paris.csv --output data/dataset_traffic_weather.csv`
6. `python src/train_lstm.py`
7. `python src/train_hgb.py`
8. `streamlit run app.py`

L’interface s’ouvre dans le navigateur ; on choisit le site, la date et l’heure, puis le mode de prédiction. Le mode **HGB en ligne** combine les derniers comptages disponibles et la météo forecast. Le mode **LSTM hors ligne** utilise le jeu local si les fichiers du modèle sont présents ; sinon l’application retombe sur la baseline.


## 5. Résultats et discussion

### 5.1 Fichiers output

- **Données** :  
  - `data/traffic_raw_cleaned.csv` : trafic prétraité.  
  - `data/traffic_timeseries.csv` : agrégation (id_site, hour, nb_usagers).  
  - `data/weather_paris.csv` : météo Paris sur la plage demandée.  
  - `data/dataset_traffic_weather.csv` : jeu fusionné (trafic + météo + variables temporelles), utilisé pour l’entraînement et l’app.
- **Modèle** :  
  - `data/model_lstm.pt` : poids du LSTM.  
  - `data/scaler_lstm_x.joblib`, `data/scaler_lstm_y.joblib` : normalisation entrées/sortie.  
  - `data/model_lstm_meta.json` : noms des features, `seq_len`, hyperparamètres, et métriques (ex. MAE, RMSE, R², MAPE, accuracy within 10 % / 20 %).

Exemple de métriques enregistrées (à titre indicatif, dépendantes des données et du split) : R² ≈ 0.87, MAPE ≈ 17.5 %, accuracy within 10 % ≈ 54 %, accuracy within 20 % ≈ 79 %.

### 5.2 Visualisations

- **Streamlit** :  
  - Métrique principale : prédiction de flux (nb_usagers) pour le site, la date et l’heure choisis.  
  - Indication du modèle utilisé (LSTM ou baseline).  
  - Graphique en ligne : historique récent du flux pour le site sélectionné (dernières 168 heures par défaut), via `st.line_chart`.
Les sorties sont des fichiers CSV/JSON/joblib et l’écran Streamlit (prédiction + courbe d’historique). Aucune capture d’écran de code dans la documentation.

### 5.3 Discussion

- **Ce qui a été fait** : pipeline complet (données → fusion → entraînement LSTM/HGB → interface), reproductible via des commandes documentées ; métriques de régression et indicateurs de type “accuracy within X %” ; gestion des cas sans LSTM ou sans assez d’historique (baseline). Une voie de prédiction en ligne est disponible avec HGB lorsque les API externes répondent correctement.
- **Limites et pistes d’amélioration** :  
  - Météo unique pour toute l’agglomération (un point lat/lon) ; une approche multi-sites ou grille pourrait affiner.  
  - Pas de prédiction multi-horizon (seulement l’heure suivante).  
  - Pas de comparaison formelle avec d’autres modèles (ex. modèles plus simples ou autres architectures) dans cette documentation.  
  - Les métriques rapportées sont sur un split validation/test fixe ; une évaluation temporelle (train sur passé, test sur futur) serait plus réaliste pour un déploiement.
- **Souhaits vs réalisations** : l’objectif était de livrer un système opérationnel (données, LSTM, interface) avec une documentation technique claire ; c’est atteint. Les extensions possibles (multi-horizon, comparaison de modèles, météo spatialisée) restent des évolutions futures.
