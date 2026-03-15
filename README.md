# [NN2026] Projet final
## Prédiction du trafic urbain à Paris à partir des données météo et de réseaux de neurones

Ce projet fait partie du M2 TAL de l'Université Paris Nanterre, au second semestre, et correspond au projet final du cours **Réseaux de neurones** enseigné par **Loïc Grobol**.

### Membres du groupe
- CAO Yue
- XU Hongying

**Établissement : INALCO**

---

## 1. Description du projet

Ce projet s’appuie principalement sur les données de **comptage multimodal** de **Paris Open Data** afin d’extraire les flux de circulation sous forme de séries temporelles pour plusieurs sites de comptage à Paris. Nous utilisons également l’API **Open-Meteo** pour récupérer les variables météorologiques correspondant à la même zone géographique et à la même période. Les données de trafic et de météo sont ensuite alignées temporellement afin de construire un jeu de données exploitable pour la modélisation.

Afin de mieux représenter les habitudes de déplacement, nous ajoutons aussi des **variables temporelles et saisonnières**. L’objectif est de permettre au modèle d’apprendre à la fois les régularités du trafic et l’influence des conditions météorologiques.

Le projet propose une application de prédiction du trafic urbain à Paris à partir de ces données. Trois approches sont disponibles :
- une **baseline historique** ;
- un modèle tabulaire **HistGradientBoosting (HGB)** ;
- deux modèles neuronaux récurrents : **LSTM** et **GRU**.

L’interface finale est fournie sous forme d’application **Streamlit**.

---

## 2. Objectif

L’objectif principal est de construire une application capable de **prédire le trafic urbain à court terme** à partir :
- du trafic récent observé ;
- des variables météorologiques ;
- des variables calendaires.

Le projet répond ainsi à un problème de **prévision de séries temporelles multivariées** appliqué à des données urbaines réelles.

---

## 3. Données utilisées

### 3.1 Données de trafic
Source : **Paris Open Data**  
Jeu de données : comptages multimodaux / trajectoires liées aux sites de comptage.

Ces données permettent de récupérer, pour différents sites parisiens, le nombre d’usagers observés selon le temps, le site, le mode de déplacement, le sens et parfois la trajectoire.

### 3.2 Données météo
Source : **Open-Meteo API**

Les variables météo sont récupérées sur la même période que les données de trafic afin de produire un jeu de données fusionné utilisable pour l’apprentissage.

### 3.3 Variables construites
Nous ajoutons plusieurs variables dérivées pour enrichir les modèles :
- heure ;
- jour de la semaine ;
- mois ;
- encodages cycliques (`sin` / `cos`) ;
- autres variables temporelles utiles selon le script.

---

## 4. Structure du projet

nn2026-Prediction_trafic_urbain/
├── data/
│   ├── comptage_2024_2026.json
│   ├── traffic_raw_cleaned.csv
│   ├── traffic_timeseries.csv
│   ├── weather_paris.csv
│   ├── dataset_traffic_weather.csv
│   ├── model_lstm.pt
│   ├── model_lstm_meta.json
│   ├── model_gru.pt
│   ├── model_gru_meta.json
│   ├── model_hgb.joblib
│   ├── model_hgb_meta.json
│   ├── scaler_lstm_x.joblib
│   ├── scaler_lstm_y.joblib
│   ├── scaler_gru_x.joblib
│   ├── scaler_gru_y.joblib
│   ├── predictions_hgb.csv
│   └── predictions_gru.csv
├── src/
│   ├── preprocess_traffic_json.py
│   ├── process_traffic_data.py
│   ├── fetch_weather.py
│   ├── build_dataset.py
│   ├── train_lstm.py
│   ├── train_hgb.py
│   └── train_gru.py
├── DOCUMENTATION.md
├── README.md
├── app.py
└── requirements.txt

### Contenu des dossiers

- data/ : données d’entrée, fichiers intermédiaires, modèles entraînés, scalers et fichiers de sortie

- src/ : scripts Python de prétraitement, fusion des données et entraînement des modèles

- app.py : point d’entrée principal de l’application Streamlit

- DOCUMENTATION.md : documentation technique du projet

- README.md : guide d’installation et d’exécution

- requirements.txt : dépendances Python nécessaires

---

## 5. Description des scripts

### `preprocess_traffic_json.py`
Transforme les données brutes JSON en un format tabulaire plus facilement exploitable.

### `process_traffic_data.py`
Nettoie, filtre et agrège les données de trafic afin de construire des séries temporelles par site et par heure.

### `fetch_weather.py`
Télécharge les données météo nécessaires via l’API Open-Meteo.

### `build_dataset.py`
Fusionne les données de trafic et de météo pour produire le dataset final de modélisation.

### `train_hgb.py`
Entraîne un modèle tabulaire de type HistGradientBoosting et sauvegarde le modèle ainsi que ses métadonnées.

### `train_lstm.py`
Entraîne un modèle neuronal récurrent LSTM et sauvegarde les poids, les scalers et les métadonnées nécessaires à l’inférence.

### `train_gru.py`
Entraîne un modèle neuronal récurrent GRU et sauvegarde les poids, les scalers et les métadonnées nécessaires à l’inférence.

### `app.py`
Application Streamlit permettant de charger les modèles sauvegardés et d’obtenir des prédictions sans relancer l’entraînement.

---

## 6. Installation

Il faut dans un environnement virtuel puis installer les dépendances :

```bash
uv pip install -r requirements.txt
```

---

## 7. Exécution de l’application

Depuis le dossier du projet :

```bash
streamlit run app.py
```

L’application ouvre une interface locale dans le navigateur. Elle permet notamment :
- de charger les modèles déjà entraînés ;
- de sélectionner un site de comptage ;
- de choisir un modèle ;
- d’obtenir une prédiction à partir du trafic récent et de la météo.

---

## 8. Important pour l’évaluation

### Option recommandée
Pour simplifier la correction, il n’est **pas nécessaire de relancer l’entraînement**. Les modèles déjà entraînés peuvent être fournis directement dans le dossier `data/`.

### Ce que fait `app.py`
`app.py` :
1. charge les modèles sauvegardés ;
2. récupère le trafic récent ;
3. récupère la météo ;
4. reconstruit les entrées nécessaires ;
5. affiche la prédiction.

Autrement dit, l’application est pensée pour être **utilisable directement** avec les fichiers de modèles déjà produits.

---

## 9. Reproduction complète du pipeline (facultatif)

Si l’on souhaite reconstruire tout le pipeline à partir des données brutes :

### 9.1 Prétraitement du JSON
```bash
python src/preprocess_traffic_json.py --input data/comptage_2024_2026.json --output data/traffic_raw_cleaned.csv
```

### 9.2 Agrégation du trafic
```bash
python src/process_traffic_data.py --input data/traffic_raw_cleaned.csv --output data/traffic_timeseries.csv
```

### 9.3 Téléchargement de la météo
```bash
python src/fetch_weather.py --start 2024-01-01 --end 2026-02-28 --output data/weather_paris.csv
```

### 9.4 Fusion trafic + météo
```bash
python src/build_dataset.py --traffic data/traffic_timeseries.csv --weather data/weather_paris.csv --output data/dataset_traffic_weather.csv
```

### 9.5 Entraînement des modèles
```bash
python src/train_lstm.py
python src/train_hgb.py
python src/train_gru.py
```

---

## 10. Remarques techniques

- Les scripts `train_lstm.py` et `train_gru.py` doivent rester dans le projet, car `app.py` réimporte les classes des modèles lors du chargement des poids `.pt`.
- Les chemins utilisés supposent que les fichiers de données et les modèles sont placés dans le dossier `data/`.
- Si l’API réseau n’est pas accessible, certaines fonctionnalités en ligne de l’application peuvent être limitées.

---

## 11. Fichiers volumineux

Certains fichiers du dossier `data/` sont volumineux et sont suivis avec **Git LFS**.

Pour récupérer correctement tous les fichiers du projet, il faut :

- cloner le dépôt ;

- avoir Git LFS installé ;

- éviter le téléchargement direct du dépôt en ZIP.

Commandes recommandées :

```bash
git lfs install
git clone <URL_DU_DEPOT>
cd nn2026-Prediction_trafic_urbain
git lfs pull
```
---

## 13. Fichiers fournis

- `DOCUMENTATION.md` : documentation technique du projet ;
- `requirements.txt` : dépendances Python ;
- scripts commentés en français : version lisible pour la correction et la maintenance ;
- modèles déjà entraînés : pour exécuter directement l’application sans réentraînement.
