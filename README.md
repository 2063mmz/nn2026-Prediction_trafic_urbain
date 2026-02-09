# [NN2026] Projet final
## Prédiction du trafic urbain à Paris à partir des données météo et de réseaux de neurone
Ce projet fait partie du M2 TAL de l'Université Paris Nanterre, au second semestre, et correspond au projet final du cours **Réseaux de neurones** enseigné par Loïc Grobol.

#### Membres du groupe :
- CAO Yue 
- XU Hongying

**Établissement: l'INALCO**
  
### Description du projet

Ce projet s’appuie principalement sur les données [les trajectoires liées aux sites de comptages multimodaux](https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/table/disjunctive.mode&disjunctive.trajectoire&disjunctive.voie&disjunctive.sens&disjunctive.label&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJjb2x1bW4iLCJmdW5jIjoiU1VNIiwieUF4aXMiOiJuYl91c2FnZXJzIiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWUsImNvbG9yIjoicmFuZ2UtY3VzdG9tIn1dLCJ4QXhpcyI6InQiLCJtYXhwb2ludHMiOiIiLCJ0aW1lc2NhbGUiOiJob3VyIiwic29ydCI6IiIsInNlcmllc0JyZWFrZG93biI6Im1vZGUiLCJzdGFja2VkIjoibm9ybWFsIiwiY29uZmlnIjp7ImRhdGFzZXQiOiJjb21wdGFnZS1tdWx0aW1vZGFsLWNvbXB0YWdlcyIsIm9wdGlvbnMiOnsiZGlzanVuY3RpdmUubW9kZSI6dHJ1ZSwiZGlzanVuY3RpdmUudHJhamVjdG9pcmUiOnRydWUsImRpc2p1bmN0aXZlLnZvaWUiOnRydWUsImRpc2p1bmN0aXZlLnNlbnMiOnRydWUsImRpc2p1bmN0aXZlLmxhYmVsIjp0cnVlfX19XSwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D&basemap=jawg.dark&location=12,48.86047,2.32427) de [Paris Open Data](https://opendata.paris.fr). Nous sélectionnerons plusieurs stations de comptage à Paris afin d’extraire les flux de véhicules sous forme de séries temporelles. Ensuite, nous utiliserons l’API d’[Open-Meteo](https://github.com/open-meteo/open-meteo) pour obtenir les variables météorologiques correspondant à la même zone géographique et à la même période. Les données de trafic et de météo seront ensuite alignées selon le temps et la localisation pour former un jeu de données utilisable pour la modélisation. Afin de mieux représenter les habitudes de déplacement, nous ajouterons aussi des variables temporelles et saisonnières, pour permettre au modèle d’apprendre à la fois les régularités et les effets des conditions météorologiques.
