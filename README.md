# 🌞 Solar AI Platform

## Plateforme Intelligente de Prédiction et de Supervision des Centrales Solaires

**Niveau : National - Industriel - ERA 2026 - Startup Ready**

---

## 🎯 Vue d'Ensemble

Solar AI Platform est une solution complète et professionnelle de gestion intelligente des centrales solaires photovoltaïques. Elle combine Intelligence Artificielle, Machine Learning et visualisation interactive pour optimiser la production énergétique et détecter les anomalies en temps réel.

### ✨ Fonctionnalités Principales

- 🤖 **Prédiction IA Avancée** : Random Forest, Gradient Boosting et modèles Ensemble
- 🔍 **Détection Intelligente d'Anomalies** : Isolation Forest avec classification par gravité
- 📊 **Dashboard Interactif** : Interface Streamlit professionnelle et responsive
- 🎯 **IA Explicable** : Analyse d'importance des variables et interprétation des décisions
- 🚨 **Système d'Alertes** : Recommandations techniques automatiques et priorisées
- 📈 **KPI Industriels** : Métriques complètes de performance énergétique
- 🌤️ **Intégration Météo** : Impact climatique sur la production
- 💾 **Export & Rapports** : Génération automatique de rapports détaillés

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Windows 10/11, Linux ou macOS

### Installation des dépendances

```bash
# Cloner ou télécharger le projet
cd Solar-AI-Platform

# Installer les dépendances
pip install -r requirements.txt
```

**Note pour Windows :** Si vous utilisez Python depuis le Microsoft Store ou si vous rencontrez des problèmes de permissions, utilisez :

```bash
pip install -r requirements.txt --user
```

---

## 🚀 Utilisation

### 1️⃣ Génération et Entraînement des Modèles

Exécutez d'abord le module principal pour générer les données, entraîner les modèles IA et détecter les anomalies :

```bash
python solar_ai_platform.py
```

**Ce script effectue automatiquement :**
- ✅ Génération de données simulées réalistes
- ✅ Ingénierie avancée des caractéristiques (40+ features)
- ✅ Entraînement de 3 modèles IA (RF, GB, Ensemble)
- ✅ Détection des anomalies avec Isolation Forest
- ✅ Génération d'alertes intelligentes
- ✅ Calcul des KPI industriels
- ✅ Sauvegarde des modèles et rapports

**Résultat attendu :**
```
================================================================================
✅ PROCESSUS TERMINÉ AVEC SUCCÈS
================================================================================

📁 Les résultats sont disponibles dans : ./outputs
💾 Les modèles sont sauvegardés dans : ./models

🚀 Lancez le dashboard avec : streamlit run app.py
================================================================================
```

### 2️⃣ Lancement du Dashboard

Une fois les modèles entraînés, lancez l'interface web :

```bash
streamlit run app.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse : `http://localhost:8501`

---

## 📁 Structure du Projet

```
Solar-AI-Platform/
│
├── data/                           # Données
│   └── merged_cleaned_data.csv    # Données brutes/simulées
│
├── models/                         # Modèles IA sauvegardés
│   ├── solar_predict_model.pkl    # Modèle principal
│   └── all_models.pkl             # Tous les modèles
│
├── outputs/                        # Résultats et rapports
│   ├── reports/                   # Rapports détaillés
│   ├── anomalies_report.csv       # Rapport d'anomalies
│   ├── alerts_active.csv          # Alertes actives
│   └── kpi_report.csv             # KPI industriels
│
├── solar_ai_platform.py           # Module principal (backend)
├── app.py                         # Dashboard Streamlit (frontend)
├── requirements.txt               # Dépendances Python
└── README.md                      # Documentation (ce fichier)
```

---

## 🎨 Pages du Dashboard

### 🏠 Vue Générale Nationale
- KPI Cards en temps réel (Production, Puissance, Onduleurs, Capacité)
- Graphique de production énergétique (7 derniers jours)
- Performance comparative des onduleurs
- Profil de production journalier avec plages de variation

### 🤖 Prédiction de Production IA
- Comparaison des performances des 3 modèles (MAE, RMSE, R²)
- Visualisation de l'importance des variables (Explainable AI)
- Simulateur de prédiction interactif
- Jauge de confiance des prédictions

### 🔍 Détection & Alertes
- Statistiques d'anomalies en temps réel
- Répartition par gravité (Élevée, Moyenne, Faible)
- Types d'anomalies détectés
- Alertes prioritaires avec recommandations techniques
- Timeline des anomalies
- Filtres dynamiques (gravité, onduleurs)

### 🔋 Performance des Onduleurs
- Podium des 3 meilleurs onduleurs (🥇🥈🥉)
- Tableau détaillé de performance
- Graphiques comparatifs de production
- Analyse d'efficacité de conversion
- Classement en temps réel

### 🌤️ Impact Climatique
- Corrélation Irradiance - Production (scatter plot interactif)
- Impact de la température sur le rendement
- Prévisions de production (J+1, J+2, J+3)
- Conditions météorologiques simulées

### 📄 Rapports & Export
- Sélection de période d'analyse
- Résumé des KPI
- Export CSV des données filtrées
- Aperçu des données brutes
- Statistiques descriptives complètes

---

## 🤖 Modèles d'Intelligence Artificielle

### Random Forest Regressor
- **N° d'arbres :** 100
- **Profondeur max :** 20
- **Utilisation :** Prédiction robuste de la production AC Power
- **Avantage :** Excellente gestion des données complexes et non-linéaires

### Gradient Boosting Regressor
- **N° d'estimateurs :** 100
- **Learning rate :** 0.1
- **Profondeur max :** 5
- **Utilisation :** Prédiction optimisée par boosting
- **Avantage :** Réduction progressive de l'erreur

### Modèle Ensemble
- **Composition :** Moyenne pondérée RF + GB
- **Utilisation :** Prédiction la plus stable
- **Avantage :** Combine les forces des deux modèles

### Isolation Forest (Détection d'Anomalies)
- **Contamination :** 5%
- **Utilisation :** Détection automatique des comportements anormaux
- **Avantage :** Fonctionne sans données d'entraînement étiquetées

---

## 📊 KPI et Métriques

### Métriques de Production
- Production totale (MWh)
- Puissance moyenne (kW)
- Puissance maximale (kW)
- Heures de production effectives

### Indicateurs de Performance
- Facteur de capacité (%)
- Rendement énergétique moyen (%)
- Efficacité de conversion DC/AC (%)
- Energy Yield (kWh/kW)

### Qualité et Disponibilité
- Taux de disponibilité (%)
- Taux d'anomalies (%)
- Pertes énergétiques estimées (kWh)
- Nombre d'alertes actives

### Performance IA
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient de détermination)
- Confiance des prédictions

---

## 🔧 Features Engineering (40+ Variables)

### Features Temporelles
- Heure, Jour, Mois, Jour de la semaine, Trimestre
- Saison (Hiver, Printemps, Été, Automne)
- Indicateurs Jour/Nuit, Heures de pointe

### Features Cycliques
- Hour_Sin, Hour_Cos (cycle journalier)
- Month_Sin, Month_Cos (cycle annuel)

### Ratios de Performance
- DC/AC Ratio
- Conversion Efficiency
- Power per Irradiance

### Indicateurs Thermiques
- Température Difference (Module - Ambient)
- Thermal Stress Index
- Temperature Efficiency Loss

### Features Dynamiques
- Rolling Mean (1h) pour AC Power, DC Power, Irradiance, Temperature
- Rolling Std (1h) pour détection de variabilité

---

## 🔍 Détection d'Anomalies

### Types d'Anomalies Détectés
1. **Défaut de conversion DC/AC** : Efficacité < 80%
2. **Surchauffe module** : Température > 80°C
3. **Dysfonctionnement onduleur** : Ratio DC/AC > 1.3
4. **Anomalie générale** : Comportement inhabituel détecté

### Classification par Gravité
- **🔴 Élevée** : Efficacité < 70% OU Temp > 40°C OU Score > 0.7
- **🟠 Moyenne** : Efficacité < 85% OU Temp > 30°C OU Score > 0.5
- **🟢 Faible** : Autres anomalies détectées

### Recommandations Automatiques
Chaque anomalie génère une recommandation technique spécifique :
- Vérification urgente de l'onduleur
- Contrôle de la ventilation
- Diagnostic complet requis
- Inspection recommandée

---

## 📈 Cas d'Usage

### Pour les Opérateurs de Centrales
- Supervision en temps réel de la production
- Détection précoce des pannes
- Optimisation de la maintenance préventive
- Analyse de performance par onduleur

### Pour les Ingénieurs
- Analyse approfondie des données
- Compréhension des facteurs d'influence
- Validation de l'impact climatique
- Amélioration continue du système

### Pour les Décideurs
- KPI synthétiques et visuels
- Rapports automatisés
- Prévisions de production
- Estimation des pertes évitées

### Pour les Chercheurs
- Données simulées réalistes
- Modèles IA pré-entraînés
- Méthodologie complète d'analyse
- Base pour développements futurs

---

## 🛠️ Personnalisation et Extension

### Utilisation de vos propres données

Remplacez le fichier `data/merged_cleaned_data.csv` par vos données réelles. Format attendu :

```csv
Timestamp,Inverter_ID,DC_Voltage,DC_Current,DC_Power,AC_Power,Ambient_Temperature,Module_Temperature,Irradiance
2026-01-01 00:00:00,INV_001,600.5,0.0,0.0,0.0,15.2,15.0,0.0
2026-01-01 00:15:00,INV_001,600.8,0.0,0.0,0.0,15.1,14.9,0.0
...
```

### Ajout de nouveaux modèles IA

Dans `solar_ai_platform.py`, section `train_models()` :

```python
# Exemple : XGBoost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

self.models['XGBoost'] = xgb_model
# Ajoutez les métriques correspondantes
```

### Personnalisation du Dashboard

Modifiez `app.py` pour :
- Ajouter de nouvelles pages
- Changer les couleurs et styles (CSS personnalisé)
- Intégrer d'autres visualisations
- Ajouter des fonctionnalités d'export

---

## 🐛 Dépannage

### Erreur : "ModuleNotFoundError"
```bash
# Solution : Réinstallez les dépendances
pip install -r requirements.txt --upgrade
```

### Erreur : "FileNotFoundError: merged_cleaned_data.csv"
```bash
# Solution : Exécutez d'abord le module principal
python solar_ai_platform.py
```

### Dashboard ne se lance pas
```bash
# Vérifiez l'installation de Streamlit
streamlit --version

# Réinstallez si nécessaire
pip install streamlit --upgrade
```

### Performances lentes
```bash
# Réduisez la taille des données simulées
# Dans solar_ai_platform.py, modifiez :
platform.generate_sample_data(n_inverters=3, days=15)  # Au lieu de 5 et 30
```

---

## 📚 Technologies Utilisées

- **Python 3.8+** : Langage de programmation
- **NumPy** : Calcul numérique
- **Pandas** : Manipulation de données
- **Scikit-learn** : Machine Learning
- **Matplotlib / Seaborn** : Visualisation statique
- **Plotly** : Visualisation interactive
- **Streamlit** : Framework web pour dashboard
- **Pickle** : Sérialisation des modèles

---

## 🎓 Niveau Académique et Professionnel

Ce projet est conçu pour répondre aux exigences suivantes :

### ✅ Niveau Universitaire
- Rigueur scientifique et méthodologique
- Documentation complète
- Code commenté et structuré
- Analyse statistique approfondie

### ✅ Niveau Industriel
- Architecture professionnelle
- Gestion d'erreurs robuste
- Logging détaillé
- Compatibilité multi-plateforme

### ✅ Niveau Startup / Incubation
- Interface utilisateur soignée
- Présentation claire des résultats
- KPI orientés business
- Scalabilité du code

### ✅ Niveau National (ERA 2026)
- Solution complète et autonome
- Prête pour déploiement
- Documentation technique exhaustive
- Vision stratégique énergétique

---

## 🏆 Qualité et Standards

- ✅ Code PEP 8 compliant
- ✅ Gestion des erreurs et exceptions
- ✅ Logging professionnel
- ✅ Chemins relatifs (compatibilité Windows/Linux)
- ✅ Aucune dépendance système externe
- ✅ Installation facile (pip install)
- ✅ Documentation exhaustive
- ✅ Interface intuitive

---

## 📝 Licence et Utilisation

Ce projet est développé à des fins éducatives, académiques et de démonstration pour ERA 2026.

**Utilisation autorisée pour :**
- Projets académiques et universitaires
- Présentations et démonstrations
- Développement de prototypes
- Formation et apprentissage

**Utilisation commerciale :**
Contactez les auteurs pour les conditions d'utilisation commerciale.

---

## 👥 Équipe et Support

**Chalabi Mohammed El Amine**
Développé pour l'Excellence et l'Innovation Énergétique

### Contact
- 📧 Email : [mohac6442@gmail.com]
- 🌐 Site web : [En développement]
- 📱 Support : [github.com/Moham440]

---

## 🚀 Roadmap Future

### Version 2.0 (Planifiée)
- [ ] Intégration API météo réelle (OpenWeatherMap)
- [ ] Prévisions ML à 7 jours
- [ ] Module de maintenance prédictive
- [ ] Détection avancée par Deep Learning
- [ ] Export PDF automatique des rapports
- [ ] Notifications email/SMS
- [ ] Multi-langue (FR/EN/AR)
- [ ] Mode offline complet

### Version 3.0 (Vision)
- [ ] Déploiement cloud (AWS/Azure)
- [ ] Application mobile (iOS/Android)
- [ ] Base de données temps réel
- [ ] API REST complète
- [ ] Tableau de bord national multi-sites
- [ ] Intégration blockchain pour certification
- [ ] IA conversationnelle (chatbot)

---

## 🙏 Remerciements

Merci à tous ceux qui contribuent à l'avancement de l'énergie solaire en Algérie et dans le monde.

**Technologies open-source utilisées :**
- Scikit-learn Team
- Streamlit Team
- Plotly Team
- Pandas Development Team
- Python Software Foundation

---

## ⭐ Star et Contribuer

Si ce projet vous est utile, n'hésitez pas à :
- ⭐ Mettre une étoile sur GitHub : Moham440
- 🐛 Signaler des bugs via Issues
- 💡 Proposer des améliorations
- 🤝 Contribuer au code

---

<div align="center">

**🌞 Solar AI Platform - Powered by Chalabi Mohammed El Amine 🌞**

*L'avenir de l'énergie solaire commence aujourd'hui*

---

**Développé avec ❤️ pour ERA 2026 et l'Innovation Énergétique Nationale**

</div>
