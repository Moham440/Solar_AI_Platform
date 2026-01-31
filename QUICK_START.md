# 🚀 Guide de Démarrage Rapide - Solar AI Platform

## Installation en 3 étapes

### Étape 1 : Installation des dépendances

```bash
pip install -r requirements.txt
```

**Note Windows :** Si vous avez des problèmes de permissions :
```bash
pip install -r requirements.txt --user
```

### Étape 2 : Génération des données et entraînement des modèles

```bash
python solar_ai_platform.py
```

**Durée :** ~30 secondes

**Ce qui est généré automatiquement :**
- ✅ Données simulées réalistes (14,400 échantillons)
- ✅ 40+ features engineering
- ✅ 3 modèles IA entraînés (RF, GB, Ensemble)
- ✅ Détection d'anomalies
- ✅ Alertes intelligentes
- ✅ KPI industriels
- ✅ Rapports complets

### Étape 3 : Lancer le Dashboard

```bash
streamlit run app.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur : **http://localhost:8501**

---

## Structure du Projet

```
Solar-AI-Platform/
├── data/                    # Données (générées automatiquement)
├── models/                  # Modèles IA (sauvegardés automatiquement)
├── outputs/                 # Rapports et résultats
├── solar_ai_platform.py    # Backend (IA, ML, détection)
├── app.py                  # Frontend (Dashboard Streamlit)
├── requirements.txt        # Dépendances Python
└── README.md              # Documentation complète
```

---

## Résultats Attendus

### Performance des Modèles IA

- **Random Forest** : R² = 0.9998, MAE = 7.44 kW
- **Gradient Boosting** : R² = 0.9999, MAE = 7.32 kW
- **Ensemble** : R² = 0.9999, MAE = 6.54 kW ⭐

### Détection d'Anomalies

- 720 anomalies détectées (5.00%)
- Classification par gravité (Élevée, Moyenne, Faible)
- Recommandations techniques automatiques
- Estimation des pertes énergétiques

### KPI Principaux

- Production totale : 5,774 MWh
- Facteur de capacité : 23.5%
- Rendement moyen : 68.1%
- Disponibilité : 95.0%

---

## Pages du Dashboard

1. **🏠 Vue Générale** - KPI en temps réel, production, performances
2. **🤖 Prédiction IA** - Comparaison modèles, importance features, simulateur
3. **🔍 Anomalies & Alertes** - Détection, classification, recommandations
4. **🔋 Performance Onduleurs** - Classement, comparaison, analyses
5. **🌤️ Impact Climatique** - Corrélations, prévisions météo
6. **📄 Rapports & Export** - KPI, exports CSV, statistiques

---

## Dépannage Rapide

### Erreur : ModuleNotFoundError
```bash
pip install -r requirements.txt --upgrade
```

### Le dashboard ne démarre pas
```bash
# Vérifier l'installation
streamlit --version

# Réinstaller si nécessaire
pip install streamlit --upgrade
```

### Données non trouvées
```bash
# Exécuter d'abord le backend
python solar_ai_platform.py
```

---

## Utilisation de vos propres données

Remplacez `data/merged_cleaned_data.csv` par vos données avec le format :

```csv
Timestamp,Inverter_ID,DC_Voltage,DC_Current,DC_Power,AC_Power,Ambient_Temperature,Module_Temperature,Irradiance
2026-01-01 00:00:00,INV_001,600.5,0.0,0.0,0.0,15.2,15.0,0.0
```

Puis relancez :
```bash
python solar_ai_platform.py
streamlit run app.py
```

---

## Support et Contact

**Solar AI Team**

📧 Email : mohac6442@gmail.com
🌐 Documentation : README.md
🐛 Issues : [GitHub: Moham440]

---

## Technologies

- Python 3.8+
- Scikit-learn (ML)
- Streamlit (Dashboard)
- Plotly (Visualisation)
- Pandas & NumPy (Data Science)

---

**🌞 Solar AI Platform - Développé pour ERA 2026 🌞**

*Niveau : National - Industriel - Startup Ready*
