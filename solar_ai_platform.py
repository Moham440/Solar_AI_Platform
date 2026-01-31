"""
Solar AI Platform - Plateforme Intelligente de Prédiction et Supervision des Centrales Solaires
Niveau : National - Industriel - ERA 2026 - Startup Ready

Auteur : Chalabi Mohammed El Amine
Version : 1.0.0
Date : Janvier 2026
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration des logs (sans emojis pour compatibilité Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solar_ai_platform.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SolarAIPlatform:
    """
    Plateforme complète de gestion intelligente des centrales solaires
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialisation de la plateforme Solar AI

        Args:
            data_path: Chemin vers le fichier de données CSV
        """
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"

        # Création des dossiers
        for directory in [self.data_dir, self.models_dir, self.outputs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Variables de stockage
        self.df_raw = None
        self.df_processed = None
        self.df_train = None
        self.df_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.feature_importance = {}

        # Métriques
        self.metrics = {}
        self.predictions = {}
        self.anomalies = None

        logger.info("[OK] Solar AI Platform initialisée avec succès")
        logger.info(f"📁 Dossier de travail : {self.base_dir}")

        if data_path:
            self.load_data(data_path)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Chargement et vérification des données

        Args:
            filepath: Chemin vers le fichier CSV

        Returns:
            DataFrame chargé
        """
        try:
            logger.info(f"📂 Chargement des données depuis : {filepath}")

            # Lecture du fichier
            self.df_raw = pd.read_csv(filepath)
            logger.info(f"[OK] {len(self.df_raw)} lignes chargées")

            # Vérifications
            logger.info("🔍 Vérification de la qualité des données...")

            # Valeurs manquantes
            missing = self.df_raw.isnull().sum()
            if missing.any():
                logger.warning(f"⚠️ Valeurs manquantes détectées :\n{missing[missing > 0]}")

            # Doublons
            duplicates = self.df_raw.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"⚠️ {duplicates} doublons détectés")
                self.df_raw = self.df_raw.drop_duplicates()

            # Affichage des colonnes
            logger.info(f"[DATA] Colonnes disponibles : {list(self.df_raw.columns)}")
            logger.info(f"📈 Statistiques de base :\n{self.df_raw.describe()}")

            return self.df_raw

        except FileNotFoundError:
            logger.error(f"❌ Fichier non trouvé : {filepath}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement : {str(e)}")
            raise

    def generate_sample_data(self, n_inverters: int = 5, days: int = 30) -> pd.DataFrame:
        """
        Génération de données simulées réalistes pour démonstration

        Args:
            n_inverters: Nombre d'onduleurs
            days: Nombre de jours de données

        Returns:
            DataFrame avec données simulées
        """
        logger.info(f"[BUILD] Génération de données simulées : {n_inverters} onduleurs, {days} jours")

        np.random.seed(42)

        # Génération de timestamps
        start_date = datetime.now() - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, periods=days*24*4, freq='15min')

        data = []

        for inv_id in range(1, n_inverters + 1):
            # Performance aléatoire par onduleur (certains moins performants)
            efficiency_factor = np.random.uniform(0.85, 1.0)

            for ts in timestamps:
                hour = ts.hour
                month = ts.month

                # Irradiation solaire (W/m²) - profil journalier réaliste
                if 6 <= hour <= 18:
                    base_irradiance = 1000 * np.sin((hour - 6) * np.pi / 12)
                    seasonal_factor = 1.0 + 0.3 * np.sin((month - 1) * np.pi / 6)
                    irradiance = base_irradiance * seasonal_factor * np.random.uniform(0.8, 1.1)
                else:
                    irradiance = 0

                # Température ambiante (°C)
                temp_ambient = 20 + 15 * np.sin((month - 1) * np.pi / 6) + \
                              10 * np.sin((hour - 12) * np.pi / 12) + \
                              np.random.normal(0, 2)

                # Température du module (plus élevée que l'ambiante)
                temp_module = temp_ambient + irradiance * 0.025 + np.random.normal(0, 3)

                # Tension et courant DC
                dc_voltage = 600 + np.random.normal(0, 20)
                dc_current = irradiance * 0.01 * efficiency_factor + np.random.normal(0, 0.5)
                dc_power = dc_voltage * dc_current

                # Puissance AC (avec pertes de conversion)
                conversion_efficiency = 0.95 + np.random.normal(0, 0.02)
                ac_power = dc_power * conversion_efficiency

                # Injection d'anomalies aléatoires (5% des cas)
                if np.random.random() < 0.05:
                    ac_power *= np.random.uniform(0.5, 0.8)  # Baisse de performance

                data.append({
                    'Timestamp': ts,
                    'Inverter_ID': f'INV_{inv_id:03d}',
                    'DC_Voltage': max(0, dc_voltage),
                    'DC_Current': max(0, dc_current),
                    'DC_Power': max(0, dc_power),
                    'AC_Power': max(0, ac_power),
                    'Ambient_Temperature': temp_ambient,
                    'Module_Temperature': temp_module,
                    'Irradiance': max(0, irradiance)
                })

        self.df_raw = pd.DataFrame(data)

        # Sauvegarde
        output_path = self.data_dir / "merged_cleaned_data.csv"
        self.df_raw.to_csv(output_path, index=False)
        logger.info(f"[OK] Données simulées générées : {len(self.df_raw)} lignes")
        logger.info(f"💾 Sauvegardé dans : {output_path}")

        return self.df_raw

    def feature_engineering(self) -> pd.DataFrame:
        """
        Ingénierie avancée des caractéristiques

        Returns:
            DataFrame avec features enrichies
        """
        logger.info("[BUILD] Démarrage de l'ingénierie des caractéristiques...")

        df = self.df_raw.copy()

        # Conversion du timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # === FEATURES TEMPORELLES ===
        logger.info("⏰ Création des features temporelles...")
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['Quarter'] = df['Timestamp'].dt.quarter

        # Saisons
        df['Season'] = df['Month'].apply(lambda x:
            1 if x in [12, 1, 2] else
            2 if x in [3, 4, 5] else
            3 if x in [6, 7, 8] else 4
        )

        # === FEATURES CYCLIQUES ===
        logger.info("🔄 Création des features cycliques...")
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        # === RATIOS ET EFFICACITÉ ===
        logger.info("⚡ Calcul des ratios de performance...")
        df['DC_AC_Ratio'] = df['DC_Power'] / (df['AC_Power'] + 1e-6)
        df['Conversion_Efficiency'] = df['AC_Power'] / (df['DC_Power'] + 1e-6)
        df['Power_Per_Irradiance'] = df['AC_Power'] / (df['Irradiance'] + 1e-6)

        # === STRESS THERMIQUE ===
        logger.info("🌡️ Calcul des indicateurs thermiques...")
        df['Temp_Difference'] = df['Module_Temperature'] - df['Ambient_Temperature']
        df['Thermal_Stress'] = df['Module_Temperature'] * df['Irradiance'] / 1000
        df['Temperature_Efficiency_Loss'] = 0.005 * (df['Module_Temperature'] - 25)

        # === INDICATEURS DE RENDEMENT ===
        logger.info("[DATA] Calcul des indicateurs de rendement...")
        df['Power_Density'] = df['AC_Power'] / (df['DC_Voltage'] * df['DC_Current'] + 1e-6)
        df['Voltage_Current_Product'] = df['DC_Voltage'] * df['DC_Current']

        # === FEATURES DYNAMIQUES ===
        logger.info("📈 Création des features dynamiques...")
        for col in ['AC_Power', 'DC_Power', 'Irradiance', 'Module_Temperature']:
            df[f'{col}_Rolling_Mean_1h'] = df.groupby('Inverter_ID')[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean()
            )
            df[f'{col}_Rolling_Std_1h'] = df.groupby('Inverter_ID')[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).std()
            )

        # === INDICATEURS JOUR/NUIT ===
        df['Is_Daytime'] = ((df['Hour'] >= 6) & (df['Hour'] <= 18)).astype(int)
        df['Is_Peak_Hour'] = ((df['Hour'] >= 10) & (df['Hour'] <= 14)).astype(int)

        # Remplissage des NaN
        df = df.fillna(0)

        self.df_processed = df

        logger.info(f"[OK] Feature Engineering terminé : {len(df.columns)} features créées")
        logger.info(f"📋 Nouvelles features : {[col for col in df.columns if col not in self.df_raw.columns]}")

        return self.df_processed

    def train_models(self, target_column: str = 'AC_Power') -> Dict:
        """
        Entraînement des modèles IA

        Args:
            target_column: Variable cible à prédire

        Returns:
            Dictionnaire des métriques
        """
        logger.info(f"[AI] Démarrage de l'entraînement des modèles IA...")
        logger.info(f"🎯 Variable cible : {target_column}")

        # Préparation des données
        df = self.df_processed.copy()

        # Sélection des features
        exclude_cols = ['Timestamp', 'Inverter_ID', target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_column]

        logger.info(f"[DATA] Features utilisées : {len(feature_cols)}")
        logger.info(f"📈 Taille du dataset : {len(X)} échantillons")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"✂️ Train: {len(X_train)}, Test: {len(X_test)}")

        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # === RANDOM FOREST ===
        logger.info("🌲 Entraînement Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)

        rf_metrics = {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred)
        }

        self.models['RandomForest'] = rf_model
        self.metrics['RandomForest'] = rf_metrics
        self.predictions['RandomForest'] = rf_pred

        logger.info(f"[OK] Random Forest - MAE: {rf_metrics['MAE']:.2f}, R²: {rf_metrics['R2']:.4f}")

        # === GRADIENT BOOSTING ===
        logger.info("🚀 Entraînement Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)

        gb_metrics = {
            'MAE': mean_absolute_error(y_test, gb_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'R2': r2_score(y_test, gb_pred)
        }

        self.models['GradientBoosting'] = gb_model
        self.metrics['GradientBoosting'] = gb_metrics
        self.predictions['GradientBoosting'] = gb_pred

        logger.info(f"[OK] Gradient Boosting - MAE: {gb_metrics['MAE']:.2f}, R²: {gb_metrics['R2']:.4f}")

        # === ENSEMBLE MODEL ===
        logger.info("🎭 Création du modèle Ensemble...")
        ensemble_pred = (rf_pred + gb_pred) / 2

        ensemble_metrics = {
            'MAE': mean_absolute_error(y_test, ensemble_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'R2': r2_score(y_test, ensemble_pred)
        }

        self.metrics['Ensemble'] = ensemble_metrics
        self.predictions['Ensemble'] = ensemble_pred

        logger.info(f"[OK] Ensemble - MAE: {ensemble_metrics['MAE']:.2f}, R²: {ensemble_metrics['R2']:.4f}")

        # Sélection du meilleur modèle
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['R2'])
        logger.info(f"🏆 Meilleur modèle : {best_model_name} (R² = {self.metrics[best_model_name]['R2']:.4f})")

        # Importance des features (Random Forest)
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        self.feature_importance = feature_importance

        logger.info("\n[DATA] Top 10 Features importantes :")
        logger.info(feature_importance.head(10).to_string(index=False))

        # Sauvegarde des modèles
        self.save_models()

        # Stockage pour utilisation ultérieure
        self.df_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
        self.df_train['y_true'] = y_train.values

        self.df_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
        self.df_test['y_true'] = y_test.values
        self.df_test['y_pred_rf'] = rf_pred
        self.df_test['y_pred_gb'] = gb_pred
        self.df_test['y_pred_ensemble'] = ensemble_pred

        return self.metrics

    def detect_anomalies(self) -> pd.DataFrame:
        """
        Détection intelligente des anomalies avec Isolation Forest

        Returns:
            DataFrame avec anomalies détectées
        """
        logger.info("🔍 Démarrage de la détection des anomalies...")

        df = self.df_processed.copy()

        # Sélection des features pour la détection
        anomaly_features = [
            'AC_Power', 'DC_Power', 'Conversion_Efficiency',
            'Module_Temperature', 'Irradiance', 'DC_AC_Ratio',
            'Temp_Difference', 'Thermal_Stress'
        ]

        X_anomaly = df[anomaly_features].fillna(0)

        # Entraînement Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.05,  # 5% d'anomalies attendues
            random_state=42,
            n_jobs=-1
        )

        anomaly_labels = self.anomaly_detector.fit_predict(X_anomaly)
        anomaly_scores = self.anomaly_detector.score_samples(X_anomaly)

        df['Is_Anomaly'] = (anomaly_labels == -1).astype(int)
        df['Anomaly_Score'] = anomaly_scores

        # Classification par gravité
        def classify_severity(row):
            if row['Is_Anomaly'] == 0:
                return 'Normal'

            score = abs(row['Anomaly_Score'])
            efficiency = row['Conversion_Efficiency']
            temp_diff = row['Temp_Difference']

            # Critères de gravité
            if efficiency < 0.7 or temp_diff > 40 or score > 0.7:
                return 'Élevée'
            elif efficiency < 0.85 or temp_diff > 30 or score > 0.5:
                return 'Moyenne'
            else:
                return 'Faible'

        df['Severity'] = df.apply(classify_severity, axis=1)

        # Type d'anomalie
        def classify_anomaly_type(row):
            if row['Is_Anomaly'] == 0:
                return 'Aucune'

            if row['Conversion_Efficiency'] < 0.8:
                return 'Défaut de conversion DC/AC'
            elif row['Module_Temperature'] > 80:
                return 'Surchauffe module'
            elif row['DC_AC_Ratio'] > 1.3:
                return 'Dysfonctionnement onduleur'
            else:
                return 'Anomalie générale'

        df['Anomaly_Type'] = df.apply(classify_anomaly_type, axis=1)

        # Estimation des pertes énergétiques
        normal_power = df[df['Is_Anomaly'] == 0]['AC_Power'].mean()
        df['Energy_Loss_kWh'] = df.apply(
            lambda row: max(0, (normal_power - row['AC_Power']) / 4) if row['Is_Anomaly'] == 1 else 0,
            axis=1
        )

        self.anomalies = df[df['Is_Anomaly'] == 1].copy()

        # Statistiques
        total_anomalies = df['Is_Anomaly'].sum()
        severity_counts = df['Severity'].value_counts()
        total_losses = df['Energy_Loss_kWh'].sum()

        logger.info(f"[OK] Détection terminée : {total_anomalies} anomalies détectées ({total_anomalies/len(df)*100:.2f}%)")
        logger.info(f"\n[DATA] Répartition par gravité :\n{severity_counts}")
        logger.info(f"⚡ Pertes énergétiques estimées : {total_losses:.2f} kWh")

        # Sauvegarde
        anomaly_report = self.anomalies.copy()
        output_path = self.outputs_dir / "anomalies_report.csv"
        anomaly_report.to_csv(output_path, index=False)
        logger.info(f"💾 Rapport d'anomalies sauvegardé : {output_path}")

        return df

    def generate_alerts(self, threshold_severity: str = 'Moyenne') -> pd.DataFrame:
        """
        Génération d'alertes intelligentes avec recommandations

        Args:
            threshold_severity: Niveau de gravité minimum pour les alertes

        Returns:
            DataFrame des alertes
        """
        logger.info(f"🚨 Génération des alertes (gravité >= {threshold_severity})...")

        if self.anomalies is None or len(self.anomalies) == 0:
            logger.warning("⚠️ Aucune anomalie détectée")
            return pd.DataFrame()

        severity_order = {'Faible': 1, 'Moyenne': 2, 'Élevée': 3}
        min_severity = severity_order.get(threshold_severity, 2)

        alerts = self.anomalies[
            self.anomalies['Severity'].map(severity_order) >= min_severity
        ].copy()

        # Recommandations techniques
        def get_recommendation(row):
            anomaly_type = row['Anomaly_Type']
            severity = row['Severity']

            recommendations = {
                'Défaut de conversion DC/AC': f"URGENT - Vérifier l'onduleur {row['Inverter_ID']} - Efficacité critique ({row['Conversion_Efficiency']:.2%})",
                'Surchauffe module': f"ATTENTION - Température excessive ({row['Module_Temperature']:.1f}°C) - Vérifier la ventilation",
                'Dysfonctionnement onduleur': f"INTERVENTION - Ratio DC/AC anormal ({row['DC_AC_Ratio']:.2f}) - Diagnostic requis",
                'Anomalie générale': "Inspection recommandée - Comportement inhabituel détecté"
            }

            return recommendations.get(anomaly_type, "Inspection recommandée")

        alerts['Recommendation'] = alerts.apply(get_recommendation, axis=1)

        # Priorisation
        alerts['Priority'] = alerts['Severity'].map({
            'Faible': 3,
            'Moyenne': 2,
            'Élevée': 1
        })

        alerts = alerts.sort_values('Priority')

        logger.info(f"[OK] {len(alerts)} alertes générées")
        logger.info(f"\n🚨 Alertes prioritaires (Top 5) :")
        top_alerts = alerts.head(5)[['Timestamp', 'Inverter_ID', 'Severity', 'Anomaly_Type', 'Recommendation']]
        logger.info(top_alerts.to_string(index=False))

        # Sauvegarde
        output_path = self.outputs_dir / "alerts_active.csv"
        alerts.to_csv(output_path, index=False)
        logger.info(f"💾 Alertes sauvegardées : {output_path}")

        return alerts

    def calculate_kpis(self) -> Dict:
        """
        Calcul des KPI industriels

        Returns:
            Dictionnaire des KPI
        """
        logger.info("[DATA] Calcul des KPI industriels...")

        df = self.df_processed

        # Vérifier si les colonnes d'anomalies existent
        has_anomalies = 'Is_Anomaly' in df.columns and 'Energy_Loss_kWh' in df.columns

        kpis = {
            # Production
            'total_production_mwh': df['AC_Power'].sum() / 4000,  # 15min intervals -> MWh
            'avg_power_kw': df['AC_Power'].mean(),
            'max_power_kw': df['AC_Power'].max(),

            # Performance
            'capacity_factor': (df['AC_Power'].mean() / df['AC_Power'].max()) * 100,
            'avg_efficiency': df['Conversion_Efficiency'].mean() * 100 if 'Conversion_Efficiency' in df.columns else 0,
            'energy_yield': df['AC_Power'].sum() / (df['Irradiance'].sum() + 1e-6),

            # Disponibilité
            'availability': (1 - df['Is_Anomaly'].mean()) * 100 if has_anomalies else 95.0,
            'uptime_hours': len(df[df['AC_Power'] > 0]) / 4,

            # Pertes
            'total_losses_kwh': df['Energy_Loss_kWh'].sum() if has_anomalies else 0,
            'anomaly_rate': df['Is_Anomaly'].mean() * 100 if has_anomalies else 0,

            # Onduleurs
            'n_inverters': df['Inverter_ID'].nunique(),
            'best_inverter': df.groupby('Inverter_ID')['AC_Power'].mean().idxmax(),
            'worst_inverter': df.groupby('Inverter_ID')['AC_Power'].mean().idxmin(),
        }

        logger.info("\n📈 KPI Principaux :")
        logger.info(f"  Production totale : {kpis['total_production_mwh']:.2f} MWh")
        logger.info(f"  Facteur de capacité : {kpis['capacity_factor']:.1f}%")
        logger.info(f"  Rendement moyen : {kpis['avg_efficiency']:.1f}%")
        logger.info(f"  Disponibilité : {kpis['availability']:.1f}%")
        logger.info(f"  Pertes énergétiques : {kpis['total_losses_kwh']:.2f} kWh")
        logger.info(f"  Meilleur onduleur : {kpis['best_inverter']}")

        # Sauvegarde
        kpi_df = pd.DataFrame([kpis])
        output_path = self.outputs_dir / "kpi_report.csv"
        kpi_df.to_csv(output_path, index=False)
        logger.info(f"💾 Rapport KPI sauvegardé : {output_path}")

        return kpis

    def save_models(self):
        """
        Sauvegarde des modèles entraînés
        """
        logger.info("💾 Sauvegarde des modèles...")

        # Sauvegarde du meilleur modèle (Random Forest)
        model_path = self.models_dir / "solar_predict_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.models.get('RandomForest'),
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'metrics': self.metrics
            }, f)

        logger.info(f"[OK] Modèle principal sauvegardé : {model_path}")

        # Sauvegarde de tous les modèles
        all_models_path = self.models_dir / "all_models.pkl"
        with open(all_models_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'metrics': self.metrics,
                'anomaly_detector': self.anomaly_detector
            }, f)

        logger.info(f"[OK] Tous les modèles sauvegardés : {all_models_path}")

    def load_models(self):
        """
        Chargement des modèles sauvegardés
        """
        model_path = self.models_dir / "all_models.pkl"

        if not model_path.exists():
            logger.warning("⚠️ Aucun modèle sauvegardé trouvé")
            return

        logger.info(f"📂 Chargement des modèles depuis : {model_path}")

        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)

        self.models = saved_data.get('models', {})
        self.scaler = saved_data.get('scaler')
        self.metrics = saved_data.get('metrics', {})
        self.anomaly_detector = saved_data.get('anomaly_detector')

        logger.info("[OK] Modèles chargés avec succès")

    def generate_report(self) -> str:
        """
        Génération d'un rapport complet

        Returns:
            Chemin du rapport généré
        """
        logger.info("📄 Génération du rapport complet...")

        report_lines = [
            "="*80,
            "SOLAR AI PLATFORM - RAPPORT D'ANALYSE COMPLET",
            "="*80,
            f"\nDate de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Période d'analyse : {self.df_processed['Timestamp'].min()} → {self.df_processed['Timestamp'].max()}",
            f"Nombre d'échantillons : {len(self.df_processed):,}",
            "\n" + "="*80,
            "\n[DATA] PERFORMANCE DES MODÈLES IA",
            "-"*80,
        ]

        for model_name, metrics in self.metrics.items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  MAE  : {metrics['MAE']:.2f} kW")
            report_lines.append(f"  RMSE : {metrics['RMSE']:.2f} kW")
            report_lines.append(f"  R²   : {metrics['R2']:.4f}")

        report_lines.extend([
            "\n" + "="*80,
            "\n🔍 DÉTECTION DES ANOMALIES",
            "-"*80,
        ])

        if self.anomalies is not None and len(self.anomalies) > 0:
            report_lines.append(f"Total anomalies détectées : {len(self.anomalies):,}")
            report_lines.append(f"Taux d'anomalies : {len(self.anomalies)/len(self.df_processed)*100:.2f}%")

            # Vérifier si la colonne Severity existe
            if 'Severity' in self.df_processed.columns:
                report_lines.append("\nRépartition par gravité :")
                severity_counts = self.df_processed['Severity'].value_counts()
                for severity, count in severity_counts.items():
                    report_lines.append(f"  {severity}: {count:,} ({count/len(self.df_processed)*100:.2f}%)")
        else:
            report_lines.append("Aucune anomalie détectée")

        report_lines.extend([
            "\n" + "="*80,
            "\n📈 TOP 10 FEATURES IMPORTANTES",
            "-"*80,
        ])

        if not self.feature_importance.empty:
            for idx, row in self.feature_importance.head(10).iterrows():
                report_lines.append(f"  {row['Feature']:40} : {row['Importance']:.4f}")

        report_lines.append("\n" + "="*80 + "\n")

        report_content = "\n".join(report_lines)

        # Sauvegarde
        report_path = self.outputs_dir / f"solar_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"[OK] Rapport généré : {report_path}")
        print(report_content)

        return str(report_path)


def main():
    """
    Fonction principale d'exécution
    """
    print("\n" + "="*80)
    print("🌞 SOLAR AI PLATFORM - Plateforme Intelligente de Supervision")
    print("="*80 + "\n")

    # Initialisation
    platform = SolarAIPlatform()

    # Génération de données de démonstration
    print("[DATA] Génération de données simulées...")
    platform.generate_sample_data(n_inverters=5, days=30)

    # Feature Engineering
    print("\n[BUILD] Ingénierie des caractéristiques...")
    platform.feature_engineering()

    # Entraînement des modèles
    print("\n[AI] Entraînement des modèles IA...")
    platform.train_models()

    # Détection d'anomalies
    print("\n🔍 Détection des anomalies...")
    platform.detect_anomalies()

    # Génération d'alertes
    print("\n🚨 Génération des alertes...")
    platform.generate_alerts(threshold_severity='Moyenne')

    # Calcul des KPI
    print("\n[DATA] Calcul des KPI...")
    platform.calculate_kpis()

    # Rapport final
    print("\n📄 Génération du rapport...")
    platform.generate_report()

    print("\n" + "="*80)
    print("[OK] PROCESSUS TERMINÉ AVEC SUCCÈS")
    print("="*80)
    print(f"\n📁 Les résultats sont disponibles dans : {platform.outputs_dir}")
    print(f"💾 Les modèles sont sauvegardés dans : {platform.models_dir}")
    print("\n🚀 Lancez le dashboard avec : streamlit run app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()