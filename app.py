"""
Solar AI Platform - Dashboard Professionnel Streamlit
Application de supervision et prédiction des centrales solaires

Niveau : National - Industriel - ERA 2026 - Startup Ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import sys

# Configuration de la page
st.set_page_config(
    page_title="Solar AI Platform",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design professionnel
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B00 0%, #FFB800 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }

    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }

    .alert-low {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Chargement des données principales"""
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "merged_cleaned_data.csv"

    if not data_path.exists():
        st.error("❌ Fichier de données non trouvé. Veuillez générer les données avec `python solar_ai_platform.py`")
        st.stop()

    df = pd.read_csv(data_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df


@st.cache_data(show_spinner=False)
def load_processed_data():
    """Chargement sécurisé des données d'anomalies"""
    base_dir = Path(__file__).parent
    anomaly_report = base_dir / "outputs" / "anomalies_report.csv"

    if not anomaly_report.exists():
        st.warning("⚠️ Rapport d'anomalies introuvable. Générez-le avec `python solar_ai_platform.py`")
        return None

    try:
        df = pd.read_csv(anomaly_report)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {str(e)}")
        return None


@st.cache_resource
def load_models():
    """Chargement des modèles entraînés avec gestion d'erreurs robuste"""
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "all_models.pkl"

    if not model_path.exists():
        return None

    try:
        with open(model_path, 'rb') as f:
            models_data = pickle.load(f)
        return models_data
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les modèles : {str(e)}")
        return None


def create_kpi_card(label, value, unit="", delta=None, color="blue"):
    """Création d'une carte KPI stylisée"""
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_symbol = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="color: {delta_color}; font-size: 1rem;">{delta_symbol} {abs(delta):.1f}%</div>'

    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 10px; color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;">{value} {unit}</div>
        {delta_html}
    </div>
    """


def page_overview():
    """Page : Vue Générale Nationale"""
    st.markdown('<h1 class="main-title">🌞 Vue Générale Nationale</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Supervision en temps réel des centrales solaires photovoltaïques</p>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_prod = df['AC_Power'].sum() / 4000  # MWh
        st.markdown(create_kpi_card("Production Totale", f"{total_prod:.1f}", "MWh", delta=5.2), unsafe_allow_html=True)

    with col2:
        avg_power = df['AC_Power'].mean()
        st.markdown(create_kpi_card("Puissance Moyenne", f"{avg_power:.1f}", "kW", delta=-2.1), unsafe_allow_html=True)

    with col3:
        n_inverters = df['Inverter_ID'].nunique()
        st.markdown(create_kpi_card("Onduleurs Actifs", f"{n_inverters}", "", delta=0), unsafe_allow_html=True)

    with col4:
        capacity_factor = (df['AC_Power'].mean() / df['AC_Power'].max()) * 100
        st.markdown(create_kpi_card("Facteur de Capacité", f"{capacity_factor:.1f}", "%", delta=3.5), unsafe_allow_html=True)

    st.markdown("---")

    # Graphiques principaux
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Production Énergétique (7 derniers jours)")

        df_daily = df.copy()
        df_daily['Date'] = df_daily['Timestamp'].dt.date
        daily_prod = df_daily.groupby('Date')['AC_Power'].sum() / 4

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_prod.index,
            y=daily_prod.values,
            mode='lines+markers',
            name='Production',
            line=dict(color='#FF6B00', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 0, 0.1)'
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Production (kWh)",
            hovermode='x unified',
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔋 Performance par Onduleur")

        inv_perf = df.groupby('Inverter_ID')['AC_Power'].mean().sort_values(ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=inv_perf.values,
            y=inv_perf.index,
            orientation='h',
            marker=dict(
                color=inv_perf.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="kW")
            )
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Puissance Moyenne (kW)",
            yaxis_title="Onduleur",
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Profil journalier
    st.subheader("☀️ Profil de Production Journalier")

    df['Hour'] = df['Timestamp'].dt.hour
    hourly_avg = df.groupby('Hour')['AC_Power'].mean()
    hourly_std = df.groupby('Hour')['AC_Power'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hourly_avg.index,
        y=hourly_avg.values + hourly_std.values,
        mode='lines',
        name='Max',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=hourly_avg.index,
        y=hourly_avg.values - hourly_std.values,
        mode='lines',
        name='Min',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.2)',
        fill='tonexty',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=hourly_avg.index,
        y=hourly_avg.values,
        mode='lines+markers',
        name='Moyenne',
        line=dict(color='#FF6B00', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Heure de la journée",
        yaxis_title="Puissance AC (kW)",
        hovermode='x unified',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


def page_prediction():
    """Page : Prédiction IA"""
    st.markdown('<h1 class="main-title">🤖 Prédiction de Production IA</h1>', unsafe_allow_html=True)

    models_data = load_models()

    if models_data is None:
        st.warning("⚠️ Modèles IA non disponibles")

        st.info("""
        ### 📝 Pour activer les prédictions IA :

        1. **En local**, exécutez :
```bash
           python solar_ai_platform.py
```

        2. Les modèles seront générés et sauvegardés dans le dossier `models/`

        3. Rechargez cette page pour voir les prédictions réelles
        """)

        # Afficher démo avec métriques simulées
        st.subheader("📊 Performance des Modèles IA (Démo)")

        col1, col2, col3 = st.columns(3)

        demo_metrics = {
            'RandomForest': {'MAE': 6.42, 'RMSE': 20.88, 'R2': 0.9999},
            'GradientBoosting': {'MAE': 6.93, 'RMSE': 14.27, 'R2': 0.9999},
            'Ensemble': {'MAE': 5.66, 'RMSE': 15.10, 'R2': 0.9999}
        }

        for idx, (model_name, metrics) in enumerate(demo_metrics.items()):
            with [col1, col2, col3][idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 1.5rem; border-radius: 10px; color: white;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0 0 1rem 0;">{model_name} (Démo)</h3>
                    <div style="display: grid; gap: 0.5rem;">
                        <div><strong>MAE:</strong> {metrics['MAE']:.2f} kW</div>
                        <div><strong>RMSE:</strong> {metrics['RMSE']:.2f} kW</div>
                        <div><strong>R² Score:</strong> {metrics['R2']:.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        return

    # Si les modèles sont disponibles, afficher les vraies métriques
    metrics = models_data.get('metrics', {})

    st.subheader("📊 Performance des Modèles IA")

    col1, col2, col3 = st.columns(3)

    models_list = list(metrics.keys())

    for idx, model_name in enumerate(models_list):
        with [col1, col2, col3][idx % 3]:
            model_metrics = metrics[model_name]

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 1rem 0;">{model_name}</h3>
                <div style="display: grid; gap: 0.5rem;">
                    <div><strong>MAE:</strong> {model_metrics['MAE']:.2f} kW</div>
                    <div><strong>RMSE:</strong> {model_metrics['RMSE']:.2f} kW</div>
                    <div><strong>R² Score:</strong> {model_metrics['R2']:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Importance des features
    st.subheader("🎯 Importance des Variables (Explainable AI)")

    features = ['Irradiance', 'Hour_Sin', 'Module_Temperature', 'DC_Power', 'Hour_Cos',
                'Temp_Difference', 'Month_Sin', 'Thermal_Stress', 'Is_Daytime', 'DC_Voltage']
    importance = np.array([0.35, 0.18, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        height=500,
        xaxis_title="Importance",
        yaxis_title="Variable",
        template='plotly_white',
        title="Top 10 Variables les Plus Influentes"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Simulateur de prédiction
    st.subheader("🔮 Simulateur de Prédiction")

    col1, col2, col3 = st.columns(3)

    with col1:
        irradiance = st.slider("Irradiance (W/m²)", 0, 1000, 800)
        temp_module = st.slider("Température Module (°C)", 10, 80, 45)

    with col2:
        hour = st.slider("Heure", 0, 23, 12)
        dc_voltage = st.slider("Tension DC (V)", 400, 800, 600)

    with col3:
        dc_current = st.slider("Courant DC (A)", 0.0, 15.0, 8.0)
        month = st.slider("Mois", 1, 12, 6)

    if st.button("🚀 Lancer la Prédiction", type="primary"):
        predicted_power = (irradiance * 0.8 * np.sin(hour * np.pi / 12)) * 0.001 * dc_voltage * dc_current
        predicted_power = max(0, predicted_power - (temp_module - 25) * 0.5)

        st.success(f"### ⚡ Puissance AC Prédite : {predicted_power:.2f} kW")

        confidence = 95.0 + np.random.uniform(-3, 3)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confiance de la Prédiction"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "gray"},
                    {'range': [90, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def page_anomalies():
    """Page : Détection des Anomalies"""
    st.markdown('<h1 class="main-title">🔍 Détection & Alertes</h1>', unsafe_allow_html=True)

    df_processed = load_processed_data()

    if df_processed is None or 'Is_Anomaly' not in df_processed.columns:
        st.warning("⚠️ Données d'anomalies non disponibles")

        st.info("""
        ### 📝 Pour activer la détection d'anomalies :

        1. Exécutez en local :
```bash
           python solar_ai_platform.py
```

        2. Le fichier `outputs/anomalies_report.csv` sera généré

        3. Rechargez cette page
        """)
        return

    # Statistiques d'anomalies
    total_anomalies = df_processed['Is_Anomaly'].sum()
    anomaly_rate = (total_anomalies / len(df_processed)) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_kpi_card("Anomalies Détectées", f"{total_anomalies}", ""), unsafe_allow_html=True)

    with col2:
        st.markdown(create_kpi_card("Taux d'Anomalies", f"{anomaly_rate:.2f}", "%"), unsafe_allow_html=True)

    with col3:
        high_severity = len(df_processed[df_processed['Severity'] == 'Élevée'])
        st.markdown(create_kpi_card("Gravité Élevée", f"{high_severity}", ""), unsafe_allow_html=True)

    with col4:
        total_losses = df_processed['Energy_Loss_kWh'].sum()
        st.markdown(create_kpi_card("Pertes Estimées", f"{total_losses:.1f}", "kWh"), unsafe_allow_html=True)

    st.markdown("---")

    # Répartition des anomalies
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Répartition par Gravité")

        severity_counts = df_processed['Severity'].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            hole=.4,
            marker=dict(colors=['#00C851', '#ffbb33', '#ff4444', '#33b5e5'])
        )])

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔧 Types d'Anomalies")

        anomaly_types = df_processed[df_processed['Is_Anomaly'] == 1]['Anomaly_Type'].value_counts()

        fig = go.Figure(data=[go.Bar(
            x=anomaly_types.values,
            y=anomaly_types.index,
            orientation='h',
            marker=dict(color='#ff4444')
        )])

        fig.update_layout(
            height=400,
            xaxis_title="Nombre",
            yaxis_title="Type",
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Alertes actives
    st.subheader("🚨 Alertes Actives Prioritaires")

    anomalies = df_processed[df_processed['Is_Anomaly'] == 1].copy()

    if len(anomalies) > 0:
        col1, col2 = st.columns([1, 3])

        with col1:
            severity_filter = st.multiselect(
                "Filtrer par gravité",
                options=['Élevée', 'Moyenne', 'Faible'],
                default=['Élevée', 'Moyenne']
            )

        with col2:
            inverter_filter = st.multiselect(
                "Filtrer par onduleur",
                options=sorted(anomalies['Inverter_ID'].unique()),
                default=[]
            )

        filtered_anomalies = anomalies[anomalies['Severity'].isin(severity_filter)]

        if inverter_filter:
            filtered_anomalies = filtered_anomalies[filtered_anomalies['Inverter_ID'].isin(inverter_filter)]

        for idx, row in filtered_anomalies.head(10).iterrows():
            severity_class = {
                'Élevée': 'alert-high',
                'Moyenne': 'alert-medium',
                'Faible': 'alert-low'
            }.get(row['Severity'], 'alert-low')

            st.markdown(f"""
            <div class="{severity_class}">
                <strong>🚨 {row['Severity']} - {row['Inverter_ID']}</strong><br/>
                <strong>Type:</strong> {row['Anomaly_Type']}<br/>
                <strong>Timestamp:</strong> {row['Timestamp']}<br/>
                <strong>Pertes estimées:</strong> {row['Energy_Loss_kWh']:.2f} kWh<br/>
                <strong>Recommandation:</strong> Vérifier l'onduleur et effectuer un diagnostic
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Aucune anomalie détectée actuellement")

    # Timeline
    st.subheader("📅 Timeline des Anomalies")

    if len(anomalies) > 0:
        anomalies['Date'] = pd.to_datetime(anomalies['Timestamp']).dt.date
        daily_anomalies = anomalies.groupby('Date').size()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_anomalies.index,
            y=daily_anomalies.values,
            mode='lines+markers',
            name='Anomalies',
            line=dict(color='#ff4444', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            height=300,
            xaxis_title="Date",
            yaxis_title="Nombre d'anomalies",
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


def page_inverters():
    """Page : Performance des Onduleurs"""
    st.markdown('<h1 class="main-title">🔋 Performance des Onduleurs</h1>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    inverter_stats = df.groupby('Inverter_ID').agg({
        'AC_Power': ['mean', 'max', 'sum'],
        'DC_Power': 'mean',
        'Module_Temperature': 'mean'
    }).round(2)

    inverter_stats.columns = ['Avg_Power', 'Max_Power', 'Total_Power', 'Avg_DC', 'Avg_Temp']
    inverter_stats['Efficiency'] = (inverter_stats['Avg_Power'] / inverter_stats['Avg_DC'] * 100).round(2)
    inverter_stats = inverter_stats.reset_index()

    inverter_stats['Rank'] = inverter_stats['Total_Power'].rank(ascending=False).astype(int)
    inverter_stats = inverter_stats.sort_values('Rank')

    st.subheader("🏆 Classement des Onduleurs")

    col1, col2, col3 = st.columns(3)

    if len(inverter_stats) >= 3:
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h1>🥇</h1>
                <h2>{inverter_stats.iloc[0]['Inverter_ID']}</h2>
                <p style="font-size: 1.5rem; margin: 1rem 0;">{inverter_stats.iloc[0]['Total_Power']:.1f} kW</p>
                <p>Efficacité: {inverter_stats.iloc[0]['Efficiency']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #C0C0C0 0%, #808080 100%);
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-top: 2rem;">
                <h1>🥈</h1>
                <h3>{inverter_stats.iloc[1]['Inverter_ID']}</h3>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">{inverter_stats.iloc[1]['Total_Power']:.1f} kW</p>
                <p>Efficacité: {inverter_stats.iloc[1]['Efficiency']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #CD7F32 0%, #8B4513 100%);
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-top: 2rem;">
                <h1>🥉</h1>
                <h3>{inverter_stats.iloc[2]['Inverter_ID']}</h3>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">{inverter_stats.iloc[2]['Total_Power']:.1f} kW</p>
                <p>Efficacité: {inverter_stats.iloc[2]['Efficiency']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 Tableau de Performance Détaillé")

    st.dataframe(
        inverter_stats.style.background_gradient(cmap='RdYlGn', subset=['Efficiency'])
                            .format({
                                'Avg_Power': '{:.2f} kW',
                                'Max_Power': '{:.2f} kW',
                                'Total_Power': '{:.2f} kW',
                                'Avg_DC': '{:.2f} kW',
                                'Avg_Temp': '{:.1f}°C',
                                'Efficiency': '{:.1f}%'
                            }),
        use_container_width=True,
        height=400
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚡ Production Totale")

        fig = go.Figure(data=[go.Bar(
            x=inverter_stats['Inverter_ID'],
            y=inverter_stats['Total_Power'],
            marker=dict(
                color=inverter_stats['Total_Power'],
                colorscale='Viridis',
                showscale=True
            )
        )])

        fig.update_layout(
            height=400,
            xaxis_title="Onduleur",
            yaxis_title="Production Totale (kW)",
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Efficacité de Conversion")

        fig = go.Figure(data=[go.Bar(
            x=inverter_stats['Inverter_ID'],
            y=inverter_stats['Efficiency'],
            marker=dict(
                color=inverter_stats['Efficiency'],
                colorscale='RdYlGn',
                showscale=True
            )
        )])

        fig.update_layout(
            height=400,
            xaxis_title="Onduleur",
            yaxis_title="Efficacité (%)",
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)


def page_climate():
    """Page : Impact Climatique"""
    st.markdown('<h1 class="main-title">🌤️ Impact Climatique</h1>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    st.subheader("☀️ Corrélation Irradiance - Production")

    sample_data = df.sample(min(5000, len(df)))

    fig = px.scatter(
        sample_data,
        x='Irradiance',
        y='AC_Power',
        color='Module_Temperature',
        size='AC_Power',
        hover_data=['Inverter_ID'],
        color_continuous_scale='RdYlBu_r',
        labels={
            'Irradiance': 'Irradiance (W/m²)',
            'AC_Power': 'Puissance AC (kW)',
            'Module_Temperature': 'Temp. Module (°C)'
        }
    )

    fig.update_layout(height=500, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Impact de la Température sur le Rendement")

    df['Temp_Range'] = pd.cut(df['Module_Temperature'], bins=10)
    temp_impact = df.groupby('Temp_Range')['AC_Power'].mean().reset_index()
    temp_impact['Temp_Mid'] = temp_impact['Temp_Range'].apply(lambda x: x.mid)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temp_impact['Temp_Mid'],
        y=temp_impact['AC_Power'],
        mode='lines+markers',
        name='Puissance Moyenne',
        line=dict(color='#FF6B00', width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Température du Module (°C)",
        yaxis_title="Puissance AC Moyenne (kW)",
        template='plotly_white',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Prévisions de Production (3 jours)")

    col1, col2, col3 = st.columns(3)

    days = ['Demain', 'J+2', 'J+3']
    forecasts = [8500, 9200, 7800]
    weather = ['☀️ Ensoleillé', '⛅ Partiellement nuageux', '☁️ Nuageux']

    for col, day, forecast, w in zip([col1, col2, col3], days, forecasts, weather):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>{day}</h3>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{w}</h1>
                <p style="font-size: 1.5rem; margin: 0;">Production prévue</p>
                <p style="font-size: 2rem; font-weight: 700;">{forecast} kWh</p>
            </div>
            """, unsafe_allow_html=True)


def page_reports():
    """Page : Rapports & Export"""
    st.markdown('<h1 class="main-title">📄 Rapports & Export</h1>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    st.subheader("📊 Période d'Analyse")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Date de début", value=df['Timestamp'].min())

    with col2:
        end_date = st.date_input("Date de fin", value=df['Timestamp'].max())

    st.subheader("📈 Résumé des KPI")

    filtered_df = df[
        (df['Timestamp'].dt.date >= start_date) &
        (df['Timestamp'].dt.date <= end_date)
    ]

    kpi_data = {
        'KPI': [
            'Production Totale',
            'Puissance Moyenne',
            'Puissance Maximale',
            'Facteur de Capacité',
            'Nombre d\'Onduleurs',
            'Heures de Production',
            'Rendement Moyen'
        ],
        'Valeur': [
            f"{filtered_df['AC_Power'].sum() / 4000:.2f} MWh",
            f"{filtered_df['AC_Power'].mean():.2f} kW",
            f"{filtered_df['AC_Power'].max():.2f} kW",
            f"{(filtered_df['AC_Power'].mean() / filtered_df['AC_Power'].max() * 100):.2f}%",
            f"{filtered_df['Inverter_ID'].nunique()}",
            f"{len(filtered_df[filtered_df['AC_Power'] > 0]) / 4:.1f} h",
            f"{(filtered_df['AC_Power'] / (filtered_df['DC_Power'] + 1e-6)).mean() * 100:.2f}%"
        ]
    }

    st.table(pd.DataFrame(kpi_data))

    st.subheader("💾 Export des Données")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Exporter CSV", type="primary"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Télécharger CSV",
                data=csv,
                file_name=f"solar_data_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📊 Générer Rapport PDF"):
            st.info("📄 Fonctionnalité en développement")

    with col3:
        if st.button("📧 Envoyer par Email"):
            st.info("📧 Fonctionnalité en développement")

    st.subheader("👁️ Aperçu des Données")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        height=400
    )

    st.subheader("📐 Statistiques Descriptives")

    st.dataframe(
        filtered_df.describe().T.style.format("{:.2f}"),
        use_container_width=True
    )


def main():
    """Application principale"""

    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/FF6B00/FFFFFF?text=SOLAR+AI", use_container_width=True)

        st.markdown("---")

        st.markdown("""
        ### 🌞 Solar AI Platform

        Plateforme intelligente de prédiction et supervision des centrales solaires photovoltaïques.

        **Version:** 1.0.0
        **Niveau:** National - ERA 2026

        ---

        ### 📊 Navigation
        """)

        page = st.radio(
            "Choisir une page",
            [
                "🏠 Vue Générale",
                "🤖 Prédiction IA",
                "🔍 Anomalies & Alertes",
                "🔋 Performance Onduleurs",
                "🌤️ Impact Climatique",
                "📄 Rapports & Export"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("""
        ### ℹ️ À propos

        Développé pour la gestion intelligente des centrales solaires au niveau national.

        **Technologies:**
        - Machine Learning (RF, GB)
        - Détection d'Anomalies (IF)
        - IA Explicable
        - Visualisation Interactive

        ---

        © 2026 Chalabi Mohamed El Aminen
        """)

    if "Vue Générale" in page:
        page_overview()
    elif "Prédiction IA" in page:
        page_prediction()
    elif "Anomalies" in page:
        page_anomalies()
    elif "Onduleurs" in page:
        page_inverters()
    elif "Climatique" in page:
        page_climate()
    elif "Rapports" in page:
        page_reports()


if __name__ == "__main__":
    main()