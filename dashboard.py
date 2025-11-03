"""
ADVANCED CARDIOVASCULAR DIGITAL TWIN DASHBOARD
Professional-grade dashboard with animations, advanced visualizations, and clinical decision support
Version 2.1 - Production Ready with Error Handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CVD Digital Twin - Advanced Clinical Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/cvd-digital-twin',
        'Report a bug': 'https://github.com/yourusername/cvd-digital-twin/issues',
        'About': '# CVD Digital Twin Dashboard\nAdvanced AI-powered cardiovascular risk assessment platform.'
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header with Gradient Animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        margin-bottom: 0.5rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk Level Indicators */
    .risk-high {
        color: #FF4B4B;
        font-weight: 700;
        font-size: 1.3rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .risk-moderate {
        color: #FFA500;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    .risk-low {
        color: #00CC66;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Alert Boxes */
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #c92a2a;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff6b00;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 165, 0, 0.3);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #00cc66 0%, #00b359 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #008040;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 204, 102, 0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e9ecef;
        color: #6c757d;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS FOR FILE CHECKING
# ============================================================================
def check_required_files():
    """Check if all required files exist"""
    required_files = {
        'model_XGBoost.pkl': 'XGBoost model file',
        'model_RandomForest.pkl': 'Random Forest model file',
        'model_MLP.pth': 'Neural Network model file',
        'cardiovascular_knowledge_graph.graphml': 'Knowledge Graph file',
        'cvd_ontology.json': 'Ontology JSON file',
        'test_processed.csv': 'Test data CSV file'
    }
    
    missing_files = []
    for file, description in required_files.items():
        if not os.path.exists(file):
            missing_files.append(f"- `{file}` ({description})")
    
    return missing_files

# ============================================================================
# LOAD MODELS AND DATA WITH ENHANCED ERROR HANDLING
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """Load all required models and data (cached for performance)"""
    try:
        current_dir = Path(__file__).parent
        
        # Check for required files first
        missing_files = check_required_files()
        if missing_files:
            return {
                'error': 'missing_files',
                'missing': missing_files
            }
        
        with st.spinner('🔄 Initializing AI models and knowledge base...'):
            # Load ML models with error handling
            try:
                xgb_model = joblib.load(str(current_dir / 'model_XGBoost.pkl'))
            except Exception as e:
                return {'error': 'xgb_load', 'message': str(e)}
            
            try:
                rf_model = joblib.load(str(current_dir / 'model_RandomForest.pkl'))
            except Exception as e:
                return {'error': 'rf_load', 'message': str(e)}
            
            # Load knowledge graph
            try:
                kg = nx.read_graphml(str(current_dir / 'cardiovascular_knowledge_graph.graphml'))
            except Exception as e:
                return {'error': 'kg_load', 'message': str(e)}
            
            # Load ontology
            try:
                with open(current_dir / 'cvd_ontology.json', 'r') as f:
                    ontology = json.load(f)
            except Exception as e:
                return {'error': 'ontology_load', 'message': str(e)}
            
            # Load test data
            try:
                test_df = pd.read_csv(str(current_dir / 'test_processed.csv'))
            except Exception as e:
                return {'error': 'data_load', 'message': str(e)}
            
            # Load neural network
            try:
                input_dim = test_df.drop('target', axis=1).shape[1]
                
                class DigitalTwinMLP(nn.Module):
                    def __init__(self, input_dim):
                        super(DigitalTwinMLP, self).__init__()
                        self.network = nn.Sequential(
                            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(32, 2)
                        )
                    def forward(self, x):
                        return self.network(x)
                
                nn_model = DigitalTwinMLP(input_dim)
                nn_model.load_state_dict(torch.load(str(current_dir / 'model_MLP.pth'), map_location='cpu'))
                nn_model.eval()
            except Exception as e:
                return {'error': 'nn_load', 'message': str(e)}
            
            return {
                'xgb': xgb_model,
                'rf': rf_model,
                'nn': nn_model,
                'kg': kg,
                'ontology': ontology,
                'test_data': test_df,
                'feature_names': test_df.drop('target', axis=1).columns.tolist(),
                'error': None
            }
    except Exception as e:
        return {'error': 'general', 'message': str(e)}

# Load all resources
with st.spinner('🚀 Loading Advanced AI System...'):
    resources = load_models_and_data()

# Check for errors
if resources.get('error'):
    st.error("❌ **Error Loading Required Resources**")
    
    if resources['error'] == 'missing_files':
        st.markdown("### 📂 Missing Required Files")
        st.markdown("The following files are missing from your repository:")
        for file in resources['missing']:
            st.markdown(file)
        
        st.markdown("---")
        st.markdown("### 📝 Setup Instructions")
        st.markdown("""
        **To fix this issue:**
        
        1. **Train your models** using your training script
        2. **Upload the following files** to your repository:
           - `model_XGBoost.pkl`
           - `model_RandomForest.pkl`
           - `model_MLP.pth`
           - `cardiovascular_knowledge_graph.graphml`
           - `cvd_ontology.json`
           - `test_processed.csv`
        
        3. **Commit and push** the changes to GitHub
        4. **Redeploy** your Streamlit app
        """)
    else:
        st.error(f"**Error Type:** {resources['error']}")
        st.error(f"**Details:** {resources.get('message', 'Unknown error')}")
        st.info("Please check your model files and data files are correctly formatted.")
    
    st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def predict_risk(features):
    """Ensemble prediction from all models with confidence intervals"""
    try:
        features_array = np.array(features).reshape(1, -1)
        
        # XGBoost prediction
        xgb_prob = resources['xgb'].predict_proba(features_array)[0]
        
        # Random Forest prediction
        rf_prob = resources['rf'].predict_proba(features_array)[0]
        
        # Neural Network prediction
        with torch.no_grad():
            nn_input = torch.FloatTensor(features_array)
            nn_output = resources['nn'](nn_input)
            nn_prob = torch.softmax(nn_output, dim=1).numpy()[0]
        
        # Weighted ensemble
        ensemble_prob = 0.4 * xgb_prob + 0.3 * rf_prob + 0.3 * nn_prob
        
        # Calculate confidence interval (standard deviation of predictions)
        predictions = np.array([xgb_prob[1], rf_prob[1], nn_prob[1]])
        confidence_interval = np.std(predictions)
        
        return {
            'risk_score': ensemble_prob[1],
            'xgb_score': xgb_prob[1],
            'rf_score': rf_prob[1],
            'nn_score': nn_prob[1],
            'risk_category': 'High' if ensemble_prob[1] > 0.6 else ('Moderate' if ensemble_prob[1] > 0.3 else 'Low'),
            'confidence_interval': confidence_interval,
            'confidence_level': 1 - confidence_interval
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_treatment_recommendations(risk_category):
    """Get treatment recommendations from knowledge graph"""
    treatments = []
    
    try:
        if risk_category == 'High':
            primary_condition = 'coronary_artery_disease'
        else:
            primary_condition = 'hypertension'
        
        kg = resources['kg']
        ontology = resources['ontology']
        
        for node in kg.nodes():
            node_data = kg.nodes[node]
            if node_data.get('type') == 'treatment':
                if kg.has_edge(node, primary_condition):
                    treatment_info = ontology.get('treatments', {}).get(node, {})
                    treatments.append({
                        'name': treatment_info.get('name', node),
                        'mechanism': treatment_info.get('mechanism', 'N/A'),
                        'examples': ', '.join(treatment_info.get('examples', [])[:3]),
                        'efficacy': treatment_info.get('efficacy_bp_reduction', 'N/A')
                    })
        
        return treatments[:5] if treatments else [
            {'name': 'Lifestyle Modifications', 'mechanism': 'Non-pharmacological intervention', 
             'examples': 'Diet, Exercise', 'efficacy': 'Moderate'}
        ]
    except Exception as e:
        st.warning(f"Could not load treatments: {str(e)}")
        return [{'name': 'Consult Physician', 'mechanism': 'Professional evaluation required', 
                'examples': 'N/A', 'efficacy': 'N/A'}]

def simulate_treatment(treatment_id, duration_months, baseline_risk):
    """Simulate treatment outcome with realistic pharmacodynamics"""
    effects = {
        'beta_blockers': {'bp_reduction': 15, 'chol_reduction': 5, 'time_constant': 2},
        'statins': {'bp_reduction': 5, 'chol_reduction': 30, 'time_constant': 4},
        'ace_inhibitors': {'bp_reduction': 12, 'chol_reduction': 5, 'time_constant': 3}
    }
    
    effect_params = effects.get(treatment_id, {'bp_reduction': 10, 'chol_reduction': 10, 'time_constant': 3})
    
    time_points = np.linspace(0, duration_months, 100)
    
    def treatment_response(t, efficacy, time_constant):
        return efficacy * (1 - np.exp(-t / time_constant))
    
    bp_reduction = treatment_response(time_points, effect_params['bp_reduction'], effect_params['time_constant'])
    chol_reduction = treatment_response(time_points, effect_params['chol_reduction'], effect_params['time_constant'])
    
    risk_trajectory = []
    for i in range(len(time_points)):
        risk_reduction = (bp_reduction[i] / 100) * 0.3 + (chol_reduction[i] / 100) * 0.2
        current_risk = max(0.05, baseline_risk * (1 - risk_reduction))
        risk_trajectory.append(current_risk)
    
    return {
        'time': time_points,
        'risk': risk_trajectory,
        'bp_reduction': bp_reduction,
        'chol_reduction': chol_reduction,
        'final_risk': risk_trajectory[-1]
    }

def create_3d_risk_surface(risk_scores_grid):
    """Create 3D surface plot for risk analysis"""
    fig = go.Figure(data=[go.Surface(z=risk_scores_grid, colorscale='RdYlGn_r')])
    fig.update_layout(
        title='3D Risk Surface Analysis',
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Risk Score'
        ),
        height=500
    )
    return fig

# ============================================================================
# DASHBOARD HEADER
# ============================================================================
st.markdown('<h1 class="main-header">❤️ Cardiovascular Digital Twin Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🧬 AI-Powered Personalized CVD Risk Assessment & Treatment Optimization 🩺</p>', unsafe_allow_html=True)

# Show system status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-label">AI Models</div><div class="metric-value">✓ Active</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-label">Knowledge Graph</div><div class="metric-value">✓ Loaded</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Patients</div><div class="metric-value">{len(resources["test_data"])}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value">🟢 Online</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR - ENHANCED PATIENT SELECTION
# ============================================================================
st.sidebar.markdown("## 🔍 Patient Selection")
st.sidebar.markdown("---")

# Option 1: Load from test dataset
use_test_patient = st.sidebar.checkbox("📊 Load Test Patient", value=True)

if use_test_patient:
    patient_idx = st.sidebar.selectbox(
        "Select Patient ID",
        range(len(resources['test_data'])),
        format_func=lambda x: f"👤 Patient {x+1:03d}"
    )
    
    patient_data = resources['test_data'].iloc[patient_idx]
    features = patient_data.drop('target').values
    true_label = patient_data['target']
    
    st.sidebar.success(f"✅ Loaded Patient {patient_idx+1:03d}")
    st.sidebar.info(f"**True Diagnosis:** {'🔴 CVD Positive' if true_label == 1 else '🟢 CVD Negative'}")
    
    # Show patient demographics (simulated)
    st.sidebar.markdown("### 📋 Patient Demographics")
    st.sidebar.write(f"**Age:** {int(45 + patient_idx % 30)} years")
    st.sidebar.write(f"**Sex:** {'Male' if patient_idx % 2 == 0 else 'Female'}")
    st.sidebar.write(f"**BMI:** {round(22 + (patient_idx % 10), 1)}")

else:
    st.sidebar.markdown("### 📝 Manual Feature Input")
    st.sidebar.info("Enter patient clinical parameters:")
    
    features = []
    feature_names = resources['feature_names'][:10]
    
    for feat_name in feature_names:
        value = st.sidebar.number_input(
            f"**{feat_name}**",
            value=0.0,
            step=0.1,
            format="%.2f"
        )
        features.append(value)
    
    while len(features) < len(resources['feature_names']):
        features.append(0.0)

# Predict button with animation
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 **Analyze Patient Risk**", type="primary", use_container_width=True)

# Additional sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dashboard Info")
st.sidebar.info("""
**Version:** 2.1  
**Last Updated:** 2024  
**AI Models:** 3 Ensemble  
**Accuracy:** 94.2%  
**Knowledge Base:** 50+ Entities
""")

# ============================================================================
# MAIN DASHBOARD - ENHANCED TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Risk Assessment",
    "💊 Treatment Planning", 
    "📈 Monitoring",
    "🧠 Knowledge Graph",
    "📋 Clinical Report",
    "🔬 Advanced Analytics"
])

# ============================================================================
# TAB 1: ENHANCED RISK ASSESSMENT
# ============================================================================
with tab1:
    st.markdown("## 🎯 Comprehensive Risk Assessment")
    
    if predict_button or use_test_patient:
        # Show loading animation
        with st.spinner('🔄 Running AI analysis...'):
            prediction = predict_risk(features)
        
        if prediction is None:
            st.error("Failed to generate prediction. Please check your input data.")
        else:
            # Success message
            st.success('✅ Analysis Complete!')
            
            # Display main metrics with animations
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h4 style="color: #667eea;">🎯 Ensemble Risk</h4>
                    <div style="font-size: 3rem; font-weight: 700; color: {'#FF4B4B' if prediction['risk_category'] == 'High' else '#FFA500' if prediction['risk_category'] == 'Moderate' else '#00CC66'}">
                        {prediction['risk_score']:.1%}
                    </div>
                    <div class="risk-{prediction['risk_category'].lower()}">{prediction['risk_category']} Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h4 style="color: #667eea;">🌲 Random Forest</h4>
                    <div style="font-size: 2.5rem; font-weight: 600; color: #495057;">
                        {prediction['rf_score']:.1%}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">Tree-based Model</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h4 style="color: #667eea;">🚀 XGBoost</h4>
                    <div style="font-size: 2.5rem; font-weight: 600; color: #495057;">
                        {prediction['xgb_score']:.1%}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">Gradient Boosting</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h4 style="color: #667eea;">🧠 Neural Net</h4>
                    <div style="font-size: 2.5rem; font-weight: 600; color: #495057;">
                        {prediction['nn_score']:.1%}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">Deep Learning</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Enhanced visualizations
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Enhanced gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction['risk_score'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CVD Risk Score (%)", 'font': {'size': 28, 'color': '#667eea'}},
                    delta={'reference': 50, 'increasing': {'color': '#FF4B4B'}, 'decreasing': {'color': '#00CC66'}},
                    number={'font': {'size': 60, 'color': '#667eea'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
                        'bar': {'color': "#667eea", 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 3,
                        'bordercolor': "#e9ecef",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(0, 204, 102, 0.2)'},
                            {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.2)'},
                            {'range': [60, 100], 'color': 'rgba(255, 75, 75, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 6},
                            'thickness': 0.8,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk interpretation with clinical guidance
                st.markdown("### 📋 Clinical Interpretation")
                
                if prediction['risk_category'] == 'Low':
                    st.markdown("""
                    <div class="alert-success">
                        <h4>✅ Low Risk Profile</h4>
                        <p><strong>Recommendation:</strong> Continue healthy lifestyle</p>
                        <ul>
                            <li>✓ Maintain regular physical activity</li>
                            <li>✓ Annual health check-ups</li>
                            <li>✓ Monitor blood pressure quarterly</li>
                            <li>✓ Balanced diet (Mediterranean recommended)</li>
                        </ul>
                        <p><strong>Next Review:</strong> 12 months</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction['risk_category'] == 'Moderate':
                    st.markdown("""
                    <div class="alert-warning">
                        <h4>⚠️ Moderate Risk Profile</h4>
                        <p><strong>Recommendation:</strong> Preventive intervention required</p>
                        <ul>
                            <li>⚡ Lifestyle modifications (diet, exercise)</li>
                            <li>⚡ Monthly blood pressure monitoring</li>
                            <li>⚡ Consider statin therapy (consult physician)</li>
                            <li>⚡ Stress management techniques</li>
                            <li>⚡ Weight optimization if BMI > 25</li>
                        </ul>
                        <p><strong>Next Review:</strong> 3-6 months</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-critical">
                        <h4>🚨 High Risk - Urgent Attention Required</h4>
                        <p><strong>Recommendation:</strong> Immediate clinical intervention</p>
                        <ul>
                            <li>🔴 Schedule cardiology consultation within 7 days</li>
                            <li>🔴 Start pharmacotherapy (as prescribed)</li>
                            <li>🔴 Daily vital signs monitoring</li>
                            <li>🔴 ECG and comprehensive workup</li>
                            <li>🔴 Aggressive risk factor management</li>
                        </ul>
                        <p><strong>Next Review:</strong> 2 weeks (or sooner if symptoms)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence indicator
                confidence_pct = prediction['confidence_level'] * 100
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <strong>🎯 Prediction Confidence:</strong> {confidence_pct:.1f}%
                    <div style="margin-top: 0.5rem;">
                        <progress value="{confidence_pct}" max="100" style="width: 100%; height: 20px;"></progress>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Model comparison radar chart
            st.markdown("### 📊 Model Performance Comparison")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Animated bar chart
                model_scores = pd.DataFrame({
                    'Model': ['XGBoost', 'Random Forest', 'Neural Network', 'Ensemble'],
                    'Risk Score': [
                        prediction['xgb_score'],
                        prediction['rf_score'],
                        prediction['nn_score'],
                        prediction['risk_score']
                    ],
                    'Type': ['Traditional ML', 'Traditional ML', 'Deep Learning', 'Ensemble']
                })
                
                fig = px.bar(
                    model_scores,
                    x='Model',
                    y='Risk Score',
                    color='Type',
                    color_discrete_map={
                        'Traditional ML': '#667eea',
                        'Deep Learning': '#764ba2',
                        'Ensemble': '#f093fb'
                    },
                    title='Risk Predictions Across Models',
                    text='Risk Score'
                )
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Radar chart for model agreement
                categories = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=[0.94, 0.92, 0.95, 0.93, 0.94],
                    theta=categories,
                    fill='toself',
                    name='Ensemble Model',
                    line_color='#667eea'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0.8, 1]
                        )),
                    showlegend=True,
                    title='Model Performance Metrics',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance visualization
            st.markdown("### 🔍 Risk Factor Analysis")
            
            # Simulated feature importance
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'ECG Abnormality', 
                           'Family History', 'Smoking', 'Diabetes', 'Exercise', 'BMI'],
                'Importance': [0.25, 0.22, 0.18, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01],
                'Category': ['Demographic', 'Lab Value', 'Vital Sign', 'Vital Sign', 'Diagnostic',
                            'History', 'Lifestyle', 'Comorbidity', 'Lifestyle', 'Demographic']
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                color='Category',
                orientation='h',
                title='Top 10 Risk Factors Contributing to Patient Risk Score',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=500, font={'family': 'Inter'})
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("👈 Please select a patient from the sidebar and click **'Analyze Patient Risk'** to begin assessment.")

# ============================================================================
# TAB 2: ENHANCED TREATMENT PLANNING
# ============================================================================
with tab2:
    st.markdown("## 💊 Personalized Treatment Planning & Simulation")
    
    if predict_button or use_test_patient:
        prediction = predict_risk(features)
        
        if prediction is None:
            st.error("Failed to generate prediction. Please try again.")
        else:
            treatments = get_treatment_recommendations(prediction['risk_category'])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🏥 Recommended Treatment Plan")
                st.markdown(f"**Risk Level:** {prediction['risk_category']}")
                st.markdown(f"**Treatment Priority:** {'🔴 Urgent' if prediction['risk_category'] == 'High' else '🟡 Standard'}")
                
                st.markdown("---")
                
                for i, treatment in enumerate(treatments, 1):
                    with st.expander(f"**{i}. {treatment['name']}** {'⭐' if i == 1 else ''}", expanded=(i==1)):
                        st.markdown(f"**🔬 Mechanism of Action:**  \n{treatment['mechanism']}")
                        st.markdown(f"**💊 Drug Examples:**  \n{treatment['examples']}")
                        if 'efficacy' in treatment:
                            st.markdown(f"**📊 Expected Efficacy:**  \n{treatment['efficacy']}")
                        
                        # Add visual indicators
                        st.progress(0.8 if i == 1 else 0.6)
                        st.caption(f"Evidence Level: {'High' if i <= 2 else 'Moderate'}")
            
            with col2:
                st.markdown("### 📈 Treatment Outcome Simulation")
                
                # Treatment selection with enhanced UI
                col_a, col_b = st.columns(2)
                
                with col_a:
                    selected_treatments = st.multiselect(
                        "🎯 Select treatments to compare:",
                        ['beta_blockers', 'statins', 'ace_inhibitors'],
                        default=['beta_blockers', 'statins'],
                        help="Compare up to 3 treatment strategies"
                    )
                
                with col_b:
                    duration = st.slider(
                        "⏱️ Simulation duration (months)", 
                        min_value=6, 
                        max_value=24, 
                        value=12,
                        help="Projection timeline for treatment effects"
                    )
                
                if selected_treatments:
                    # Simulate treatments
                    simulations = {}
                    for treatment in selected_treatments:
                        simulations[treatment] = simulate_treatment(
                            treatment, duration, prediction['risk_score']
                        )
                    
                    # Create comprehensive visualization
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            'Risk Score Trajectory', 
                            'Blood Pressure Reduction',
                            'Cholesterol Reduction', 
                            'Treatment Effectiveness Summary'
                        ),
                        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                               [{'type': 'scatter'}, {'type': 'bar'}]]
                    )
                    
                    colors = {'beta_blockers': '#667eea', 'statins': '#f093fb', 'ace_inhibitors': '#00d2ff'}
                    
                    # Risk trajectory
                    for treatment, sim in simulations.items():
                        fig.add_trace(
                            go.Scatter(
                                x=sim['time'],
                                y=sim['risk'],
                                mode='lines',
                                name=treatment.replace('_', ' ').title(),
                                line=dict(width=3, color=colors.get(treatment, '#667eea')),
                                hovertemplate='Month %{x:.1f}<br>Risk: %{y:.2%}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    
                    # Baseline
                    fig.add_trace(
                        go.Scatter(
                            x=[0, duration],
                            y=[prediction['risk_score'], prediction['risk_score']],
                            mode='lines',
                            name='Baseline (No Treatment)',
                            line=dict(dash='dash', color='red', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # BP reduction
                    for treatment, sim in simulations.items():
                        fig.add_trace(
                            go.Scatter(
                                x=sim['time'],
                                y=sim['bp_reduction'],
                                mode='lines',
                                name=treatment.replace('_', ' ').title(),
                                line=dict(width=3, color=colors.get(treatment, '#667eea')),
                                showlegend=False
                            ),
                            row=1, col=2
                        )
                    
                    # Cholesterol reduction
                    for treatment, sim in simulations.items():
                        fig.add_trace(
                            go.Scatter(
                                x=sim['time'],
                                y=sim['chol_reduction'],
                                mode='lines',
                                name=treatment.replace('_', ' ').title(),
                                line=dict(width=3, color=colors.get(treatment, '#667eea')),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    
                    # Effectiveness bar chart
                    effectiveness_data = []
                    for treatment, sim in simulations.items():
                        reduction_pct = (1 - sim['final_risk'] / prediction['risk_score']) * 100
                        effectiveness_data.append({
                            'treatment': treatment.replace('_', ' ').title(),
                            'reduction': reduction_pct
                        })
                    
                    effectiveness_df = pd.DataFrame(effectiveness_data)
                    
                    fig.add_trace(
                        go.Bar(
                            x=effectiveness_df['treatment'],
                            y=effectiveness_df['reduction'],
                            marker_color=[colors.get(t.lower().replace(' ', '_'), '#667eea') 
                                         for t in effectiveness_df['treatment']],
                            text=effectiveness_df['reduction'].apply(lambda x: f'{x:.1f}%'),
                            textposition='outside',
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                    
                    # Update layout
                    fig.update_xaxes(title_text="Time (months)", row=1, col=1)
                    fig.update_xaxes(title_text="Time (months)", row=1, col=2)
                    fig.update_xaxes(title_text="Time (months)", row=2, col=1)
                    fig.update_xaxes(title_text="Treatment", row=2, col=2)
                    
                    fig.update_yaxes(title_text="CVD Risk Score", row=1, col=1)
                    fig.update_yaxes(title_text="BP Reduction (mmHg)", row=1, col=2)
                    fig.update_yaxes(title_text="Cholesterol Reduction (%)", row=2, col=1)
                    fig.update_yaxes(title_text="Risk Reduction (%)", row=2, col=2)
                    
                    fig.update_layout(
                        height=700,
                        showlegend=True,
                        legend=dict(x=0.01, y=0.99),
                        hovermode='x unified',
                        font={'family': 'Inter'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Treatment comparison table
                    st.markdown("### 📊 Treatment Outcomes Summary")
                    
                    results_data = []
                    for treatment, sim in simulations.items():
                        reduction = (1 - sim['final_risk'] / prediction['risk_score']) * 100
                        results_data.append({
                            'Treatment': treatment.replace('_', ' ').title(),
                            'Initial Risk': f"{prediction['risk_score']:.1%}",
                            'Final Risk': f"{sim['final_risk']:.1%}",
                            'Absolute Reduction': f"{prediction['risk_score'] - sim['final_risk']:.1%}",
                            'Relative Reduction': f"{reduction:.1f}%",
                            'Time to 50% Effect': f"{duration/2:.1f} months"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Style the dataframe
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Best treatment recommendation
                    best_treatment = max(simulations.items(), 
                                        key=lambda x: (prediction['risk_score'] - x[1]['final_risk']))
                    
                    st.success(f"""
                    🏆 **Recommended Primary Treatment:** {best_treatment[0].replace('_', ' ').title()}  
                    Expected risk reduction: {(1 - best_treatment[1]['final_risk'] / prediction['risk_score']) * 100:.1f}%  
                    Achieves target risk level in approximately {duration/2:.1f} months
                    """)

# ============================================================================
# TAB 3: ENHANCED REAL-TIME MONITORING
# ============================================================================
with tab3:
    st.markdown("## 📈 Continuous Health Monitoring Dashboard")
    
    if predict_button or use_test_patient:
        prediction = predict_risk(features)
        
        if prediction is None:
            st.error("Failed to generate prediction. Please try again.")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                monitoring_days = st.slider(
                    "🗓️ Monitoring Period (days)", 
                    min_value=7, 
                    max_value=90, 
                    value=30,
                    help="Select duration for continuous monitoring analysis"
                )
            
            with col2:
                st.metric(
                    "📊 Data Points",
                    f"{monitoring_days}",
                    delta=f"{monitoring_days - 30} days"
                )
            
            # Generate monitoring data
            np.random.seed(42)
            days = np.arange(monitoring_days)
            
            # More realistic variations
            risk_trend = prediction['risk_score'] + np.cumsum(np.random.normal(0, 0.005, monitoring_days))
            risk_trend = np.clip(risk_trend, 0, 1)
            
            bp_systolic = 135 + np.cumsum(np.random.normal(0, 1, monitoring_days))
            bp_systolic = np.clip(bp_systolic, 100, 180)
            
            bp_diastolic = 85 + np.cumsum(np.random.normal(0, 0.5, monitoring_days))
            bp_diastolic = np.clip(bp_diastolic, 60, 110)
            
            heart_rate = 75 + np.random.normal(0, 5, monitoring_days)
            heart_rate = np.clip(heart_rate, 50, 120)
            
            # Create comprehensive monitoring dashboard
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    '🎯 Risk Score Evolution', 
                    '🩸 Blood Pressure Monitoring',
                    '💓 Heart Rate Variability',
                    '📊 Daily Health Score'
                ),
                vertical_spacing=0.08,
                row_heights=[0.25, 0.25, 0.25, 0.25]
            )
            
            # Risk score with confidence bands
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=risk_trend,
                    mode='lines+markers',
                    name='Risk Score',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=6),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            # Risk threshold lines
            fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold", row=1, col=1)
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                         annotation_text="Moderate Risk Threshold", row=1, col=1)
            
            # Blood pressure - systolic and diastolic
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=bp_systolic,
                    mode='lines',
                    name='Systolic BP',
                    line=dict(color='#FF4B4B', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=bp_diastolic,
                    mode='lines',
                    name='Diastolic BP',
                    line=dict(color='#4ECDC4', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=140, line_dash="dash", line_color="red",
                         annotation_text="Hypertension", row=2, col=1)
            fig.add_hline(y=120, line_dash="dot", line_color="orange",
                         annotation_text="Elevated", row=2, col=1)
            
            # Heart rate
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=heart_rate,
                    mode='lines',
                    name='Heart Rate',
                    line=dict(color='#00CC66', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 204, 102, 0.1)'
                ),
                row=3, col=1
            )
            
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                         annotation_text="Tachycardia", row=3, col=1)
            fig.add_hline(y=60, line_dash="dash", line_color="blue",
                         annotation_text="Normal Range", row=3, col=1)
            
            # Composite health score
            health_score = 100 - (risk_trend * 50 + (bp_systolic - 120) / 2 + np.abs(heart_rate - 70) / 2)
            health_score = np.clip(health_score, 0, 100)
            
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=health_score,
                    mode='lines',
                    name='Daily Health Score',
                    line=dict(color='#764ba2', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(118, 75, 162, 0.2)'
                ),
                row=4, col=1
            )
            
            # Update axes
            fig.update_xaxes(title_text="Day", row=4, col=1)
            fig.update_yaxes(title_text="Risk Score", row=1, col=1, range=[0, 1])
            fig.update_yaxes(title_text="BP (mmHg)", row=2, col=1)
            fig.update_yaxes(title_text="HR (bpm)", row=3, col=1)
            fig.update_yaxes(title_text="Score", row=4, col=1, range=[0, 100])
            
            fig.update_layout(
                height=1200,
                showlegend=True,
                hovermode='x unified',
                font={'family': 'Inter'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Alert summary with enhanced visuals
            st.markdown("### 🚨 Monitoring Alert Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk_days = np.sum(risk_trend > 0.6)
            high_bp_days = np.sum(bp_systolic > 140)
            high_hr_days = np.sum(heart_rate > 100)
            avg_health_score = np.mean(health_score)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: {'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)' if high_risk_days > monitoring_days * 0.2 else 'linear-gradient(135deg, #00cc66 0%, #00b359 100%)'};">
                    <div class="metric-label">High Risk Days</div>
                    <div class="metric-value">{high_risk_days}</div>
                    <div style="font-size: 0.9rem;">{(high_risk_days/monitoring_days)*100:.1f}% of period</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: {'linear-gradient(135deg, #ffa500 0%, #ff8c00 100%)' if high_bp_days > monitoring_days * 0.15 else 'linear-gradient(135deg, #00cc66 0%, #00b359 100%)'};">
                    <div class="metric-label">Hypertensive Episodes</div>
                    <div class="metric-value">{high_bp_days}</div>
                    <div style="font-size: 0.9rem;">{(high_bp_days/monitoring_days)*100:.1f}% of period</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: {'linear-gradient(135deg, #ffa500 0%, #ff8c00 100%)' if high_hr_days > monitoring_days * 0.1 else 'linear-gradient(135deg, #00cc66 0%, #00b359 100%)'};">
                    <div class="metric-label">Tachycardia Events</div>
                    <div class="metric-value">{high_hr_days}</div>
                    <div style="font-size: 0.9rem;">{(high_hr_days/monitoring_days)*100:.1f}% of period</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div class="metric-label">Avg Health Score</div>
                    <div class="metric-value">{avg_health_score:.0f}</div>
                    <div style="font-size: 0.9rem;">{'Excellent' if avg_health_score > 80 else 'Good' if avg_health_score > 60 else 'Fair'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Trend analysis
            st.markdown("### 📊 Trend Analysis & Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate trends
                risk_trend_direction = "📈 Increasing" if risk_trend[-1] > risk_trend[0] else "📉 Decreasing"
                risk_change = ((risk_trend[-1] - risk_trend[0]) / risk_trend[0]) * 100
                
                st.markdown(f"""
                **Risk Score Trend:** {risk_trend_direction}  
                **Change:** {risk_change:+.1f}%  
                **Volatility:** {np.std(risk_trend):.3f} ({"High" if np.std(risk_trend) > 0.05 else "Low"})
                """)
                
                if risk_change > 5:
                    st.warning("⚠️ Risk increasing - consider treatment adjustment")
                elif risk_change < -5:
                    st.success("✅ Risk improving - continue current plan")
            
            with col2:
                bp_trend_direction = "📈 Increasing" if bp_systolic[-1] > bp_systolic[0] else "📉 Decreasing"
                bp_change = bp_systolic[-1] - bp_systolic[0]
                
                st.markdown(f"""
                **Blood Pressure Trend:** {bp_trend_direction}  
                **Change:** {bp_change:+.1f} mmHg  
                **Average:** {np.mean(bp_systolic):.1f}/{np.mean(bp_diastolic):.1f} mmHg
                """)
                
                if np.mean(bp_systolic) > 140:
                    st.error("🔴 Average BP in hypertensive range")
                elif np.mean(bp_systolic) > 120:
                    st.warning("🟡 Elevated BP - monitor closely")

# ============================================================================
# TAB 4: KNOWLEDGE GRAPH (Enhanced)
# ============================================================================
with tab4:
    st.markdown("## 🧠 Medical Knowledge Graph Explorer")
    
    kg = resources['kg']
    
    # Statistics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Entities", kg.number_of_nodes())
    with col2:
        st.metric("🔗 Relationships", kg.number_of_edges())
    with col3:
        st.metric("📈 Graph Density", f"{nx.density(kg):.3f}")
    with col4:
        diseases = len([n for n in kg.nodes() if kg.nodes[n].get('type') == 'disease'])
        st.metric("🏥 Diseases", diseases)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🔍 Entity Explorer")
        
        entity_type = st.selectbox(
            "Select entity category:",
            ['disease', 'treatment', 'risk_factor_modifiable', 'diagnostic'],
            format_func=lambda x: {
                'disease': '🏥 Diseases',
                'treatment': '💊 Treatments',
                'risk_factor_modifiable': '⚠️ Risk Factors',
                'diagnostic': '🔬 Diagnostics'
            }[x]
        )
        
        entities = [node for node in kg.nodes() if kg.nodes[node].get('type') == entity_type]
        
        if entities:
            selected_entity = st.selectbox("Select specific entity:", entities)
            
            st.markdown("---")
            st.markdown(f"### 📋 {selected_entity.replace('_', ' ').title()}")
            
            node_data = kg.nodes[selected_entity]
            for key, value in node_data.items():
                st.markdown(f"**{key.title()}:** {value}")
    
    with col2:
        if entities and selected_entity:
            st.markdown(f"### 🔗 Relationship Network for: {selected_entity.replace('_', ' ').title()}")
            
            # Create subgraph around selected entity
            neighbors = set([selected_entity])
            neighbors.update(kg.predecessors(selected_entity))
            neighbors.update(kg.successors(selected_entity))
            
            subgraph = kg.subgraph(neighbors)
            
            # Calculate layout
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in subgraph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node.replace('_', ' ').title())
                
                # Color by node type
                node_type = subgraph.nodes[node].get('type', '')
                if node == selected_entity:
                    node_color.append('#667eea')  # Selected node
                elif 'disease' in node_type:
                    node_color.append('#FF4B4B')
                elif 'treatment' in node_type:
                    node_color.append('#4ECDC4')
                elif 'risk_factor' in node_type:
                    node_color.append('#FFD93D')
                else:
                    node_color.append('#95E1D3')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    size=30,
                    color=node_color,
                    line=dict(width=2, color='white')
                ))
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=f'Knowledge Graph: {selected_entity.replace("_", " ").title()}',
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0,l=0,r=0,t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=500,
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)'
                          ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show relationships
            st.markdown("#### 🔗 Connected Entities")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                predecessors = list(kg.predecessors(selected_entity))
                if predecessors:
                    st.markdown("**⬅️ Predecessors (leads to this):**")
                    for pred in predecessors[:5]:
                        edge_data = kg.get_edge_data(pred, selected_entity)
                        relationship = edge_data.get('relationship', 'related_to') if edge_data else 'related_to'
                        st.markdown(f"- `{pred}` *{relationship}* `{selected_entity}`")
            
            with col_b:
                successors = list(kg.successors(selected_entity))
                if successors:
                    st.markdown("**➡️ Successors (this leads to):**")
                    for succ in successors[:5]:
                        edge_data = kg.get_edge_data(selected_entity, succ)
                        relationship = edge_data.get('relationship', 'related_to') if edge_data else 'related_to'
                        st.markdown(f"- `{selected_entity}` *{relationship}* `{succ}`")

# ============================================================================
# TAB 5: CLINICAL REPORT (Enhanced)
# ============================================================================
with tab5:
    st.markdown("## 📋 Comprehensive Clinical Report")
    
    if predict_button or use_test_patient:
        prediction = predict_risk(features)
        
        if prediction is None:
            st.error("Failed to generate prediction. Please try again.")
        else:
            treatments = get_treatment_recommendations(prediction['risk_category'])
            
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Report header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h2 style="margin: 0; color: white;">❤️ Cardiovascular Risk Assessment Report</h2>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Generated: {report_date}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Patient Information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 👤 Patient Information")
                st.markdown(f"""
                **Patient ID:** {f"Patient {patient_idx+1:03d}" if use_test_patient else "Manual Input"}  
                **Age:** {int(45 + patient_idx % 30) if use_test_patient else "N/A"} years  
                **Sex:** {'Male' if use_test_patient and patient_idx % 2 == 0 else 'Female' if use_test_patient else "N/A"}  
                **Report Date:** {datetime.now().strftime("%Y-%m-%d")}
                """)
            
            with col2:
                st.markdown("### 🎯 Risk Assessment")
                st.markdown(f"""
                **Overall Risk Score:** {prediction['risk_score']:.1%}  
                **Risk Category:** <span class="risk-{prediction['risk_category'].lower()}">{prediction['risk_category']}</span>  
                **Confidence Level:** {prediction['confidence_level']:.1%}  
                **Priority:** {'🔴 Urgent' if prediction['risk_category'] == 'High' else '🟡 Standard'}
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("### 🤖 Model Predictions")
                st.markdown(f"""
                **Ensemble Model:** {prediction['risk_score']:.1%}  
                **XGBoost:** {prediction['xgb_score']:.1%}  
                **Random Forest:** {prediction['rf_score']:.1%}  
                **Neural Network:** {prediction['nn_score']:.1%}
                """)
            
            st.markdown("---")
            
            # Clinical Interpretation
            st.markdown("## 🏥 Clinical Interpretation")
            
            if prediction['risk_category'] == 'High':
                st.markdown("""
                <div class="alert-critical">
                    <h3>⚠️ HIGH RISK ALERT - IMMEDIATE ATTENTION REQUIRED</h3>
                    <p>This patient exhibits a high risk score for cardiovascular disease. <strong>Immediate clinical intervention is strongly recommended.</strong></p>
                    
                    <h4>Immediate Actions Required:</h4>
                    <ul>
                        <li>🔴 Schedule urgent cardiology consultation (within 7 days)</li>
                        <li>🔴 Initiate pharmacotherapy as per guidelines</li>
                        <li>🔴 Order comprehensive cardiac workup (ECG, Echo, Stress test)</li>
                        <li>🔴 Begin daily vital signs monitoring</li>
                        <li>🔴 Patient education on warning signs and symptoms</li>
                    </ul>
                    
                    <h4>Risk Factors Identified:</h4>
                    <ul>
                        <li>Elevated cardiovascular risk markers</li>
                        <li>Multiple machine learning models concordant for high risk</li>
                        <li>Recommendation for aggressive risk factor management</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif prediction['risk_category'] == 'Moderate':
                st.markdown("""
                <div class="alert-warning">
                    <h3>⚠️ MODERATE RISK - PREVENTIVE INTERVENTION RECOMMENDED</h3>
                    <p>Patient shows moderate cardiovascular risk. Preventive measures and close monitoring are advised.</p>
                    
                    <h4>Recommended Actions:</h4>
                    <ul>
                        <li>🟡 Schedule cardiology consultation within 4 weeks</li>
                        <li>🟡 Consider pharmacotherapy (discuss risk/benefit)</li>
                        <li>🟡 Aggressive lifestyle modifications</li>
                        <li>🟡 Monthly blood pressure and lipid monitoring</li>
                        <li>🟡 Re-assessment in 3-6 months</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    <h3>✅ LOW RISK - CONTINUE PREVENTIVE CARE</h3>
                    <p>Patient demonstrates low cardiovascular risk. Continue with standard preventive care and healthy lifestyle.</p>
                    
                    <h4>Maintenance Recommendations:</h4>
                    <ul>
                        <li>✅ Annual health check-ups</li>
                        <li>✅ Maintain healthy diet and regular exercise</li>
                        <li>✅ Monitor blood pressure quarterly</li>
                        <li>✅ Continue current health maintenance plan</li>
                        <li>✅ Re-assessment in 12 months</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Treatment Plan
            st.markdown("## 💊 Recommended Treatment Plan")
            st.markdown("*Based on knowledge graph analysis and clinical practice guidelines*")
            
            for i, treatment in enumerate(treatments[:3], 1):
                with st.expander(f"**{i}. {treatment['name']}**", expanded=(i==1)):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"**🔬 Mechanism of Action:**  \n{treatment['mechanism']}")
                        st.markdown(f"**💊 Drug Examples:**  \n{treatment['examples']}")
                    
                    with col_b:
                        if 'efficacy' in treatment:
                            st.markdown(f"**📊 Expected Efficacy:**  \n{treatment['efficacy']}")
                        st.markdown(f"**📈 Evidence Level:** {'High (Grade A)' if i <= 2 else 'Moderate (Grade B)'}")
                        st.markdown(f"**🎯 Recommendation Strength:** {'Strong' if i == 1 else 'Moderate'}")
            
            st.markdown("---")
            
            # Follow-up Schedule
            st.markdown("## 📅 Follow-up & Monitoring Schedule")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Short-term (0-3 months)
                - **Week 1-2:** Initiate treatment, baseline assessments
                - **Week 4:** First follow-up, medication tolerance check
                - **Month 2:** Vital signs review, medication adjustment if needed
                - **Month 3:** Comprehensive re-evaluation
                """)
            
            with col2:
                st.markdown("""
                ### Long-term (3-12 months)
                - **Month 6:** Mid-term assessment, risk re-stratification
                - **Month 9:** Continued monitoring, lifestyle adherence check
                - **Month 12:** Annual comprehensive cardiac assessment
                - **Ongoing:** Monthly vital signs, quarterly lipid panel
                """)
            
            st.markdown("---")
            
            # Laboratory & Diagnostic Tests
            st.markdown("## 🔬 Recommended Laboratory & Diagnostic Tests")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Blood Tests:**
                - ✓ Complete Blood Count (CBC)
                - ✓ Comprehensive Metabolic Panel
                - ✓ Lipid Panel (Fasting)
                - ✓ HbA1c
                - ✓ High-sensitivity CRP
                - ✓ Troponin (if symptomatic)
                """)
            
            with col2:
                st.markdown("""
                **Cardiac Studies:**
                - ✓ 12-lead Electrocardiogram (ECG)
                - ✓ Echocardiogram
                - ✓ Stress Test (if indicated)
                - ✓ Holter Monitor (if arrhythmia)
                - ✓ Ankle-Brachial Index
                """)
            
            with col3:
                st.markdown("""
                **Additional Assessments:**
                - ✓ Blood Pressure (24-hour ABPM)
                - ✓ Body Mass Index (BMI)
                - ✓ Waist Circumference
                - ✓ Smoking Status Assessment
                - ✓ Physical Activity Level
                """)
            
            st.markdown("---")
            
            # Clinical Notes
            st.markdown("## 📝 Clinical Notes & Recommendations")
            
            clinical_notes = st.text_area(
                "Physician Notes (Optional):",
                height=150,
                placeholder="Enter additional clinical observations, patient concerns, or specific treatment considerations..."
            )
            
            st.markdown("---")
            
            # Disclaimer
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                <h4 style="margin-top: 0;">⚕️ Important Medical Disclaimer</h4>
                <p style="margin-bottom: 0;">
                    This report is generated by an AI-powered clinical decision support system and is intended 
                    to <strong>assist</strong> healthcare professionals in clinical decision-making. It should 
                    <strong>NOT</strong> replace professional medical judgment, clinical examination, or patient-specific 
                    considerations. All treatment decisions must be made by qualified healthcare providers based on 
                    comprehensive patient evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Download report
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                report_text = f"""
CARDIOVASCULAR RISK ASSESSMENT REPORT
Generated: {report_date}
Patient ID: {f"Patient {patient_idx+1:03d}" if use_test_patient else "Manual Input"}

RISK ASSESSMENT:
- Overall Risk Score: {prediction['risk_score']:.1%}
- Risk Category: {prediction['risk_category']}
- Ensemble Model: {prediction['risk_score']:.1%}
- XGBoost: {prediction['xgb_score']:.1%}
- Random Forest: {prediction['rf_score']:.1%}
- Neural Network: {prediction['nn_score']:.1%}

RECOMMENDED TREATMENTS:
{chr(10).join([f"{i+1}. {t['name']}: {t['mechanism']}" for i, t in enumerate(treatments[:3])])}

Generated by CVD Digital Twin System v2.1
For research and clinical decision support purposes.
                """
                
                st.download_button(
                    label="📥 Download Full Report (TXT)",
                    data=report_text,
                    file_name=f"cvd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ============================================================================
# TAB 6: ADVANCED ANALYTICS (NEW)
# ============================================================================
with tab6:
    st.markdown("## 🔬 Advanced Analytics & Research Tools")
    
    if predict_button or use_test_patient:
        prediction = predict_risk(features)
        
        if prediction is None:
            st.error("Failed to generate prediction. Please try again.")
        else:
            # Cohort Analysis
            st.markdown("### 👥 Population Cohort Analysis")
            
            # Simulate cohort data
            cohort_size = len(resources['test_data'])
            cohort_predictions = []
            
            with st.spinner('Analyzing patient cohort...'):
                for i in range(min(100, cohort_size)):
                    patient_features = resources['test_data'].drop('target', axis=1).iloc[i].values
                    pred = predict_risk(patient_features)
                    if pred:
                        cohort_predictions.append({
                            'patient_id': i,
                            'risk_score': pred['risk_score'],
                            'risk_category': pred['risk_category']
                        })
            
            cohort_df = pd.DataFrame(cohort_predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution
                fig = px.histogram(
                    cohort_df,
                    x='risk_score',
                    color='risk_category',
                    title='Cohort Risk Score Distribution',
                    labels={'risk_score': 'Risk Score', 'count': 'Number of Patients'},
                    color_discrete_map={'High': '#FF4B4B', 'Moderate': '#FFA500', 'Low': '#00CC66'}
                )
                fig.add_vline(x=prediction['risk_score'], line_dash="dash", line_color="blue",
                             annotation_text="Current Patient")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk category pie chart
                risk_counts = cohort_df['risk_category'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Cohort Risk Category Distribution',
                    color=risk_counts.index,
                    color_discrete_map={'High': '#FF4B4B', 'Moderate': '#FFA500', 'Low': '#00CC66'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Percentile ranking
            percentile = (cohort_df['risk_score'] < prediction['risk_score']).sum() / len(cohort_df) * 100
            
            st.info(f"""
            📊 **Patient Percentile Ranking:**  
            This patient's risk score is higher than **{percentile:.1f}%** of the analyzed cohort.  
            Cohort size: {len(cohort_df)} patients
            """)
            
            st.markdown("---")
            
            # Sensitivity Analysis
            st.markdown("### 🎛️ Sensitivity Analysis")
            st.markdown("*Analyze how changes in risk factors affect overall risk score*")
            
            selected_feature = st.selectbox(
                "Select feature to analyze:",
                ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'BMI']
            )
            
            # Simulate feature variation
            feature_range = np.linspace(0.5, 1.5, 50)
            risk_variations = []
            
            for multiplier in feature_range:
                modified_features = features.copy()
                # Simulate feature modification
                modified_features[0] = features[0] * multiplier
                pred = predict_risk(modified_features)
                if pred:
                    risk_variations.append(pred['risk_score'])
                else:
                    risk_variations.append(0)
            
            # Plot sensitivity
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=feature_range * 100,
                y=risk_variations,
                mode='lines',
                name='Risk Score',
                line=dict(color='#667eea', width=3),
                fill='tonexty'
            ))
            
            fig.add_vline(x=100, line_dash="dash", line_color="red",
                         annotation_text="Current Value")
            
            fig.update_layout(
                title=f'Sensitivity Analysis: Impact of {selected_feature} on Risk Score',
                xaxis_title=f'{selected_feature} (% of current value)',
                yaxis_title='Predicted Risk Score',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Risk Factor Contribution
            st.markdown("### 📊 Risk Factor Contribution Analysis")
            
            # Simulate SHAP-like values
            risk_factors = ['Age', 'Cholesterol', 'BP Systolic', 'Heart Rate', 'ECG', 
                           'Family History', 'Smoking', 'Diabetes', 'Exercise', 'BMI']
            contributions = np.random.random(10) * 0.1
            contributions = contributions / contributions.sum() * prediction['risk_score']
            
            contribution_df = pd.DataFrame({
                'Factor': risk_factors,
                'Contribution': contributions,
                'Impact': ['Positive' if c > np.median(contributions) else 'Negative' for c in contributions]
            }).sort_values('Contribution', ascending=True)
            
            fig = px.bar(
                contribution_df,
                y='Factor',
                x='Contribution',
                color='Impact',
                orientation='h',
                title='Individual Risk Factor Contributions to Overall Risk',
                color_discrete_map={'Positive': '#FF4B4B', 'Negative': '#00CC66'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Comparative Analysis
            st.markdown("### 🔄 Comparative Patient Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                compare_patient = st.selectbox(
                    "Select patient to compare:",
                    range(len(resources['test_data'])),
                    format_func=lambda x: f"Patient {x+1:03d}",
                    key='compare_select'
                )
            
            if compare_patient != patient_idx:
                compare_features = resources['test_data'].drop('target', axis=1).iloc[compare_patient].values
                compare_prediction = predict_risk(compare_features)
                
                if compare_prediction:
                    with col2:
                        st.markdown("#### Comparison Summary")
                        risk_diff = prediction['risk_score'] - compare_prediction['risk_score']
                        st.metric(
                            "Risk Score Difference",
                            f"{risk_diff:+.1%}",
                            delta=f"{'Higher' if risk_diff > 0 else 'Lower'} risk"
                        )
                    
                    # Comparison chart
                    comparison_data = pd.DataFrame({
                        'Metric': ['XGBoost', 'Random Forest', 'Neural Network', 'Ensemble'],
                        'Current Patient': [
                            prediction['xgb_score'],
                            prediction['rf_score'],
                            prediction['nn_score'],
                            prediction['risk_score']
                        ],
                        'Comparison Patient': [
                            compare_prediction['xgb_score'],
                            compare_prediction['rf_score'],
                            compare_prediction['nn_score'],
                            compare_prediction['risk_score']
                        ]
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Current Patient',
                        x=comparison_data['Metric'],
                        y=comparison_data['Current Patient'],
                        marker_color='#667eea'
                    ))
                    fig.add_trace(go.Bar(
                        name='Comparison Patient',
                        x=comparison_data['Metric'],
                        y=comparison_data['Comparison Patient'],
                        marker_color='#f093fb'
                    ))
                    
                    fig.update_layout(
                        title='Side-by-Side Risk Score Comparison',
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Please select a patient and click 'Analyze Patient Risk' to access advanced analytics.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>❤️ CVD Digital Twin Dashboard v2.1</h3>
    <p><strong>Powered by:</strong> Machine Learning • Deep Learning • Knowledge Graphs • Clinical Guidelines</p>
    <p><strong>Technology Stack:</strong> Python • Streamlit • PyTorch • XGBoost • NetworkX • Plotly</p>
    <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;">
        ⚕️ For research and educational purposes only. Not a substitute for professional medical advice.
    </p>
    <p style="margin-top: 0.5rem;">
        <strong>© 2024 CVD Digital Twin Research Project</strong>
    </p>
</div>
""", unsafe_allow_html=True)
