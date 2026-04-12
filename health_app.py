import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="MyHealth AI | Healthcare Dashboard", page_icon="🧬", layout="wide")

# Research Aesthetics (Purple Theme)
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .main-header { color: #4C1D95; font-size: 32px; font-weight: 800; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_research_data():
    """Ultra-stable data loader that checks local and absolute paths."""
    required = {'p': 'patients.csv', 'o': 'observations.csv', 'c': 'conditions.csv'}
    base_dir = r"C:\Users\nalla\OneDrive\Desktop\New folder"
    loaded = {}

    for key, filename in required.items():
        if os.path.exists(filename):
            path = filename
        elif os.path.exists(os.path.join(base_dir, filename)):
            path = os.path.join(base_dir, filename)
        else:
            st.error(f"❌ MISSING FILE: {filename}")
            st.stop()
        loaded[key] = pd.read_csv(path)
    
    p, o, c = loaded['p'], loaded['o'], loaded['c']
    
    # Pre-processing
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})
    return p, o, c

@st.cache_resource
def train_research_models(df_p, df_o, df_c):
    """Trains both Random Forest and XGBoost for comparison."""
    markers = ['Creatinine', 'Urea Nitrogen', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_ml = pd.merge(v_pivot, df_p[['Id', 'AGE', 'GENDER_NUM', 'INCOME']], left_on='PATIENT', right_on='Id')
    
    # Targeting Chronic Kidney Disease labels
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Chronic Kidney Disease', case=False, na=False)]['PATIENT'].unique()
    df_ml['target'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'Id', 'target'], axis=1)
    y = df_final['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Random Forest (79%)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    # Model 2: XGBoost (94.8% per your research)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    return xgb, X.columns.tolist(), acc_xgb

# --- DASHBOARD START ---
try:
    df_p, df_o, df_c = load_research_data()
    model, features, accuracy = train_research_models(df_p, df_o, df_c)

    st.sidebar.markdown("<h2 style='color: #6D28D9;'>🔬 Project MyHealth</h2>", unsafe_allow_html=True)
    st.sidebar.info(f"Primary Model: XGBoost\nAccuracy: {accuracy:.2%}")
    st.sidebar.divider()
    
    selected_name = st.sidebar.selectbox("Patient Selection", options=df_p['FULL_NAME'].sort_values())
    patient = df_p[df_p['FULL_NAME'] == selected_name].iloc[0]
    p_id = patient['Id']

    tab1, tab2 = st.tabs(["📊 Patient Profile", "🧠 Chronic Kidney Disease Prediction"])

    user_obs = df_o[df_o['PATIENT'] == p_id]
    def get_v(d):
        val = user_obs[user_obs['DESCRIPTION'].str.contains(d, case=False)]
        return val.iloc[-1]['VALUE'] if not val.empty else 0.0

    with tab1:
        st.markdown(f"<div class='main-header'>Health Summary: {selected_name}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Age", int(patient['AGE']))
        c2.metric("Gender", patient['GENDER'])
        c3.metric("Creatinine", f"{get_v('Creatinine')} mg/dL")

        st.divider()
        st.subheader("Medical History")
        history = df_c[df_c['PATIENT'] == p_id]['DESCRIPTION'].unique()
        for item in history:
            st.write(f"🚩 {item}")

    with tab2:
        st.markdown("<div class='main-header'>Early Risk Detection Engine</div>", unsafe_allow_html=True)
        
        inputs = {
            'AGE': patient['AGE'], 'GENDER_NUM': patient['GENDER_NUM'], 'INCOME': patient['INCOME'],
            'Blood Pressure Systolic': get_v("Systolic"), 'Body Mass Index': get_v("Body Mass Index"),
            'Creatinine': get_v("Creatinine"), 'Glucose': get_v("Glucose"), 'Urea Nitrogen': get_v("Urea Nitrogen")
        }
        
        input_df = pd.DataFrame([inputs])[features]
        risk = model.predict_proba(input_df)[0][1]

        st.metric("Disease Onset Probability", f"{risk*100:.1f}%")
        st.progress(risk)
        
        if risk > 0.75:
            st.error("HIGH RISK: Markers suggest advanced renal decline. Recommend nephrologist consultation.")
        elif risk > 0.4:
            st.warning("ELEVATED: Continued clinical monitoring of metabolic markers required.")
        else:
            st.success("STABLE: Clinical markers remain within normal research variance.")

except Exception as e:
    st.error(f"Error Initializing Dashboard: {e}")
