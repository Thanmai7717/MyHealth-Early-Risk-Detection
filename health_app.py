import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="MyHealth AI | SLU Research", page_icon="🧬", layout="wide")

# Research Aesthetics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; }
    .main-header { color: #4C1D95; font-size: 30px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Your specific project path
    base_path = r"C:/Users/Desktop/new folder"
    
    # List of files we need
    required_files = {
        'p': 'patients.csv',
        'o': 'observations.csv',
        'c': 'conditions.csv'
    }
    
    loaded_dfs = {}
    
    for key, name in required_files.items():
        # Check current folder first, then your desktop folder
        if os.path.exists(name):
            path = name
        elif os.path.exists(os.path.join(base_path, name)):
            path = os.path.join(base_path, name)
        else:
            st.error(f"❌ Could not find {name}")
            st.info(f"Please check: {os.path.join(base_path, name)}")
            st.stop()
            
        loaded_dfs[key] = pd.read_csv(path)

    p, o, c = loaded_dfs['p'], loaded_dfs['o'], loaded_dfs['c']
    
    # Data Prep matching your model.ipynb logic
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})
    
    return p, o, c

@st.cache_resource
def train_rf_model(df_p, df_o, df_c):
    # Features from your research markers
    markers = ['Creatinine', 'Urea Nitrogen', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_ml = pd.merge(v_pivot, df_p[['Id', 'AGE', 'GENDER_NUM', 'INCOME']], left_on='PATIENT', right_on='Id')
    
    # Target: Kidney Disease
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney', case=False, na=False)]['PATIENT'].unique()
    df_ml['target'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'Id', 'target'], axis=1)
    y = df_final['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest params from your successful test
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, X.columns.tolist(), acc

# --- APP FLOW ---
try:
    df_p, df_o, df_c = load_data()
    model, features, accuracy = train_rf_model(df_p, df_o, df_c)

    # Sidebar
    st.sidebar.markdown("<h2 style='color: #6D28D9;'>🔬 Project MyHealth</h2>", unsafe_allow_html=True)
    st.sidebar.info(f"Random Forest Accuracy: {accuracy:.1%}")
    
    selected_name = st.sidebar.selectbox("Patient Selection", options=df_p['FULL_NAME'].sort_values())
    patient = df_p[df_p['FULL_NAME'] == selected_name].iloc[0]
    p_id = patient['Id']

    tab1, tab2 = st.tabs(["📊 Patient Stats", "🧠 Prediction Engine"])

    # Helper for vitals
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

    with tab2:
        st.markdown("<div class='main-header'>Random Forest Analysis</div>", unsafe_allow_html=True)
        
        # Build input
        inputs = {
            'AGE': patient['AGE'],
            'GENDER_NUM': patient['GENDER_NUM'],
            'INCOME': patient['INCOME'],
            'Blood Pressure Systolic': get_v("Systolic"),
            'Body Mass Index': get_v("Body Mass Index"),
            'Creatinine': get_v("Creatinine"),
            'Glucose': get_v("Glucose"),
            'Urea Nitrogen': get_v("Urea Nitrogen")
        }
        
        input_df = pd.DataFrame([inputs])[features]
        risk = model.predict_proba(input_df)[0][1]

        st.metric("Probability of Disease Onset", f"{risk*100:.1f}%")
        st.progress(risk)

except Exception as e:
    st.error(f"Error: {e}")
