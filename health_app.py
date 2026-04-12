import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="MyHealth AI | Research Dashboard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; }
    .main-header { color: #4C1D95; font-size: 32px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_standardize(file_input):
    df = pd.read_csv(file_input)
    df.columns = df.columns.str.upper()
    return df

# --- STEP 1: DATA ACQUISITION ---
st.sidebar.title("📁 Data Connection")
files_needed = {'p': 'patients.csv', 'o': 'observations.csv', 'c': 'conditions.csv'}
dfs = {}

# Try to find files automatically
base_dir = r"C:\Users\nalla\OneDrive\Desktop\New folder"
for key, name in files_needed.items():
    path = os.path.join(base_dir, name) if os.path.exists(os.path.join(base_dir, name)) else name
    if os.path.exists(path):
        dfs[key] = load_and_standardize(path)
    else:
        # MANUAL FALLBACK: If path fails, show uploader
        uploaded = st.sidebar.file_uploader(f"Select {name}", type="csv")
        if uploaded:
            dfs[key] = load_and_standardize(uploaded)
        else:
            st.error(f"❌ Missing {name}")
            st.stop()

# --- STEP 2: PROCESSING & TRAINING ---
@st.cache_resource
def process_and_train(p, o, c):
    # Pre-process Patients
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})

    # Clinical Markers for Chronic Kidney Disease
    markers = ['Creatinine', 'Urea Nitrogen', 'Systolic', 'Body Mass Index', 'Glucose']
    v_df = o[o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    # Merge
    df_ml = pd.merge(v_pivot, p[['ID', 'AGE', 'GENDER_NUM', 'INCOME', 'FULL_NAME']], left_on='PATIENT', right_on='ID')
    
    # Target labeling
    ckd_ids = c[c['DESCRIPTION'].str.contains('Kidney|Renal', case=False, na=False)]['PATIENT'].unique()
    df_ml['TARGET'] = df_ml['ID'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'ID', 'TARGET', 'FULL_NAME'], axis=1)
    y = df_final['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train, y_train),
        "Lasso (Logistic)": LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(X_train, y_train),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train, y_train)
    }
    
    accs = {name: accuracy_score(y_test, m.predict(X_test)) for name, m in models.items()}
    return models, X.columns.tolist(), p, accs

# --- STEP 3: EXECUTION ---
try:
    models, features, patients, accs = process_and_train(dfs['p'], dfs['o'], dfs['c'])

    st.sidebar.divider()
    sel_model = st.sidebar.selectbox("Active AI Model", list(models.keys()))
    st.sidebar.success(f"Model Accuracy: {accs[sel_model]:.2%}")
    
    sel_name = st.sidebar.selectbox("Select Patient", options=patients['FULL_NAME'].sort_values())
    patient = patients[patients['FULL_NAME'] == sel_name].iloc[0]

    st.markdown(f"<div class='main-header'>MyHealth: {sel_name} Analysis</div>", unsafe_allow_html=True)
    
    # Prediction logic
    user_obs = dfs['o'][dfs['o']['PATIENT'] == patient['ID']]
    input_vals = []
    for f in features:
        val = user_obs[user_obs['DESCRIPTION'] == f]
        input_vals.append(val.iloc[-1]['VALUE'] if not val.empty else 0.0)

    risk = models[sel_model].predict_proba([input_vals])[0][1]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(f"{sel_model} Risk Score", f"{risk*100:.1f}%")
        if risk > 0.7: st.error("High Risk Detected")
        else: st.success("Patient Stable")
    with col2:
        st.progress(risk)

except Exception as e:
    st.error(f"Waiting for data... {e}")
