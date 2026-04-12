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
st.set_page_config(page_title="MyHealth AI | Multi-Model Research", page_icon="🧬", layout="wide")

# Research Aesthetics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; }
    .main-header { color: #4C1D95; font-size: 32px; font-weight: 800; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_research_data():
    """Loads datasets from your specific OneDrive folder."""
    base_dir = r"C:\Users\nalla\OneDrive\Desktop\New folder"
    required = {'p': 'patients.csv', 'o': 'observations.csv', 'c': 'conditions.csv'}
    loaded = {}

    for key, filename in required.items():
        path = os.path.join(base_dir, filename) if os.path.exists(os.path.join(base_dir, filename)) else filename
        if not os.path.exists(path):
            st.error(f"❌ Cannot find {filename} at {path}")
            st.stop()
        loaded[key] = pd.read_csv(path)
    
    p, o, c = loaded['p'], loaded['o'], loaded['c']
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})
    return p, o, c

@st.cache_resource
def train_all_models(df_p, df_o, df_c):
    """Trains Random Forest, Lasso (Logistic), and XGBoost."""
    markers = ['Creatinine', 'Urea Nitrogen', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_ml = pd.merge(v_pivot, df_p[['Id', 'AGE', 'GENDER_NUM', 'INCOME']], left_on='PATIENT', right_on='Id')
    
    # Target: Chronic Kidney Disease
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Chronic Kidney Disease', case=False, na=False)]['PATIENT'].unique()
    df_ml['target'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'Id', 'target'], axis=1)
    y = df_final['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Random Forest (79% baseline)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    # 2. Lasso (Logistic Regression with L1 penalty - 92.20%)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    lasso.fit(X_train, y_train)
    
    # 3. XGBoost (94.80%)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    
    results = {
        "Random Forest": (rf, accuracy_score(y_test, rf.predict(X_test))),
        "Lasso Regression": (lasso, accuracy_score(y_test, lasso.predict(X_test))),
        "XGBoost": (xgb, accuracy_score(y_test, xgb.predict(X_test)))
    }
    return results, X.columns.tolist()

# --- APP START ---
try:
    df_p, df_o, df_c = load_research_data()
    models_dict, features = train_all_models(df_p, df_o, df_c)

    # SIDEBAR
    st.sidebar.markdown("<h2 style='color: #6D28D9;'>🔬 Model Selection</h2>", unsafe_allow_html=True)
    selected_model_name = st.sidebar.selectbox("Choose AI Model", list(models_dict.keys()))
    active_model, acc = models_dict[selected_model_name]
    st.sidebar.info(f"Current Model Accuracy: {acc:.2%}")
    
    selected_name = st.sidebar.selectbox("Patient Selection", options=df_p['FULL_NAME'].sort_values())
    patient = df_p[df_p['FULL_NAME'] == selected_name].iloc[0]
    p_id = patient['Id']

    tab1, tab2 = st.tabs(["📊 Patient Dashboard", "🧠 Multi-Model Risk Analysis"])

    user_obs = df_o[df_o['PATIENT'] == p_id]
    def get_v(d):
        val = user_obs[user_obs['DESCRIPTION'].str.contains(d, case=False)]
        return val.iloc[-1]['VALUE'] if not val.empty else 0.0

    with tab1:
        st.markdown(f"<div class='main-header'>Profile: {selected_name}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Age", int(patient['AGE']))
        c2.metric("Income", f"${patient['INCOME']:,.0f}")
        c3.metric("Creatinine", f"{get_v('Creatinine')} mg/dL")

    with tab2:
        st.markdown(f"<div class='main-header'>{selected_model_name} Prediction</div>", unsafe_allow_html=True)
        
        inputs = {
            'AGE': patient['AGE'], 'GENDER_NUM': patient['GENDER_NUM'], 'INCOME': patient['INCOME'],
            'Blood Pressure Systolic': get_v("Systolic"), 'Body Mass Index': get_v("Body Mass Index"),
            'Creatinine': get_v("Creatinine"), 'Glucose': get_v("Glucose"), 'Urea Nitrogen': get_v("Urea Nitrogen")
        }
        
        input_df = pd.DataFrame([inputs])[features]
        risk = active_model.predict_proba(input_df)[0][1]

        st.metric(f"Probability of Chronic Kidney Disease", f"{risk*100:.1f}%")
        st.progress(risk)
        
        # Display comparison of all 3 models for this patient
        st.divider()
        st.subheader("Cross-Model Comparison")
        comp_cols = st.columns(3)
        for i, (name, (m, a)) in enumerate(models_dict.items()):
            p_risk = m.predict_proba(input_df)[0][1]
            comp_cols[i].metric(name, f"{p_risk*100:.1f}% risk")

except Exception as e:
    st.error(f"Error: {e}")
