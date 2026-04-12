import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="MyHealth AI | Research Dashboard", layout="wide")

# --- DATA LOADING WITH MANUAL FALLBACK ---
def get_data():
    base_dir = r"C:\Users\nalla\OneDrive\Desktop\New folder"
    required = {'p': 'patients.csv', 'o': 'observations.csv', 'c': 'conditions.csv'}
    dfs = {}

    try:
        for key, name in required.items():
            # Check local folder first
            path = os.path.join(base_dir, name) if os.path.exists(os.path.join(base_dir, name)) else name
            if os.path.exists(path):
                dfs[key] = pd.read_csv(path)
            else:
                raise FileNotFoundError
        return dfs['p'], dfs['o'], dfs['c']
    except:
        st.warning("⚠️ Files not found on path. Please upload them manually below to start the dashboard.")
        up_p = st.file_uploader("Upload patients.csv", type="csv")
        up_o = st.file_uploader("Upload observations.csv", type="csv")
        up_c = st.file_uploader("Upload conditions.csv", type="csv")
        
        if up_p and up_o and up_c:
            return pd.read_csv(up_p), pd.read_csv(up_o), pd.read_csv(up_c)
        else:
            st.stop()

# --- MODEL TRAINING ---
@st.cache_resource
def train_models(df_p, df_o, df_c):
    markers = ['Creatinine', 'Urea Nitrogen', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_p['BIRTHDATE'] = pd.to_datetime(df_p['BIRTHDATE'])
    df_p['AGE'] = 2026 - df_p['BIRTHDATE'].dt.year
    df_p['FULL_NAME'] = df_p['FIRST'].fillna('') + " " + df_p['LAST'].fillna('')
    df_p['GENDER_NUM'] = df_p['GENDER'].map({'M': 0, 'F': 1})

    df_ml = pd.merge(v_pivot, df_p[['Id', 'AGE', 'GENDER_NUM', 'INCOME', 'FULL_NAME']], left_on='PATIENT', right_on='Id')
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Chronic Kidney Disease', case=False, na=False)]['PATIENT'].unique()
    df_ml['target'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'Id', 'target', 'FULL_NAME'], axis=1)
    y = df_final['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. RF, 2. Lasso, 3. XGB
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train, y_train)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(X_train, y_train)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train, y_train)
    
    models = {"Random Forest": rf, "Lasso Regression": lasso, "XGBoost": xgb}
    return models, X.columns.tolist(), df_p

# --- APP START ---
try:
    p_raw, o_raw, c_raw = get_data()
    models, features, df_p = train_models(p_raw, o_raw, c_raw)

    st.sidebar.title("🔬 Research Controls")
    sel_model = st.sidebar.selectbox("Model", list(models.keys()))
    sel_name = st.sidebar.selectbox("Patient", options=df_p['FULL_NAME'].sort_values())
    
    patient = df_p[df_p['FULL_NAME'] == sel_name].iloc[0]
    p_id = patient['Id']

    st.title(f"Healthcare Risk Analysis: {sel_name}")
    
    # Simple risk display
    user_obs = o_raw[o_raw['PATIENT'] == p_id]
    def get_v(d):
        res = user_obs[user_obs['DESCRIPTION'].str.contains(d, case=False)]
        return res.iloc[-1]['VALUE'] if not res.empty else 0.0

    inputs = pd.DataFrame([{
        'AGE': patient['AGE'], 'GENDER_NUM': patient['GENDER_NUM'], 'INCOME': patient['INCOME'],
        'Blood Pressure Systolic': get_v("Systolic"), 'Body Mass Index': get_v("Body Mass Index"),
        'Creatinine': get_v("Creatinine"), 'Glucose': get_v("Glucose"), 'Urea Nitrogen': get_v("Urea Nitrogen")
    }])[features]
    
    risk = models[sel_model].predict_proba(inputs)[0][1]
    st.metric(f"{sel_model} Risk Score", f"{risk*100:.1f}%")
    st.progress(risk)

except Exception as e:
    st.error(f"Error: {e}")
