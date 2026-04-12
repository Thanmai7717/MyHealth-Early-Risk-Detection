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
st.set_page_config(page_title="MyHealth AI | Research Dashboard", page_icon="🧬", layout="wide")

# Purple/Professional Research Styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; }
    .main-header { color: #4C1D95; font-size: 30px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

def find_file(filename):
    """Searches common paths for the missing files."""
    possible_paths = [
        filename, # Current folder
        os.path.join(r"C:\Users\nalla\OneDrive\Desktop\New folder", filename),
        os.path.join(os.path.expanduser("~"), "Desktop", "New folder", filename),
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "New folder", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

@st.cache_data
def load_data():
    required = {'p': 'patients.csv', 'o': 'observations.csv', 'c': 'conditions.csv'}
    dfs = {}
    
    for key, name in required.items():
        found_path = find_file(name)
        if found_path:
            dfs[key] = pd.read_csv(found_path)
        else:
            st.error(f"❌ Cannot find {name}")
            st.info("Ensure the file is in 'New folder' on your Desktop.")
            st.stop()

    p, o, c = dfs['p'], dfs['o'], dfs['c']
    
    # Cleaning
    p.columns = p.columns.str.upper()
    o.columns = o.columns.str.upper()
    c.columns = c.columns.str.upper()
    
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})
    
    return p, o, c

@st.cache_resource
def train_models(df_p, df_o, df_c):
    # Clinical markers for Kidney Disease
    markers = ['Creatinine', 'Urea Nitrogen', 'Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_ml = pd.merge(v_pivot, df_p[['ID', 'AGE', 'GENDER_NUM', 'INCOME', 'FULL_NAME']], left_on='PATIENT', right_on='ID')
    
    # Identify Kidney Disease patients
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False, na=False)]['PATIENT'].unique()
    df_ml['TARGET'] = df_ml['ID'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()
    if len(df_final) < 2:
        st.error("Not enough matching clinical data found to train models.")
        st.stop()

    X = df_final.drop(['PATIENT', 'ID', 'TARGET', 'FULL_NAME'], axis=1)
    y = df_final['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # All 3 Models
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train, y_train)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(X_train, y_train)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train, y_train)
    
    models = {"Random Forest": rf, "Lasso Regression": lasso, "XGBoost": xgb}
    accs = {name: accuracy_score(y_test, m.predict(X_test)) for name, m in models.items()}
    
    return models, X.columns.tolist(), df_p, accs

# --- APP START ---
try:
    p_raw, o_raw, c_raw = load_data()
    models, features, df_p, accs = train_models(p_raw, o_raw, c_raw)

    st.sidebar.title("🔬 Research Controls")
    sel_model = st.sidebar.selectbox("Model", list(models.keys()))
    st.sidebar.write(f"Accuracy: {accs[sel_model]:.2%}")
    
    sel_name = st.sidebar.selectbox("Patient", options=df_p['FULL_NAME'].sort_values())
    patient = df_p[df_p['FULL_NAME'] == sel_name].iloc[0]

    st.title(f"Health Risk Analysis: {sel_name}")
    
    # Prediction logic
    user_obs = o_raw[o_raw['PATIENT'] == patient['ID']]
    input_vals = []
    for f in features:
        val = user_obs[user_obs['DESCRIPTION'] == f]
        input_vals.append(val.iloc[-1]['VALUE'] if not val.empty else 0.0)

    risk = models[sel_model].predict_proba([input_vals])[0][1]
    
    st.metric(f"Risk Probability ({sel_model})", f"{risk*100:.1f}%")
    st.progress(risk)

except Exception as e:
    st.error(f"Critical Error: {e}")
