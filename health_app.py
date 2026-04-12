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

@st.cache_data
def load_and_clean_data():
    base_dir = r"C:\Users\nalla\OneDrive\Desktop\New folder"
    required = {'p': 'patients.csv', 'o': 'observations.csv', 'c': 'conditions.csv'}
    dfs = {}

    for key, name in required.items():
        path = os.path.join(base_dir, name) if os.path.exists(os.path.join(base_dir, name)) else name
        if not os.path.exists(path):
            st.error(f"❌ Missing {name}")
            st.stop()
        dfs[key] = pd.read_csv(path)

    p, o, c = dfs['p'], dfs['o'], dfs['c']
    
    # Standardize Column Names
    for df in [p, o, c]:
        df.columns = df.columns.str.upper()

    # Pre-process Patients
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})
    
    return p, o, c

@st.cache_resource
def train_models(df_p, df_o, df_c):
    # FUZZY SEARCH: Matches names like "Creatinine [Mass/Vol]..."
    markers = ['Creatinine', 'Urea Nitrogen', 'Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(markers), case=False, na=False)]
    
    # Pivot
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    # Merge
    df_ml = pd.merge(v_pivot, df_p[['ID', 'AGE', 'GENDER_NUM', 'INCOME', 'FULL_NAME']], left_on='PATIENT', right_on='ID')
    
    # Target: Look for ANY kidney related disease
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False, na=False)]['PATIENT'].unique()
    df_ml['TARGET'] = df_ml['ID'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    df_final = df_ml.dropna()

    if len(df_final) < 5:
        st.error(f"⛔ Data Error: Found only {len(df_final)} matching rows. Your CSVs might use different column names.")
        st.write("Available Observations:", df_o['DESCRIPTION'].unique()[:10])
        st.stop()

    X = df_final.drop(['PATIENT', 'ID', 'TARGET', 'FULL_NAME'], axis=1)
    y = df_final['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train, y_train),
        "Lasso (Logistic)": LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(X_train, y_train),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train, y_train)
    }
    
    accs = {name: accuracy_score(y_test, m.predict(X_test)) for name, m in models.items()}
    return models, X.columns.tolist(), df_p, accs

# --- APP EXECUTION ---
try:
    p_raw, o_raw, c_raw = load_and_clean_data()
    models, features, df_p, accuracies = train_models(p_raw, o_raw, c_raw)

    st.sidebar.title("🔬 Research Dashboard")
    sel_model = st.sidebar.selectbox("Select Model", list(models.keys()))
    st.sidebar.write(f"Accuracy: {accuracies[sel_model]:.2%}")
    
    sel_name = st.sidebar.selectbox("Patient Selection", options=df_p['FULL_NAME'].sort_values())
    patient = df_p[df_p['FULL_NAME'] == sel_name].iloc[0]
    
    st.title(f"Predictive Analysis: {sel_name}")
    
    # Prediction logic
    user_obs = o_raw[o_raw['PATIENT'] == patient['ID']]
    def get_latest(d):
        val = user_obs[user_obs['DESCRIPTION'].str.contains(d, case=False)]
        return val.iloc[-1]['VALUE'] if not val.empty else 0.0

    # Build input matching the exact features used in training
    input_data = []
    for col in features:
        # Match features back to observations
        val = user_obs[user_obs['DESCRIPTION'] == col]
        input_data.append(val.iloc[-1]['VALUE'] if not val.empty else 0.0)

    risk = models[sel_model].predict_proba([input_data])[0][1]
    
    st.metric(f"Chronic Kidney Disease Risk ({sel_model})", f"{risk*100:.1f}%")
    st.progress(risk)

except Exception as e:
    st.error(f"Critical Failure: {e}")
