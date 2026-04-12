import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="MyHealth AI | Research Dashboard", page_icon="🧬", layout="wide")

# Purple/Professional Research Styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .main-header { color: #4C1D95; font-size: 32px; font-weight: 800; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_research_data():
    """Loads datasets from the local directory with strict error handling."""
    # Define exact expected filenames
    file_map = {
        'patients': 'patients.csv',
        'observations': 'observations.csv',
        'conditions': 'conditions.csv'
    }
    
    loaded_data = {}
    
    for key, filename in file_map.items():
        if os.path.exists(filename):
            loaded_data[key] = pd.read_csv(filename)
        else:
            # Try case-insensitive check if direct match fails
            files_in_dir = os.listdir('.')
            match = next((f for f in files_in_dir if f.lower() == filename.lower()), None)
            if match:
                loaded_data[key] = pd.read_csv(match)
            else:
                st.error(f"❌ File Not Found: {filename}")
                st.info(f"Please ensure {filename} is in the same folder as this script.")
                st.stop()

    p = loaded_data['patients']
    o = loaded_data['observations']
    c = loaded_data['conditions']
    
    # Process Demographics
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'].fillna('') + " " + p['LAST'].fillna('')
    p['GENDER_NUM'] = p['GENDER'].map({'M': 0, 'F': 1})
    
    return p, o, c

@st.cache_resource
def train_final_rf_model(df_p, df_o, df_c):
    """
    Trains the Random Forest model using the 79% accuracy 
    parameters identified in model.ipynb.
    """
    # 1. Pivot clinical markers
    kidney_markers = ['Creatinine', 'Urea Nitrogen', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(kidney_markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    # 2. Merge with Demographic columns
    df_ml = pd.merge(v_pivot, df_p[['Id', 'AGE', 'GENDER_NUM', 'INCOME']], left_on='PATIENT', right_on='Id')
    
    # 3. Target: Identify patients with kidney-related conditions
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney', case=False, na=False)]['PATIENT'].unique()
    df_ml['target'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    # 4. Training
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'Id', 'target'], axis=1)
    y = df_final['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Match research notebook: n_estimators=100, balanced weights
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, X.columns.tolist(), acc

# --- MAIN APP LOGIC ---
try:
    df_p, df_o, df_c = load_research_data()
    model, features, accuracy = train_final_rf_model(df_p, df_o, df_c)

    # SIDEBAR
    st.sidebar.markdown("<h2 style='color: #6D28D9;'>🔬 Project MyHealth</h2>", unsafe_allow_html=True)
    st.sidebar.info(f"Model: Random Forest\nAccuracy: {accuracy:.1%}")
    st.sidebar.divider()
    
    selected_name = st.sidebar.selectbox("Select Patient Profile", options=df_p['FULL_NAME'].sort_values())
    patient = df_p[df_p['FULL_NAME'] == selected_name].iloc[0]
    p_id = patient['Id']

    tab1, tab2, tab3 = st.tabs(["📊 Patient Dashboard", "🧠 AI Prediction Engine", "📂 Data Records"])

    # Utility: Get latest vital
    user_obs = df_o[df_o['PATIENT'] == p_id]
    def get_latest(desc):
        val = user_obs[user_obs['DESCRIPTION'].str.contains(desc, case=False)]
        return val.iloc[-1]['VALUE'] if not val.empty else 0.0

    # TAB 1: DASHBOARD
    with tab1:
        st.markdown(f"<div class='main-header'>Health Summary: {selected_name}</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Age", int(patient['AGE']))
        c2.metric("Gender", patient['GENDER'])
        c3.metric("Annual Income", f"${patient['INCOME']:,.0f}")
        c4.metric("Creatinine", f"{get_latest('Creatinine')} mg/dL")

        st.divider()
        st.subheader("Existing Diagnoses")
        history = df_c[df_c['PATIENT'] == p_id]['DESCRIPTION'].unique()
        if len(history) > 0:
            for item in history:
                st.write(f"🚩 {item}")
        else:
            st.write("No chronic conditions currently documented.")

    # TAB 2: PREDICTION ENGINE
    with tab2:
        st.markdown("<div class='main-header'>Random Forest Risk Analysis</div>", unsafe_allow_html=True)
        
        user_input = {
            'AGE': patient['AGE'],
            'GENDER_NUM': patient['GENDER_NUM'],
            'INCOME': patient['INCOME'],
            'Blood Pressure Systolic': get_latest("Systolic"),
            'Body Mass Index': get_latest("Body Mass Index"),
            'Creatinine': get_latest("Creatinine"),
            'Glucose': get_latest("Glucose"),
            'Urea Nitrogen': get_latest("Urea Nitrogen")
        }
        
        input_df = pd.DataFrame([user_input])[features]
        risk_score = model.predict_proba(input_df)[0][1]
        risk_pct = risk_score * 100

        st.divider()
        res_col, bar_col = st.columns([1, 2])
        with res_col:
            st.metric("Risk Probability", f"{risk_pct:.1f}%")
            if risk_pct > 70:
                st.error("HIGH RISK: Early detection marker detected.")
            elif risk_pct > 30:
                st.warning("ELEVATED: Continued monitoring advised.")
            else:
                st.success("STABLE: Risk levels within normal variance.")
        with bar_col:
            st.progress(risk_score)
            st.caption("Probability analysis based on longitudinal clinical data.")

    # TAB 3: DATA RECORDS
    with tab3:
        uploaded_json = st.file_uploader("Upload Visit Summary (JSON)", type=['json'])
        if uploaded_json:
            st.json(json.load(uploaded_json))

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
