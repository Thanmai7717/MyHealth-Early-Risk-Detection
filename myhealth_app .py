import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="CKDPredict | Random Forest Edition", page_icon="🧬", layout="wide")

# Custom CSS for the "Master's Project" Look
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #6D28D9; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    """Loads datasets and prepares features from your specific CSV columns."""
    p = pd.read_csv('patients.csv')
    o = pd.read_csv('observations.csv')
    c = pd.read_csv('conditions.csv')
    
    # Calculate Age from BIRTHDATE
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'] + " " + p['LAST']
    
    return p, o, c

@st.cache_resource
def train_random_forest_model(df_p, df_o, df_c):
    """
    Trains the Random Forest model for Chronic Kidney Disease.
    Features include: Demographics (Age, Gender, Income) + Clinical Vitals.
    """
    # 1. Feature Extraction: Focus on Kidney Markers
    kidney_markers = ['Creatinine', 'Urea Nitrogen', 'Blood Pressure Systolic', 'Glucose', 'Body Mass Index']
    v_df = df_o[df_o['DESCRIPTION'].str.contains('|'.join(kidney_markers), case=False, na=False)]
    v_pivot = v_df.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    # 2. Merge with your specific Patient columns
    df_ml = pd.merge(v_pivot, df_p[['Id', 'AGE', 'GENDER', 'INCOME']], left_on='PATIENT', right_on='Id')
    df_ml['GENDER'] = df_ml['GENDER'].map({'M': 1, 'F': 0})
    
    # 3. Labeling (Target)
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney', case=False, na=False)]['PATIENT'].unique()
    df_ml['target'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)
    
    # 4. Training the Random Forest
    df_final = df_ml.dropna()
    X = df_final.drop(['PATIENT', 'Id', 'target'], axis=1)
    y = df_final['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, rf_model.predict(X_test))
    return rf_model, X.columns.tolist(), acc

# --- MAIN LOGIC ---
try:
    df_p, df_o, df_c = load_and_prep_data()
    model, features, model_acc = train_random_forest_model(df_p, df_o, df_c)

    # SIDEBAR
    st.sidebar.title("🔬 SLU Research")
    st.sidebar.subheader("CKDPredict System")
    st.sidebar.markdown(f"**Model:** Random Forest\n**Accuracy:** {model_acc:.2%}")
    st.sidebar.divider()
    
    # Patient Selector
    patient_name = st.sidebar.selectbox("Select Patient Profile", options=df_p['FULL_NAME'].sort_values())
    patient_row = df_p[df_p['FULL_NAME'] == patient_name].iloc[0]
    p_id = patient_row['Id']

    # Navigation
    tab_main, tab_predict, tab_json = st.tabs(["📊 Patient Dashboard", "🧠 AI Kidney Prediction", "📂 Data Records"])

    # Utility: Get latest vital for patient
    user_obs = df_o[df_o['PATIENT'] == p_id]
    def fetch_vital(name):
        res = user_obs[user_obs['DESCRIPTION'].str.contains(name, case=False)]
        return res.iloc[-1]['VALUE'] if not res.empty else 0.0

    # ---------------------------------------------------------
    # TAB 1: DASHBOARD
    # ---------------------------------------------------------
    with tab_main:
        st.header(f"Clinical Profile: {patient_name}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patient Age", int(patient_row['AGE']))
        m2.metric("Gender", patient_row['GENDER'])
        m3.metric("Income", f"${patient_row['INCOME']:,.0f}")
        m4.metric("Creatinine", f"{fetch_vital('Creatinine')} mg/dL")

        st.divider()
        st.subheader("Documented Conditions")
        conditions = df_c[df_c['PATIENT'] == p_id]['DESCRIPTION'].unique()
        if len(conditions) > 0:
            for cond in conditions:
                st.write(f"• {cond}")
        else:
            st.write("No existing chronic diagnoses.")

    # ---------------------------------------------------------
    # TAB 2: AI PREDICTION (Random Forest)
    # ---------------------------------------------------------
    with tab_predict:
        st.header("Random Forest Risk Analysis")
        st.write("Generating a real-time risk score for Chronic Kidney Disease Stage 3.")

        # Mapping input data to match model features
        input_data = {
            'AGE': patient_row['AGE'],
            'GENDER': 1 if patient_row['GENDER'] == 'M' else 0,
            'INCOME': patient_row['INCOME'],
            'Blood Pressure Systolic': fetch_vital("Systolic"),
            'Body Mass Index': fetch_vital("Body Mass Index"),
            'Creatinine': fetch_vital("Creatinine"),
            'Glucose': fetch_vital("Glucose"),
            'Urea Nitrogen': fetch_vital("Urea Nitrogen")
        }
        
        # Ensure correct column order
        input_df = pd.DataFrame([input_data])[features]
        
        # Prediction
        probability = model.predict_proba(input_df)[0][1]
        risk_pct = probability * 100

        st.divider()
        st.subheader("Prediction Result")
        
        col_res, col_gauge = st.columns([1, 2])
        with col_res:
            st.metric("CKD Risk Probability", f"{risk_pct:.1f}%")
            if risk_pct > 70:
                st.error("HIGH RISK: Immediate screening recommended.")
            elif risk_pct > 35:
                st.warning("MODERATE RISK: Increased monitoring of metabolic markers.")
            else:
                st.success("LOW RISK: Stable renal profile.")
        
        with col_gauge:
            st.progress(probability)
            st.caption("Random Forest Classifier confidence levels based on historical California cohort data.")

    # ---------------------------------------------------------
    # TAB 3: JSON RECORDS
    # ---------------------------------------------------------
    with tab_json:
        st.header("Digital Health Records")
        uploaded_file = st.file_uploader("Upload Hospital Visit Summary (JSON)", type=['json'])
        if uploaded_file:
            data = json.load(uploaded_file)
            st.json(data)

except Exception as e:
    st.error(f"Critical System Error: {e}. Ensure all patient CSV files are in the local directory.")
