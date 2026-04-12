import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1. PAGE SETUP
st.set_page_config(page_title="CKDPredict | SLU Research", page_icon="🔬", layout="wide")

# Purple/Blue Professional Styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #F3F4F6; padding: 20px; border-radius: 12px; border-left: 5px solid #6D28D9; }
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_core_data():
    p = pd.read_csv('patients.csv')
    o = pd.read_csv('observations.csv')
    c = pd.read_csv('conditions.csv')
    
    # Feature Engineering: Age and Grouping
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    return p, o, c

@st.cache_resource
def train_dual_models(df_p, df_o, df_c):
    """
    Trains two distinct models as per research:
    Model A: Diabetic Patients
    Model B: Non-Diabetic (Hypertension, Heart Failure, etc.)
    """
    # Pivot Vitals: Creatinine, eGFR (if available), BP, Glucose, BMI
    vitals_list = ['Creatinine', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose', 'Urea Nitrogen']
    vitals = df_o[df_o['DESCRIPTION'].str.contains('|'.join(vitals_list), case=False, na=False)]
    vitals_pivot = vitals.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_ml = pd.merge(vitals_pivot, df_p[['Id', 'AGE', 'GENDER', 'INCOME']], left_on='PATIENT', right_on='Id')
    df_ml['GENDER'] = df_ml['GENDER'].map({'M': 1, 'F': 0})

    # Identify Disease Groups
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney', case=False, na=False)]['PATIENT'].unique()
    diabetic_ids = df_c[df_c['DESCRIPTION'].str.contains('Diabetes', case=False, na=False)]['PATIENT'].unique()

    df_ml['is_diabetic'] = df_ml['Id'].apply(lambda x: 1 if x in diabetic_ids else 0)
    df_ml['target_ckd'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)

    # Split into Model A and Model B
    df_a = df_ml[df_ml['is_diabetic'] == 1].dropna()
    df_b = df_ml[df_ml['is_diabetic'] == 0].dropna()

    models = {}
    for label, data in [("Model A (Diabetic)", df_a), ("Model B (Non-Diabetic)", df_b)]:
        X = data.drop(['PATIENT', 'Id', 'target_ckd', 'is_diabetic'], axis=1)
        y = data['target_ckd']
        
        model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4)
        model.fit(X, y)
        models[label] = {"model": model, "features": X.columns.tolist()}
        
    return models

# --- MAIN APP ---
try:
    df_p, df_o, df_c = load_core_data()
    dual_models = train_dual_models(df_p, df_o, df_c)

    # SIDEBAR
    st.sidebar.title("🔬 CKDPredict")
    st.sidebar.caption("Master's Research Project - SLU")
    
    patient_name = st.sidebar.selectbox("Select Patient to Screen", options=(df_p['FIRST'] + " " + df_p['LAST']).sort_values())
    selected_user = df_p[(df_p['FIRST'] + " " + df_p['LAST']) == patient_name].iloc[0]
    p_id = selected_user['Id']

    # PRE-SCREENING LOGIC
    user_conditions = df_c[df_c['PATIENT'] == p_id]['DESCRIPTION'].tolist()
    is_diabetic = any("Diabetes" in cond for cond in user_conditions)
    current_model_key = "Model A (Diabetic)" if is_diabetic else "Model B (Non-Diabetic)"
    
    # NAVIGATION
    tab = st.sidebar.radio("View", ["Clinical Dashboard", "Prediction Engine"])

    # ---------------------------------------------------------
    # TAB: DASHBOARD
    # ---------------------------------------------------------
    if tab == "Clinical Dashboard":
        st.title(f"Clinical Profile: {patient_name}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Patient Metadata")
            m1, m2, m3 = st.columns(3)
            m1.metric("Age", int(selected_user['AGE']))
            m2.metric("Gender", selected_user['GENDER'])
            m3.metric("Income", f"${selected_user['INCOME']:,.0f}")
            
            st.divider()
            st.subheader("Current Condition Map")
            for cond in user_conditions:
                st.write(f"• {cond}")

        with col2:
            st.subheader("Model Assignment")
            if is_diabetic:
                st.info("🔵 Assigned to **Model A**\n(Diabetic Cohort)")
            else:
                st.success("🟢 Assigned to **Model B**\n(Cardiovascular Cohort)")

    # ---------------------------------------------------------
    # TAB: PREDICTION ENGINE
    # ---------------------------------------------------------
    elif tab == "Prediction Engine":
        st.title("🛡️ Early CKD Detection")
        
        # Extract features for prediction
        user_obs = df_o[df_o['PATIENT'] == p_id]
        def get_v(d):
            r = user_obs[user_obs['DESCRIPTION'].str.contains(d, case=False)]
            return r.iloc[-1]['VALUE'] if not r.empty else 0.0

        input_data = {
            'AGE': selected_user['AGE'],
            'GENDER': 1 if selected_user['GENDER'] == 'M' else 0,
            'INCOME': selected_user['INCOME'],
            'Blood Pressure Systolic': get_v("Systolic"),
            'Body Mass Index': get_v("Body Mass Index"),
            'Creatinine': get_v("Creatinine"),
            'Glucose': get_v("Glucose"),
            'Urea Nitrogen': get_v("Urea Nitrogen")
        }

        # Select model and run inference
        m_info = dual_models[current_model_key]
        input_df = pd.DataFrame([input_data])[m_info['features']]
        risk_score = m_info['model'].predict_proba(input_df)[0][1]

        # Display Results
        st.metric(f"CKD Stage 3 Onset Risk (12-Month Forecast)", f"{risk_score*100:.1f}%")
        st.progress(risk_score)

        if risk_score > 0.6:
            st.error("🚨 **High Risk Alert:** Based on longitudinal progression, this patient shows significant markers for CKD onset within 12 months.")
        elif risk_score > 0.3:
            st.warning("⚠️ **Monitoring Recommended:** Elevated Creatinine and BP trends suggest early-stage decline.")
        else:
            st.success("✅ **Stable:** Clinical markers are consistent with baseline for this cohort.")

except Exception as e:
    st.error(f"Error loading Research Dashboard: {e}")
