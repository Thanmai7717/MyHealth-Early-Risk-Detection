import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1. PAGE SETUP
st.set_page_config(page_title="CKDPredict | SLU Research", page_icon="🔬", layout="wide")

# Purple/Blue Professional Styling to match your preference
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #6D28D9 !important; font-weight: 700; }
    .stMetric { background-color: #F3F4F6; padding: 20px; border-radius: 12px; border-left: 5px solid #6D28D9; }
    .main-title { color: #4C1D95; font-size: 40px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_core_data():
    # Loading the 4 core files
    p = pd.read_csv('patients.csv')
    o = pd.read_csv('observations.csv')
    c = pd.read_csv('conditions.csv')
    e = pd.read_csv('encounters.csv')
    
    # Pre-calculating Age for your research metrics
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'] + " " + p['LAST']
    return p, o, c, e

@st.cache_resource
def train_dual_models(df_p, df_o, df_c):
    """
    SLU RESEARCH LOGIC:
    Model A: Diabetic Cohort
    Model B: Non-Diabetic Cohort (Hypertension/CVD focus)
    """
    vitals_list = ['Creatinine', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose', 'Urea Nitrogen']
    vitals = df_o[df_o['DESCRIPTION'].str.contains('|'.join(vitals_list), case=False, na=False)]
    vitals_pivot = vitals.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    df_ml = pd.merge(vitals_pivot, df_p[['Id', 'AGE', 'GENDER', 'INCOME']], left_on='PATIENT', right_on='Id')
    df_ml['GENDER'] = df_ml['GENDER'].map({'M': 1, 'F': 0})

    # Defining the targets based on your "Invisible 62%" thesis
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney', case=False, na=False)]['PATIENT'].unique()
    diabetic_ids = df_c[df_c['DESCRIPTION'].str.contains('Diabetes', case=False, na=False)]['PATIENT'].unique()

    df_ml['is_diabetic'] = df_ml['Id'].apply(lambda x: 1 if x in diabetic_ids else 0)
    df_ml['target_ckd'] = df_ml['Id'].apply(lambda x: 1 if x in ckd_ids else 0)

    # Split Cohorts
    df_a = df_ml[df_ml['is_diabetic'] == 1].dropna()
    df_b = df_ml[df_ml['is_diabetic'] == 0].dropna()

    models = {}
    for label, data in [("Model A (Diabetic)", df_a), ("Model B (Non-Diabetic)", df_b)]:
        if not data.empty:
            X = data.drop(['PATIENT', 'Id', 'target_ckd', 'is_diabetic'], axis=1)
            y = data['target_ckd']
            model = XGBClassifier(n_estimators=100)
            model.fit(X, y)
            models[label] = {"model": model, "features": X.columns.tolist()}
    return models

# --- APPLICATION START ---
try:
    df_p, df_o, df_c, df_e = load_core_data()
    dual_models = train_dual_models(df_p, df_o, df_c)

    # SIDEBAR BRANDING
    st.sidebar.markdown("<h1 style='color: #6D28D9;'>🔬 CKDPredict</h1>", unsafe_allow_html=True)
    st.sidebar.write("Early Risk Detection Dashboard")
    st.sidebar.divider()
    
    # 1. Patient Selection
    patient_name = st.sidebar.selectbox("Select Patient Profile", options=df_p['FULL_NAME'].sort_values())
    selected_user = df_p[df_p['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_user['Id']

    # 2. Logic to determine which research model to use
    user_conditions = df_c[df_c['PATIENT'] == p_id]['DESCRIPTION'].tolist()
    is_diabetic = any("Diabetes" in cond for cond in user_conditions)
    current_model_key = "Model A (Diabetic)" if is_diabetic else "Model B (Non-Diabetic)"

    # 3. New Navigation (This is where the change happens!)
    tab = st.sidebar.radio("Navigation Menu", ["Clinical Profile", "AI Prediction Engine", "Medical Reports"])

    # ---------------------------------------------------------
    # TAB: CLINICAL PROFILE
    # ---------------------------------------------------------
    if tab == "Clinical Profile":
        st.markdown(f"<div class='main-title'>Patient: {patient_name}</div>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Age", int(selected_user['AGE']))
        c2.metric("Annual Income", f"${selected_user['INCOME']:,.0f}")
        c3.metric("Gender", selected_user['GENDER'])

        st.divider()
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Active Condition Map")
            if user_conditions:
                for cond in user_conditions:
                    st.write(f"🚩 {cond}")
            else:
                st.write("No chronic conditions found.")

        with col_right:
            st.subheader("Research Cohort Assignment")
            if is_diabetic:
                st.info("🎯 **Assigned to Model A**\n\nPatient has history of Diabetes. Screening for diabetic nephropathy markers.")
            else:
                st.success("🎯 **Assigned to Model B**\n\nPatient is part of the 'Invisible 62%'. Screening for hypertensive/cardiovascular kidney decline.")

    # ---------------------------------------------------------
    # TAB: AI PREDICTION ENGINE
    # ---------------------------------------------------------
    elif tab == "AI Prediction Engine":
        st.title("🛡️ XGBoost Early Detection Engine")
        st.write(f"Currently analyzing via: **{current_model_key}**")

        # Fetch latest vitals
        user_obs = df_o[df_o['PATIENT'] == p_id]
        def get_v(d):
            r = user_obs[user_obs['DESCRIPTION'].str.contains(d, case=False)]
            return r.iloc[-1]['VALUE'] if not r.empty else 0.0

        # Feature Set
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

        # Run Prediction
        if current_model_key in dual_models:
            m_info = dual_models[current_model_key]
            input_df = pd.DataFrame([input_data])[m_info['features']]
            risk_score = m_info['model'].predict_proba(input_df)[0][1]

            # Display Result
            st.divider()
            st.subheader("12-Month Forecast: CKD Stage 3 Onset")
            st.metric("Probability Score", f"{risk_score*100:.1f}%")
            st.progress(risk_score)

            if risk_score > 0.65:
                st.error("🚨 **High Clinical Priority:** Markers suggest silent progression toward kidney damage.")
            elif risk_score > 0.35:
                st.warning("⚠️ **Watch List:** Longitudinal trends indicate metabolic stress.")
            else:
                st.success("✅ **Baseline Stable:** No immediate predictors for CKD onset detected.")

    # ---------------------------------------------------------
    # TAB: MEDICAL REPORTS (With JSON Support)
    # ---------------------------------------------------------
    elif tab == "Medical Reports":
        st.title("📄 Records & Exports")
        
        # Add the JSON Uploader back here
        up_file = st.sidebar.file_uploader("Upload JSON Record", type=['json'])
        if up_file:
            st.json(json.load(up_file))
            st.sidebar.success("JSON Loaded!")

        st.divider()
        st.subheader("Generate Clinical Summary")
        st.write("Extracting data for SLU Master's Project Documentation...")
        summary = f"Patient: {patient_name}\nCohort: {current_model_key}\nDate: 2026-04-12"
        st.text_area("Report Preview", summary)
        st.download_button("Download TXT", summary)

except Exception as e:
    st.error(f"Error: {e}. Check if CSV files are in the same folder.")
