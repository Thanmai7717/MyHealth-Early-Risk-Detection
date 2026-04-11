import streamlit as st
import pandas as pd
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. PAGE SETUP
st.set_page_config(page_title="MyHealth AI: Early Risk Detection", page_icon="👤", layout="wide")

# UI Styling (Purple/Blue professional theme)
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #8A2BE2 !important; font-weight: 700; }
    .stMetric { background-color: rgba(138, 43, 226, 0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(138, 43, 226, 0.2); }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Loads the core datasets and calculates patient Age."""
    p = pd.read_csv('patients.csv')
    o = pd.read_csv('observations.csv')
    c = pd.read_csv('conditions.csv')
    e = pd.read_csv('encounters.csv')
    
    # Pre-calculate Age for ML features
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = 2026 - p['BIRTHDATE'].dt.year
    p['FULL_NAME'] = p['FIRST'] + " " + p['LAST']
    return p, o, c, e

@st.cache_resource
def train_chronic_models(df_p, df_o, df_c):
    """
    Trains XGBoost models for specific chronic diseases using 
    Demographics (Age, Income, Gender) + Vitals (BP, Glucose, Creatinine, etc.)
    """
    # 1. Pivot Vitals to get the most recent data per patient
    vitals_list = ['Hemoglobin', 'Creatinine', 'Blood Pressure Systolic', 'Body Mass Index', 'Glucose']
    vitals = df_o[df_o['DESCRIPTION'].str.contains('|'.join(vitals_list), case=False, na=False)]
    vitals_pivot = vitals.pivot_table(index='PATIENT', columns='DESCRIPTION', values='VALUE', aggfunc='last').reset_index()
    
    # 2. Merge with Patient Demographics (using your specific columns)
    df_ml = pd.merge(vitals_pivot, df_p[['Id', 'AGE', 'GENDER', 'INCOME']], left_on='PATIENT', right_on='Id')
    df_ml['GENDER'] = df_ml['GENDER'].map({'M': 1, 'F': 0})
    
    target_diseases = {
        'Chronic Kidney Disease': 'Kidney',
        'Heart Failure': 'Heart',
        'Anemia': 'Anemia'
    }
    
    trained_models = {}

    for display_name, search_term in target_diseases.items():
        # Labeling Target: 1 if patient has condition in conditions.csv
        disease_ids = df_c[df_c['DESCRIPTION'].str.contains(search_term, case=False, na=False)]['PATIENT'].unique()
        df_ml['target'] = df_ml['PATIENT'].apply(lambda x: 1 if x in disease_ids else 0)
        
        # Clean data for training
        df_clean = df_ml.dropna().copy()
        
        if len(df_clean['target'].unique()) > 1:
            X = df_clean.drop(['PATIENT', 'Id', 'target'], axis=1)
            y = df_clean['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            trained_models[display_name] = {"model": model, "accuracy": acc, "features": X.columns.tolist()}
            
    return trained_models

# --- MAIN EXECUTION ---
try:
    df_p, df_o, df_c, df_e = load_data()
    disease_models = train_chronic_models(df_p, df_o, df_c)

    # --- SIDEBAR ---
    st.sidebar.title("👤 MyHealth AI")
    
    # Select Patient
    patient_name = st.sidebar.selectbox("Select Profile", options=df_p['FULL_NAME'].sort_values())
    user_data = df_p[df_p['FULL_NAME'] == patient_name].iloc[0]
    p_id = user_data['Id']
    
    st.sidebar.markdown(f"**Logged in:** {patient_name}")
    st.sidebar.divider()

    # File Upload (Support for JSON)
    st.sidebar.subheader("📤 Medical Records")
    uploaded_file = st.sidebar.file_uploader("Upload Visit Summary", type=['pdf', 'json', 'png'])
    uploaded_json_data = None
    if uploaded_file and uploaded_file.name.endswith('.json'):
        uploaded_json_data = json.load(uploaded_file)

    tab = st.sidebar.radio("Navigation", ["Home Dashboard", "AI Risk Analysis", "My Reports"])

    # --- DATA UTILS FOR SELECTED USER ---
    user_o = df_o[df_o['PATIENT'] == p_id]
    def get_latest_val(desc):
        res = user_o[user_o['DESCRIPTION'].str.contains(desc, case=False, na=False)]
        return res.iloc[-1]['VALUE'] if not res.empty else 0.0

    # ---------------------------------------------------------
    # TAB: HOME DASHBOARD
    # ---------------------------------------------------------
    if tab == "Home Dashboard":
        st.title(f"👋 Hello, {user_data['FIRST']}!")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age", int(user_data['AGE']))
        m2.metric("BMI", f"{get_latest_val('Body Mass Index'):.1f}")
        m3.metric("Income Level", f"${user_data['INCOME']:,.0f}")
        m4.metric("Creatinine", f"{get_latest_val('Creatinine')} mg/dL")

        st.subheader("📋 Current Diagnosed Conditions")
        current_c = df_c[df_c['PATIENT'] == p_id]
        if not current_c.empty:
            for c in current_c['DESCRIPTION'].unique():
                st.success(f"**{c}**")
        else:
            st.write("No chronic conditions on record.")

    # ---------------------------------------------------------
    # TAB: AI RISK ANALYSIS
    # ---------------------------------------------------------
    elif tab == "AI Risk Analysis":
        st.title("🤖 AI Early Risk Detection")
        st.write("Predictions based on your demographics and latest clinical observations.")

        # Construct input features matching the model
        current_vitals = {
            'AGE': user_data['AGE'],
            'GENDER': 1 if user_data['GENDER'] == 'M' else 0,
            'INCOME': user_data['INCOME'],
            'Blood Pressure Systolic': get_latest_val("Systolic"),
            'Body Mass Index': get_latest_val("Body Mass Index"),
            'Creatinine': get_latest_val("Creatinine"),
            'Hemoglobin': get_latest_val("Hemoglobin"),
            'Glucose': get_latest_val("Glucose")
        }
        
        input_df = pd.DataFrame([current_vitals])

        cols = st.columns(3)
        for i, (name, m_info) in enumerate(disease_models.items()):
            # Reorder columns to match the trained model's feature list
            final_input = input_df[m_info['features']]
            risk_prob = m_info['model'].predict_proba(final_input)[0][1]
            
            with cols[i]:
                st.metric(name, f"{risk_prob*100:.1f}% Risk")
                st.progress(risk_prob)
                st.caption(f"Model Confidence: {m_info['accuracy']:.1%}")

        st.divider()
        if any(m_info['model'].predict_proba(input_df[m_info['features']])[0][1] > 0.5 for m_info in disease_models.values()):
            st.warning("⚠️ **Note:** Some risk scores are elevated. Consider reviewing these with a healthcare provider.")

    # ---------------------------------------------------------
    # TAB: MY REPORTS
    # ---------------------------------------------------------
    elif tab == "My Reports":
        st.title("📄 Digital Records & Summary")
        
        if uploaded_json_data:
            with st.expander("📂 View Uploaded JSON Record", expanded=True):
                st.json(uploaded_json_data)
        
        st.divider()
        st.subheader("Generate Health Export")
        summary_text = f"Patient: {user_data['FULL_NAME']}\nAge: {user_data['AGE']}\nRisk Status: Generated by MyHealth AI."
        st.text_area("Preview:", summary_text)
        st.download_button("📥 Download Report", summary_text, file_name="Health_Summary.txt")

except Exception as e:
    st.error(f"System Error: {e}. Ensure all CSV files (patients, observations, conditions, encounters) are in the folder.")
