import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ── 1. PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="MyHealth: Chronic Disease AI", page_icon="👤", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #007AFF !important; font-weight: 700; }
    .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); }
    </style>
    """, unsafe_allow_html=True)

# ── 2. LOAD DATA ──────────────────────────────────────────────
@st.cache_data
def load_data():
    p = pd.read_csv('patients.csv')
    c = pd.read_csv('conditions.csv')
    e = pd.read_csv('encounters.csv')
    try:
        o = pd.read_csv('observations.csv.gz', compression='gzip')
    except:
        o = pd.read_csv('observations.csv')
    return p, o, c, e

# ── 3. BUILD BROAD MODEL DATA ─────────────────────────────────
@st.cache_data
def build_model_data(patients, conditions, observations):
    chronic_pattern = 'Diabetes|Hypertension|Heart|Kidney|Renal|COPD|Asthma|Alzheimer'
    chronic_ids = conditions[conditions['DESCRIPTION'].str.contains(
                chronic_pattern, case=False, na=False)]['PATIENT'].unique()
    
    target_df = pd.DataFrame({'Id': patients['Id']})
    target_df['target'] = target_df['Id'].isin(chronic_ids).astype(int)

    df_p = patients[['Id', 'GENDER', 'RACE', 'BIRTHDATE', 'INCOME']].copy()
    df_p['BIRTHDATE'] = pd.to_datetime(df_p['BIRTHDATE'])
    df_p['Age'] = 2026 - df_p['BIRTHDATE'].dt.year
    df_p['GENDER'] = df_p['GENDER'].map({'M': 0, 'F': 1})
    df_p = pd.get_dummies(df_p, columns=['RACE'], prefix='race', drop_first=True)
    df_p = df_p.drop(columns=['BIRTHDATE'])

    relevant_codes = ['2339-0', '4548-4', '8480-6', '2160-0', '33914-3', '39156-5']
    obs_filtered = observations[observations['CODE'].isin(relevant_codes)]
    obs_latest = (obs_filtered.sort_values('DATE')
                  .groupby(['PATIENT', 'DESCRIPTION'])['VALUE']
                  .last().unstack())
    
    obs_latest.columns = [col.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_').lower() for col in obs_latest.columns]
    obs_latest = obs_latest.reset_index().rename(columns={'PATIENT': 'Id'})

    df_final = target_df.merge(df_p, on='Id', how='left').merge(obs_latest, on='Id', how='left')
    return df_final

# ── 4. TRAIN MODELS ───────────────────────────────────────────
@st.cache_resource
def train_models(_df_final):
    X = _df_final.drop(columns=['target', 'Id'])
    y = _df_final['target']

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
    lasso_model.fit(X_scaled, y_train)

    return lasso_model, scaler, xgb_model, X, X_test, y_test

# ── 5. HELPER ─────────────────────────────────────────────────
def get_latest_vital(desc, user_o):
    res = user_o[user_o['DESCRIPTION'].str.lower().str.contains(desc.lower(), na=False)]
    if not res.empty:
        try: return f"{float(res.iloc[-1]['VALUE']):.1f}"
        except: return res.iloc[-1]['VALUE']
    return "N/A"

# ── 6. MAIN APP ───────────────────────────────────────────────
try:
    df_p, df_o, df_c, df_e = load_data()
    
    model_data = build_model_data(df_p, df_c, df_o)
    lasso_m, sc, xgb_m, X_cols, X_test, y_test = train_models(model_data)

    # SIDEBAR
    st.sidebar.title("🧬 MyHealth AI")
    df_p['FULL_NAME'] = df_p['FIRST'] + " " + df_p['LAST']
    patient_name = st.sidebar.selectbox("Select Profile", options=df_p['FULL_NAME'].sort_values())
    
    selected_row = df_p[df_p['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_row['Id']
    
    # --- MODEL PERFORMANCE REMOVED FROM HERE ---
    
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("Upload Lab Report", type=['pdf','json','png','jpg'])
    tab = st.sidebar.radio("Navigation", ["Overview", "Vitals Trend", "Daily Check-in", "AI Insights", "Upload & Export"])

    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    user_c = df_c[df_c['PATIENT'] == p_id]

    # ── TABS ──
    if tab == "Overview":
        st.title(f"Health Summary: {patient_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blood Pressure", f"{get_latest_vital('Systolic', user_o)} mmHg")
        c2.metric("Glucose", f"{get_latest_vital('Glucose', user_o)} mg/dL")
        c3.metric("BMI", get_latest_vital("Body Mass Index", user_o))
        c4.metric("eGFR", get_latest_vital("GFR", user_o))

        st.subheader("Active Chronic Conditions")
        chronic_pattern = 'Diabetes|Hypertension|Heart|Kidney|COPD|Asthma'
        my_conditions = user_c[user_c['DESCRIPTION'].str.contains(chronic_pattern, case=False)]
        if not my_conditions.empty:
            for cond in my_conditions['DESCRIPTION'].unique():
                st.error(f"● {cond}")
        else:
            st.success("No major chronic conditions detected in records.")

    elif tab == "Daily Check-in":
        st.title("📝 Daily Patient Questionnaire")
        with st.form("general_survey"):
            q1 = st.slider("Energy level (1-10)", 1, 10, 5)
            q2 = st.radio("Are you experiencing any shortness of breath?", ["No", "During exercise", "While resting"])
            q3 = st.radio("Have you taken your medications today?", ["Yes", "No", "Not prescribed"])
            q4 = st.checkbox("Any chest pain or unusual dizziness?")
            if st.form_submit_button("Submit"):
                st.success("Check-in saved!")

    elif tab == "AI Insights":
        st.title("🤖 Chronic Risk Predictors")
        feat_imp = pd.Series(xgb_m.feature_importances_, index=X_cols.columns).sort_values().tail(8)
        fig, ax = plt.subplots()
        feat_imp.plot(kind='barh', color='teal', ax=ax)
        st.pyplot(fig)

    elif tab == "Upload & Export":
        st.title("📄 Records Management")
        report = f"Health Report for {patient_name}\nBP: {get_latest_vital('Systolic', user_o)}\nGlucose: {get_latest_vital('Glucose', user_o)}"
        st.download_button("Download Report", report, file_name="health_summary.txt")

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
