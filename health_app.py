import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ── 1. PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="MyHealth Chronic Disease AI", page_icon="👤", layout="wide")

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

# ── 3. BUILD MODEL DATA ───────────────────────────────────────
@st.cache_data
def build_model_data(patients, conditions, observations):
    chronic_pattern = 'Diabetes|Hypertension|Heart|Kidney|Renal|COPD|Asthma|Alzheimer'
    chronic_ids = conditions[conditions['DESCRIPTION'].str.contains(
                chronic_pattern, case=False, na=False)]['PATIENT'].unique()
    
    target_df = pd.DataFrame({'Id': patients['Id']})
    target_df['target'] = target_df['Id'].isin(chronic_ids).astype(int)

    def clean_demographics(df):
        keep_cols = ['Id', 'GENDER', 'RACE', 'BIRTHDATE', 'CITY']
        df = df[keep_cols].copy()
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
        df['Age'] = 2026 - df['BIRTHDATE'].dt.year
        df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
        df = pd.get_dummies(df, columns=['RACE'], prefix='race', drop_first=True)
        return df.drop(columns=['BIRTHDATE'])

    patients_cleaned = clean_demographics(patients)

    relevant_codes = ['2339-0', '4548-4', '8480-6', '2160-0', '33914-3', '39156-5']
    obs_filtered = observations[observations['CODE'].isin(relevant_codes)]
    obs_latest = (obs_filtered.sort_values('DATE')
                  .groupby(['PATIENT', 'DESCRIPTION'])['VALUE']
                  .last().unstack())
    
    obs_latest.columns = [col.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_').lower() for col in obs_latest.columns]
    obs_latest = obs_latest.reset_index().rename(columns={'PATIENT': 'Id'})

    df_final = target_df.merge(patients_cleaned, on='Id', how='left').merge(obs_latest, on='Id', how='left')
    return df_final

# ── 4. TRAIN MODELS ───────────────────────────────────────────
@st.cache_resource
def train_models(_df_final):
    X = _df_final.drop(columns=['target', 'Id', 'CITY'], errors='ignore')
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

    return lasso_model, scaler, xgb_model, X

# ── 5. HELPER ─────────────────────────────────────────────────
def get_latest_vital(desc, user_o):
    res = user_o[user_o['DESCRIPTION'].str.lower().str.contains(desc.lower(), na=False)]
    if not res.empty:
        return res.iloc[-1]['VALUE']
    return "N/A"

# ── 6. MAIN EXECUTION ─────────────────────────────────────────
try:
    df_p, df_o, df_c, df_e = load_data()
    model_data = build_model_data(df_p, df_c, df_o)
    lasso_m, sc, xgb_m, X_cols = train_models(model_data)

    # SIDEBAR
    st.sidebar.title("🧬 MyHealth AI")
    df_p['FULL_NAME'] = df_p['FIRST'] + " " + df_p['LAST']
    patient_name = st.sidebar.selectbox("Select Profile", options=df_p['FULL_NAME'].sort_values())
    
    selected_row = df_p[df_p['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_row['Id']
    
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("Upload Lab Report", type=['pdf','json','png','jpg'])
    tab = st.sidebar.radio("Navigation", ["Home", "My History", "Health Check", "Model Insights", "My Reports"])

    # User Context Data
    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    user_c = df_c[df_c['PATIENT'] == p_id]
    
    # ── TAB: MY REPORTS (DETAILED) ─────────────────────────────
    if tab == "My Reports":
        st.title("📄 Export My Health Summary")
        st.write("Download a summary of your California health records to share with your doctor.")

        # Data Preparation for Report
        loc = f"{selected_row['CITY']}, CA"
        bp = get_latest_vital('Systolic', user_o)
        glu = get_latest_vital('Glucose', user_o)
        
        # Filter for Chronic Conditions
        chronic_pattern = 'Diabetes|Hypertension|Heart|Kidney|Renal|COPD|Asthma|Osteoarthritis'
        active_conds = user_c[user_c['DESCRIPTION'].str.contains(chronic_pattern, case=False)]
        cond_list = ", ".join(active_conds['DESCRIPTION'].unique()) if not active_conds.empty else "No chronic conditions recorded."
        
        gen_date = datetime.now().strftime("%Y-%m-%d")

        # The Formatted Report Text
        detailed_report = f"""
  MYHEALTH DASHBOARD REPORT
  --------------------------
  Patient: {patient_name}
  Location: {loc}
  
  LATEST VITALS:
  - Blood Pressure: {bp} mmHg
  - Glucose: {glu} mg/dL
  
  ACTIVE CHRONIC CONDITIONS:
  {cond_list}
  
  Weekly Exercise Goal: 150 minutes
  
  Generated on: {gen_date}
  """

        st.subheader("Preview your report:")
        st.code(detailed_report, language="text")
        
        st.download_button(
            label="📥 Download as Text File",
            data=detailed_report,
            file_name=f"{patient_name.replace(' ', '_')}_Health_Report.txt",
            mime="text/plain"
        )

    # ── OTHER TABS (Keeping all your existing features) ──
    elif tab == "Home":
        st.title(f"👋 Welcome, {selected_row['FIRST']}!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current eGFR", f"{get_latest_vital('GFR', user_o)} mL/min")
        m2.metric("Blood Pressure", f"{get_latest_vital('Systolic', user_o)} mmHg")
        m3.metric("Chronic Conditions", len(active_conds) if 'active_conds' in locals() else 0)

        st.divider()
        st.subheader("📝 Daily Health Check-in")
        with st.form("checkin"):
            q1 = st.select_slider("Energy level", ["Very Low", "Low", "Normal", "High"])
            q2 = st.radio("Any new swelling?", ["No", "Yes"])
            if st.form_submit_button("Submit"):
                st.success("Responses saved!")

    elif tab == "My History":
        st.title("🏥 Medical Visit History")
        user_e = df_e[df_e['PATIENT'] == p_id].sort_values('START', ascending=False)
        st.table(user_e[['START', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].head(10))

    elif tab == "Health Check":
        st.title("🩺 Vital Signs Trend")
        metric = st.selectbox("Select Metric", ["GFR", "Creatinine", "Systolic", "Glucose"])
        trend = user_o[user_o['DESCRIPTION'].str.contains(metric, case=False)].copy()
        trend['VALUE'] = pd.to_numeric(trend['VALUE'], errors='coerce')
        if not trend.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pd.to_datetime(trend['DATE']), trend['VALUE'], marker='o')
            st.pyplot(fig)

    elif tab == "Model Insights":
        st.title("🤖 AI Risk Drivers")
        feat_imp = pd.Series(xgb_m.feature_importances_, index=X_cols.columns).sort_values().tail(10)
        fig, ax = plt.subplots()
        feat_imp.plot(kind='barh', color='teal')
        st.pyplot(fig)

except Exception as e:
    st.error(f"Dashboard Error: {e}")
    st.exception(e)
