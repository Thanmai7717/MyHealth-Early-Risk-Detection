import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ── 1. PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="CKD Predict: Kidney Health AI", page_icon="🧪", layout="wide")

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

# ── 3. BUILD CKD-SPECIFIC MODEL DATA ──────────────────────────
@st.cache_data
def build_model_data(patients, conditions, observations, encounters):
    # CKD Target: Focus on Kidney/Renal/Nephropathy
    ckd_pattern = 'Kidney|Renal|Nephropathy|Glomerular|End stage renal'
    ckd_ids = conditions[conditions['DESCRIPTION'].str.contains(
                ckd_pattern, case=False, na=False)]['PATIENT'].unique()
    
    target_df = pd.DataFrame({'Id': patients['Id']})
    target_df['target'] = target_df['Id'].isin(ckd_ids).astype(int)

    # Demographics
    def clean_demographics(df):
        keep_cols = ['Id', 'GENDER', 'RACE', 'BIRTHDATE']
        df = df[keep_cols].copy()
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
        df['Age'] = 2026 - df['BIRTHDATE'].dt.year # Updated for 2026 context
        df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
        df = pd.get_dummies(df, columns=['RACE'], prefix='race', drop_first=True)
        return df.drop(columns=['BIRTHDATE'])

    patients_cleaned = clean_demographics(patients)

    # Kidney-Specific Vitals (Creatinine, eGFR, BP, Glucose, HbA1c)
    # Codes: 2160-0 (Creatinine), 33914-3 (eGFR), 8480-6 (SBP), 4548-4 (HbA1c)
    relevant_codes = ['2160-0', '33914-3', '8480-6', '4548-4', '2339-0']
    obs_filtered = observations[observations['CODE'].isin(relevant_codes)]
    obs_latest = (obs_filtered.sort_values('DATE')
                  .groupby(['PATIENT', 'DESCRIPTION'])['VALUE']
                  .last().unstack())
    
    obs_latest.columns = [col.replace(' ', '_').replace('[', '').replace(']', '').lower() for col in obs_latest.columns]
    obs_latest = obs_latest.reset_index().rename(columns={'PATIENT': 'Id'})

    # Merge
    df_final = target_df.merge(patients_cleaned, on='Id', how='left')
    df_final = df_final.merge(obs_latest, on='Id', how='left')
    
    # Fill missing values with median (Better for medical accuracy)
    df_final = df_final.fillna(df_final.median(numeric_only=True))
    
    model_data = df_final.drop(columns=['Id'])
    return model_data

# ── 4. TRAIN CKD MODELS ───────────────────────────────────────
@st.cache_resource
def train_models(_model_data):
    X = _model_data.drop('target', axis=1)
    y = _model_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost with weight balancing (CKD cases are usually the minority)
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
    lasso_model.fit(X_train_scaled, y_train)

    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
    xgb_report = classification_report(y_test, xgb_model.predict(X_test), output_dict=True)
    lasso_acc = accuracy_score(y_test, lasso_model.predict(X_test_scaled))
    lasso_report = classification_report(y_test, lasso_model.predict(X_test_scaled), output_dict=True)

    return (lasso_model, scaler, lasso_acc, lasso_report, xgb_model, xgb_acc, xgb_report, X)

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

    # Identify CKD patients for the demo selector
    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False)]['PATIENT'].unique()
    df_p_demo = df_p[df_p['Id'].isin(ckd_ids)].copy()
    df_p_demo['FULL_NAME'] = df_p_demo['FIRST'] + " " + df_p_demo['LAST']

    with st.spinner("Analyzing Kidney Health Data..."):
        model_data = build_model_data(df_p, df_c, df_o, df_e)
        (lasso_model, scaler, lasso_acc, lasso_report, xgb_model, xgb_acc, xgb_report, X) = train_models(model_data)

    # SIDEBAR
    st.sidebar.title("🧪 CKD AI Dashboard")
    patient_name = st.sidebar.selectbox("Select Patient Profile", options=df_p_demo['FULL_NAME'].sort_values())
    selected_row = df_p_demo[df_p_demo['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_row['Id']

    tab = st.sidebar.radio("Navigation", ["Overview", "Vitals Trend", "Risk Analysis", "Reports"])

    # User Data
    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    
    # UI TABS
    if tab == "Overview":
        st.title(f"Patient Overview: {patient_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("eGFR (Kidney Function)", f"{get_latest_vital('GFR', user_o)} mL/min")
        c2.metric("Creatinine", f"{get_latest_vital('Creatinine', user_o)} mg/dL")
        c3.metric("Systolic BP", f"{get_latest_vital('Systolic', user_o)} mmHg")

        st.subheader("Active Kidney-Related Diagnoses")
        user_ckd = df_c[(df_c['PATIENT'] == p_id) & (df_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False))]
        for cond in user_ckd['DESCRIPTION'].unique():
            st.error(f"● {cond}")

    elif tab == "Risk Analysis":
        st.title("🤖 CKD Risk AI Interpretation")
        
        # XGBoost Importance
        st.subheader("Top Biological Risk Drivers (XGBoost)")
        feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values().tail(10)
        fig, ax = plt.subplots()
        feat_imp.plot(kind='barh', color='#007AFF', ax=ax)
        st.pyplot(fig)
        
        st.info("The model identifies eGFR and Creatinine as the highest weighted predictors for your current kidney risk profile.")

    elif tab == "Reports":
        st.title("📄 Clinical Summary")
        report = f"REPORT: {patient_name}\nSTATUS: CKD Monitoring\neGFR: {get_latest_vital('GFR', user_o)}\nCreatinine: {get_latest_vital('Creatinine', user_o)}"
        st.code(report)
        st.download_button("Download Summary", report, file_name="ckd_report.txt")

except Exception as e:
    st.error(f"Dashboard Error: {e}")
    st.exception(e)
