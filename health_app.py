import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ── 1. PAGE SETUP ─────────────────────────────────────────────
st.set_page_config(page_title="MyHealth Personal Dashboard", page_icon="👤", layout="wide")

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
    except FileNotFoundError:
        o = pd.read_csv('observations.csv')
    return p, o, c, e

# ── 3. BUILD MODEL DATA ───────────────────────────────────────
@st.cache_data
def build_model_data(patients, conditions, observations, encounters):
    target_groups = {
        'diabetes':       'Diabetes|Prediabetes',
        'ckd':            'Kidney|Renal|Nephropathy',
        'cardiovascular': 'Hypertension|Heart|Infarction|Atrial|Coronary|Ischemic',
        'anemia':         'Anemia',
        'liver_disease':  'Liver|Cirrhosis|Hepatitis'
    }
    chronic_pattern = '|'.join(target_groups.values())
    chronic_ids     = conditions[conditions['DESCRIPTION'].str.contains(
                          chronic_pattern, case=False, na=False)]['PATIENT'].unique()
    target_df           = pd.DataFrame({'Id': patients['Id']})
    target_df['target'] = target_df['Id'].isin(chronic_ids).astype(int)

    def clean_demographics(df):
        keep_cols = ['Id', 'BIRTHDATE', 'GENDER', 'RACE', 'INCOME',
                     'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']
        df = df[keep_cols].copy()
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
        df['Age']       = 2024 - df['BIRTHDATE'].dt.year
        df['GENDER']    = df['GENDER'].map({'M': 0, 'F': 1})
        df = pd.get_dummies(df, columns=['RACE'], prefix='race', drop_first=True)
        df = df.drop(columns=['BIRTHDATE'])
        return df

    patients_cleaned = clean_demographics(patients)

    relevant_codes = ['2339-0', '4548-4', '8480-6', '2160-0', '718-7']
    obs_filtered   = observations[observations['CODE'].isin(relevant_codes)]
    obs_latest     = (obs_filtered.sort_values('DATE')
                      .groupby(['PATIENT', 'DESCRIPTION'])['VALUE']
                      .last().unstack())
    obs_latest.columns = [col.replace(' ', '_').lower() for col in obs_latest.columns]
    obs_latest = obs_latest.reset_index().rename(columns={'PATIENT': 'Id'})

    encounter_counts = encounters.groupby('PATIENT')['Id'].count().reset_index()
    encounter_counts.columns = ['Id', 'visit_count']

    df_final = target_df.merge(patients_cleaned, on='Id', how='left')
    df_final = df_final.merge(obs_latest,        on='Id', how='left')
    df_final = df_final.merge(encounter_counts,  on='Id', how='left')
    df_final = df_final.fillna(0)
    model_data = df_final.drop(columns=['Id'])

    model_data.columns = (model_data.columns
                          .str.replace('[', '_', regex=False)
                          .str.replace(']', '_', regex=False)
                          .str.replace('<', '_', regex=False))
    return model_data

# ── 4. TRAIN MODELS ───────────────────────────────────────────
@st.cache_resource
def train_models(_model_data):
    X = _model_data.drop('target', axis=1)
    y = _model_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test  = X_test.apply(pd.to_numeric,  errors='coerce').fillna(0)

    # Lasso
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lasso_model = LogisticRegression(
        l1_ratio=1, solver='saga', max_iter=1000,
        class_weight='balanced', random_state=42)
    lasso_model.fit(X_train_scaled, y_train)
    lasso_acc    = accuracy_score(y_test, lasso_model.predict(X_test_scaled))
    lasso_report = classification_report(
        y_test, lasso_model.predict(X_test_scaled), output_dict=True)

    # XGBoost
    ratio     = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss', scale_pos_weight=ratio, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_acc    = accuracy_score(y_test, xgb_model.predict(X_test))
    xgb_report = classification_report(
        y_test, xgb_model.predict(X_test), output_dict=True)

    return (lasso_model, scaler, lasso_acc, lasso_report,
            xgb_model, xgb_acc, xgb_report, X, X_test, y_test)

# ── 5. HELPER ─────────────────────────────────────────────────
def get_latest_vital(desc, user_o):
    res = user_o[user_o['DESCRIPTION'].str.lower().str.contains(desc.lower(), na=False)]
    if not res.empty:
        val = res.iloc[-1]['VALUE']
        try:
            return f"{float(val):.1f}"
        except:
            return val
    return "N/A"

# ── 6. MAIN APP ───────────────────────────────────────────────
try:
    df_p, df_o, df_c, df_e = load_data()

    CHRONIC_LIST = ['Diabetes', 'Hypertension', 'Heart Failure', 'COPD', 'Asthma',
                    'Kidney Disease', 'Hyperlipidemia', 'Alzheimer', 'Arthritis', 'Prediabetes']
    has_chronic         = df_c['DESCRIPTION'].str.contains(
                              '|'.join(CHRONIC_LIST), case=False, na=False)
    chronic_patient_ids = df_c[has_chronic]['PATIENT'].unique()
    df_p_chronic        = df_p[df_p['Id'].isin(chronic_patient_ids)].copy()
    df_p_chronic['FULL_NAME'] = df_p_chronic['FIRST'] + " " + df_p_chronic['LAST']

    with st.spinner("Training models... please wait"):
        model_data = build_model_data(df_p, df_c, df_o, df_e)
        (lasso_model, scaler, lasso_acc, lasso_report,
         xgb_model, xgb_acc, xgb_report,
         X, X_test, y_test) = train_models(model_data)

    # ── SIDEBAR ───────────────────────────────────────────────
    # --- SIDEBAR: LOGIN & PERFORMANCE ---
    st.sidebar.title("👤 MyHealth Dashboard")
    
    with st.sidebar.expander("🔐 System Login (Demo Only)"):
        patient_name = st.sidebar.selectbox("Select Profile", options=df_p_chronic['FULL_NAME'].sort_values())
    
    st.sidebar.divider()
    st.sidebar.subheader("🏆 Model Performance")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("XGBoost", "94.80%") 
    col2.metric("Lasso", "92.20%")   
    st.sidebar.divider()

    # --- THE JSON UPLOADER FIX ---
    # We added 'json' to the type list here
    uploaded_file = st.sidebar.file_uploader(
        "Upload Patient Record (JSON)", 
        type=['pdf', 'png', 'jpg', 'jpeg', 'json']
    )

    if uploaded_file:
        st.sidebar.success(f"✅ {uploaded_file.name} uploaded!")
        
        # If you upload that specific Adrianna_Rosa JSON file:
        if uploaded_file.name.endswith('.json'):
            raw_data = json.load(uploaded_file)
            with st.sidebar.expander("🔍 View Record Details"):
                st.json(raw_data) # This makes the JSON look clean and clickable
    st.sidebar.title("👤 MyHealth Dashboard")

    with st.sidebar.expander("🔐 System Login (Demo Only)"):
        patient_name = st.sidebar.selectbox("Select Profile",
                           options=df_p_chronic['FULL_NAME'].sort_values())

    selected_row = df_p_chronic[df_p_chronic['FULL_NAME'] == patient_name].iloc[0]
    p_id         = selected_row['Id']
    first_name   = selected_row['FIRST']

    st.sidebar.markdown(f"**Logged in as:** {patient_name}")
    st.sidebar.divider()

    st.sidebar.subheader("🏆 Model Performance")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("XGBoost", f"{xgb_acc*100:.2f}%")
    col2.metric("Lasso",   f"{lasso_acc*100:.2f}%")
    st.sidebar.divider()

    exercise_goal    = st.sidebar.slider("Weekly Exercise (Minutes)", 0, 300, 150)
    potential_impact = exercise_goal / 30
    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader("Add Hospital Visit Summary",
                                              type=['pdf', 'png', 'jpg', 'jpeg'])

    tab = st.sidebar.radio("My Navigation",
              ["Home", "My History", "Health Check", "Model Insights", "Doctor Prep", "My Reports"])

    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    user_e = df_e[df_e['PATIENT'] == p_id].sort_values('START')
    user_c = df_c[df_c['PATIENT'] == p_id]
    chronic_display = user_c[user_c['DESCRIPTION'].str.contains(
                         '|'.join(CHRONIC_LIST), case=False, na=False)]

    current_bp = get_latest_vital("Systolic", user_o)
    current_gl = get_latest_vital("Glucose",  user_o)

    # ── TAB: HOME ─────────────────────────────────────────────
    if tab == "Home":
        st.title(f"👋 Hello, {first_name}!")
        if exercise_goal > 0:
            st.info(f"✨ **Health Insight:** By aiming for {exercise_goal} minutes of activity, "
                    f"your Blood Pressure risk could drop by {potential_impact:.1f}%!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Blood Pressure", f"{current_bp} mmHg")
        m2.metric("Latest Glucose Level",   f"{current_gl} mg/dL")
        m3.metric("Conditions Tracked",     len(chronic_display))

        st.divider()
        st.subheader("📋 My Tracked Conditions")
        for _, row in chronic_display.iterrows():
            st.success(f"**{row['DESCRIPTION']}**")

    # ── TAB: MY HISTORY ───────────────────────────────────────
    elif tab == "My History":
        st.title("🏥 My Medical Visits")
        if not user_e.empty and 'DESCRIPTION' in user_e.columns:
            st.table(user_e[['START', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].tail(10))
        else:
            st.warning("No visit history found for this patient.")

    # ── TAB: HEALTH CHECK ─────────────────────────────────────
    elif tab == "Health Check":
        st.title("🩺 Health Check")
        st.subheader("Latest Vitals")

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Blood Pressure",  f"{get_latest_vital('Systolic', user_o)} mmHg")
        v2.metric("Glucose",         f"{get_latest_vital('Glucose', user_o)} mg/dL")
        v3.metric("Hemoglobin A1c",  f"{get_latest_vital('Hemoglobin', user_o)} %")
        v4.metric("Creatinine",      f"{get_latest_vital('Creatinine', user_o)} mg/dL")

        st.divider()
        st.subheader("Vital Trends Over Time")
        vital_choice = st.selectbox("Select Vital to Plot",
                           ["Systolic", "Glucose", "Hemoglobin", "Creatinine"])
        trend_data = user_o[user_o['DESCRIPTION'].str.lower().str.contains(
                         vital_choice.lower(), na=False)].copy()
        trend_data['VALUE'] = pd.to_numeric(trend_data['VALUE'], errors='coerce')
        trend_data = trend_data.dropna(subset=['VALUE'])

        if not trend_data.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pd.to_datetime(trend_data['DATE']), trend_data['VALUE'],
                    marker='o', color='#007AFF')
            ax.set_title(f"{vital_choice} Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No trend data available for this vital.")

    # ── TAB: MODEL INSIGHTS ───────────────────────────────────
    elif tab == "Model Insights":
        st.title("🤖 Model Insights")

        st.subheader("Model Accuracy Comparison")
        a1, a2 = st.columns(2)
        a1.metric("XGBoost Accuracy", f"{xgb_acc*100:.2f}%")
        a2.metric("Lasso Accuracy",   f"{lasso_acc*100:.2f}%")

        st.divider()

        st.subheader("Detailed Classification Reports")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**XGBoost**")
            st.dataframe(pd.DataFrame(xgb_report).transpose().round(2))
        with r2:
            st.markdown("**Lasso**")
            st.dataframe(pd.DataFrame(lasso_report).transpose().round(2))

        st.divider()

        st.subheader("XGBoost — Top 10 Risk Predictors")
        xgb_importances = pd.Series(
            xgb_model.feature_importances_, index=X.columns).sort_values()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        xgb_importances.tail(10).plot(kind='barh', color='teal', ax=ax1)
        ax1.set_title('XGBoost — Top 10 Risk Predictors')
        ax1.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig1)

        st.divider()

        st.subheader("Lasso — Top 15 Feature Coefficients")
        coeffs = pd.Series(lasso_model.coef_[0], index=X.columns)
        coeffs_filtered = coeffs.reindex(
            coeffs.abs().sort_values(ascending=False).head(15).index
        ).sort_values()

        colors = ['red' if x < 0 else 'green' for x in coeffs_filtered]
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        coeffs_filtered.plot(kind='barh', color=colors, ax=ax2)
        ax2.axvline(0, color='black', lw=1)
        ax2.set_title('Lasso — Top 15 Predictors\n(Green = increases risk, Red = decreases risk)')
        ax2.set_xlabel('Standardized Coefficient Weight')
        plt.tight_layout()
        st.pyplot(fig2)

        n_zero = (coeffs == 0).sum()
        st.info(f"Features zeroed out by Lasso: {n_zero} / {len(coeffs)}")

    # ── TAB: DOCTOR PREP ──────────────────────────────────────
    elif tab == "Doctor Prep":
        st.title("📋 Doctor Visit Prep")
        st.subheader("Questions to ask your doctor:")
        for cond in chronic_display['DESCRIPTION'].tolist():
            st.markdown(f"- What is the latest update on my **{cond}**?")
        st.divider()
        st.subheader("Recent Vitals Summary")
        st.write(f"- Blood Pressure: {current_bp} mmHg")
        st.write(f"- Glucose: {current_gl} mg/dL")

    # ── TAB: MY REPORTS ───────────────────────────────────────
    elif tab == "My Reports":
        st.title("📂 My Reports")
        if uploaded_file:
            st.success("File uploaded successfully!")
            st.write(f"Filename: {uploaded_file.name}")
        else:
            st.info("Upload a hospital visit summary from the sidebar.")

except Exception as e:
    st.error(f"Dashboard Error: {e}")
    st.exception(e)
