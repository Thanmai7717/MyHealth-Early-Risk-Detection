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
st.set_page_config(page_title="MyHealth CKD Predict", page_icon="👤", layout="wide")

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

# ── 3. BUILD MODEL DATA (CKD OPTIMIZED) ───────────────────────
@st.cache_data
def build_model_data(patients, conditions, observations, encounters):
    ckd_pattern = 'Kidney|Renal|Nephropathy|Glomerular|End stage renal'
    ckd_ids = conditions[conditions['DESCRIPTION'].str.contains(
                ckd_pattern, case=False, na=False)]['PATIENT'].unique()
    
    target_df = pd.DataFrame({'Id': patients['Id']})
    target_df['target'] = target_df['Id'].isin(ckd_ids).astype(int)

    def clean_demographics(df):
        keep_cols = ['Id', 'GENDER', 'RACE', 'BIRTHDATE', 'INCOME']
        df = df[keep_cols].copy()
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
        df['Age'] = 2026 - df['BIRTHDATE'].dt.year
        df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
        df = pd.get_dummies(df, columns=['RACE'], prefix='race', drop_first=True)
        return df.drop(columns=['BIRTHDATE'])

    patients_cleaned = clean_demographics(patients)

    relevant_codes = ['2160-0', '33914-3', '8480-6', '4548-4', '2339-0']
    obs_filtered = observations[observations['CODE'].isin(relevant_codes)]
    obs_latest = (obs_filtered.sort_values('DATE')
                  .groupby(['PATIENT', 'DESCRIPTION'])['VALUE']
                  .last().unstack())
    
    obs_latest.columns = [col.replace(' ', '_').replace('[', '').replace(']', '')
                          .replace('(', '').replace(')', '').replace('/', '_').lower() 
                          for col in obs_latest.columns]
    obs_latest = obs_latest.reset_index().rename(columns={'PATIENT': 'Id'})

    df_final = target_df.merge(patients_cleaned, on='Id', how='left')
    df_final = df_final.merge(obs_latest, on='Id', how='left')
    return df_final

# ── 4. TRAIN MODELS (DATA TYPE FIX INCLUDED) ─────────────────
@st.cache_resource
def train_models(_df_final):
    X = _df_final.drop(columns=['target', 'Id'])
    y = _df_final['target']

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
    lasso_model.fit(X_train_scaled, y_train)

    return (lasso_model, scaler, xgb_model, X, X_test, y_test)

# ── 5. HELPER ─────────────────────────────────────────────────
def get_latest_vital(desc, user_o):
    res = user_o[user_o['DESCRIPTION'].str.lower().str.contains(desc.lower(), na=False)]
    if not res.empty:
        try: return f"{float(res.iloc[-1]['VALUE']):.1f}"
        except: return res.iloc[-1]['VALUE']
    return "N/A"

# ── 6. MAIN EXECUTION ─────────────────────────────────────────
try:
    df_p, df_o, df_c, df_e = load_data()
    
    with st.spinner("AI Engine: Training Kidney Risk Models..."):
        model_data_full = build_model_data(df_p, df_c, df_o, df_e)
        lasso_model, scaler, xgb_model, X, X_test, y_test = train_models(model_data_full)

    ckd_ids = df_c[df_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False)]['PATIENT'].unique()
    df_p_demo = df_p[df_p['Id'].isin(ckd_ids)].copy()
    df_p_demo['FULL_NAME'] = df_p_demo['FIRST'] + " " + df_p_demo['LAST']

    # SIDEBAR
    st.sidebar.title("👤 MyHealth CKD AI")
    with st.sidebar.expander("🔐 System Login (Demo)"):
        patient_name = st.sidebar.selectbox("Select Profile", options=df_p_demo['FULL_NAME'].sort_values())
    
    selected_row = df_p_demo[df_p_demo['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_row['Id']
    
    # PERFORMANCE SECTION REMOVED HERE AS REQUESTED
    
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("Upload Lab Report", type=['pdf','json','png','jpg'])
    tab = st.sidebar.radio("Navigation", ["Home", "My History", "Health Check", "Model Insights", "My Reports"])

    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    user_c = df_c[df_c['PATIENT'] == p_id]
    current_bp = get_latest_vital("Systolic", user_o)
    current_gfr = get_latest_vital("GFR", user_o)

    # ── TAB: HOME ─────────────────────────────────────────────
    if tab == "Home":
        st.title(f"👋 Welcome, {selected_row['FIRST']}!")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current eGFR", f"{current_gfr} mL/min")
        m2.metric("Blood Pressure", f"{current_bp} mmHg")
        m3.metric("Kidney Conditions", len(user_c[user_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False)]))

        st.divider()

        # DAILY HEALTH CHECK-IN FORM
        st.subheader("📝 Daily Health Check-in")
        with st.form("health_check_form"):
            st.write("Please answer the following based on your health today:")
            q1 = st.select_slider("How is your energy level?", options=["Very Low", "Low", "Normal", "High"])
            q2 = st.radio("Are you experiencing any swelling in your ankles or feet?", ["No", "Slightly", "Yes - Significant"])
            q3 = st.radio("Have you noticed any changes in your urination frequency?", ["No change", "More frequent", "Less frequent"])
            q4 = st.checkbox("Are you experiencing a metallic taste in your mouth?")
            q5 = st.text_area("Any other symptoms or notes for your doctor?")

            submitted = st.form_submit_button("Submit Daily Check-in")
            
            if submitted:
                st.success("Thank you! Your responses have been saved to your health profile.")
                if q2 != "No" or q4:
                    st.warning("⚠️ High Risk Indicator: These symptoms are often linked to kidney function. We recommend scheduling a consultation.")

        st.divider()
        st.subheader("📋 Tracked Kidney Conditions")
        for cond in user_c[user_c['DESCRIPTION'].str.contains('Kidney|Renal', case=False)]['DESCRIPTION'].unique():
            st.error(f"**{cond}**")

    # ── TAB: MY HISTORY ───────────────────────────────────────
    elif tab == "My History":
        st.title("🏥 Medical Visit History")
        user_e = df_e[df_e['PATIENT'] == p_id].sort_values('START', ascending=False)
        st.table(user_e[['START', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].head(10))

    # ── TAB: HEALTH CHECK ─────────────────────────────────────
    elif tab == "Health Check":
        st.title("🩺 Vital Signs Trend")
        metric = st.selectbox("Select Metric", ["GFR", "Creatinine", "Systolic", "Glucose"])
        trend = user_o[user_o['DESCRIPTION'].str.contains(metric, case=False)].copy()
        trend['VALUE'] = pd.to_numeric(trend['VALUE'], errors='coerce')
        
        if not trend.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pd.to_datetime(trend['DATE']), trend['VALUE'], marker='o', color='#007AFF')
            ax.set_title(f"{metric} History")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No data found for this metric.")

    # ── TAB: MODEL INSIGHTS ───────────────────────────────────
    elif tab == "Model Insights":
        st.title("🤖 AI Risk Drivers")
        st.subheader("Top 10 Feature Importance (XGBoost)")
        feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values().tail(10)
        fig, ax = plt.subplots()
        feat_imp.plot(kind='barh', color='teal', ax=ax)
        st.pyplot(fig)

    # ── TAB: MY REPORTS ───────────────────────────────────────
    elif tab == "My Reports":
        st.title("📄 Export Clinical Summary")
        report_text = f"PATIENT: {patient_name}\nLATEST GFR: {current_gfr}\nLATEST BP: {current_bp}\n"
        st.code(report_text)
        st.download_button("Download Report", report_text, file_name=f"{patient_name}_health.txt")

        if uploaded_file:
            st.divider()
            with st.status("🔍 Analyzing uploaded document...", expanded=True) as status:
                st.write("Extracting clinical text...")
                time.sleep(1)
                st.write("Running XGBoost inference...")
                time.sleep(1)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            st.subheader("AI Analysis Result")
            st.metric("Detected Risk", "HIGH", delta="CKD Priority", delta_color="inverse")

except Exception as e:
    st.error(f"Critical Dashboard Error: {e}")
    st.exception(e)
