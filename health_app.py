import streamlit as st
import pandas as pd

# 1. PAGE SETUP
st.set_page_config(page_title="MyHealth Personal Dashboard", page_icon="👤", layout="wide")

# UI Styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #007AFF !important; font-weight: 700; }
    .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    p = pd.read_csv('patients.csv')
    o = pd.read_csv('observations_small.csv') # <--- Added 'o =' here
    c = pd.read_csv('conditions.csv')
    e = pd.read_csv('encounters.csv')
    return p, o, c, e

try:
    df_p, df_o, df_c, df_e = load_data()

    # --- THE HARD CHRONIC FILTER ---
    CHRONIC_LIST = ['Diabetes', 'Hypertension', 'Heart Failure', 'COPD', 'Asthma', 'Kidney Disease', 'Hyperlipidemia', 'Alzheimer', 'Arthritis', 'Prediabetes']
    has_chronic = df_c['DESCRIPTION'].str.contains('|'.join(CHRONIC_LIST), case=False, na=False)
    chronic_patient_ids = df_c[has_chronic]['PATIENT'].unique()
    df_p_chronic = df_p[df_p['Id'].isin(chronic_patient_ids)].copy()
    df_p_chronic['FULL_NAME'] = df_p_chronic['FIRST'] + " " + df_p_chronic['LAST']

    # --- SIDEBAR (Updated for One-Patient Experience) ---
    st.sidebar.title("👤 MyHealth Dashboard")
    
    # Hidden Selection for Demo (Simulates a Login)
    with st.sidebar.expander("🔐 System Login (Demo Only)"):
        patient_name = st.selectbox("Select Profile to Load", options=df_p_chronic['FULL_NAME'].sort_values())
    
    # Identifying the selected patient
    selected_row = df_p_chronic[df_p_chronic['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_row['Id']
    first_name = selected_row['FIRST']
    
    # Once logged in, show the simple account name
    st.sidebar.markdown(f"**Logged in as:** {patient_name}")
    st.sidebar.divider()

    # NEW FEATURE: HEALTH GOAL SIMULATOR
    st.sidebar.subheader("🏃 My Activity Goal")
    exercise_goal = st.sidebar.slider("Weekly Exercise (Minutes)", 0, 300, 150)
    potential_impact = exercise_goal / 30 
    
    # FILE UPLOAD SECTION
    st.sidebar.subheader("📤 My Medical Records")
    uploaded_file = st.sidebar.file_uploader("Add Hospital Visit Summary", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    doc_risk_alert = False
    if uploaded_file is not None:
        st.sidebar.success("Document added!")
        doc_risk_alert = True 

    # --- DATA FETCHING ---
    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    user_e = df_e[df_e['PATIENT'] == p_id].sort_values('START')
    user_c = df_c[df_c['PATIENT'] == p_id]
    chronic_display = user_c[user_c['DESCRIPTION'].str.contains('|'.join(CHRONIC_LIST), case=False, na=False)]

   # NEW VERSION (More flexible)
def get_latest_vital(desc):
    # This searches the 'DESCRIPTION' column for your keyword (like 'Systolic')
    res = user_o[user_o['DESCRIPTION'].str.lower().str.contains(desc.lower(), na=False)]
    if not res.empty:
        return res.iloc[-1]['VALUE']
    return "N/A"

    current_bp = get_latest_vital("Systolic")
    current_gl = get_latest_vital("Glucose")

    tab = st.sidebar.radio("My Navigation", ["Home", "My History", "Health Check", "Doctor Prep", "My Reports"])

    # ---------------------------------------------------------
    # TAB 1: HOME
    # ---------------------------------------------------------
    if tab == "Home":
        st.title(f"👋 Hello, {first_name}!")
        
        if exercise_goal > 0:
            st.info(f"✨ **Health Insight:** By aiming for {exercise_goal} minutes of activity, your individual Blood Pressure risk could drop by {potential_impact:.1f}%!")

        if doc_risk_alert:
            st.warning(f"🔔 **Personal Update:** New data detected in: '{uploaded_file.name}'.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Blood Pressure", f"{current_bp} mmHg")
        m2.metric("Latest Glucose Level", f"{current_gl} mg/dL")
        m3.metric("Conditions Tracked", len(chronic_display))

        st.divider()
        st.subheader("📋 My Tracked Conditions")
        for _, row in chronic_display.iterrows():
            st.success(f"**{row['DESCRIPTION']}**")

    # ---------------------------------------------------------
    # TAB 2: MY HISTORY
    # ---------------------------------------------------------
    elif tab == "My History":
        st.title("🏥 My Medical Visits")
        if not user_e.empty:
            total_cost = user_e['TOTAL_CLAIM_COST'].sum()
            st.metric("Total Healthcare Value", f"${total_cost:,.2f}")
            st.table(user_e[['START', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].tail(10))

    # ---------------------------------------------------------
    # TAB 3: HEALTH CHECK
    # ---------------------------------------------------------
    elif tab == "Health Check":
        st.title("🧘 Daily Check-in")
        q1 = st.checkbox("Any trouble breathing?")
        q2 = st.checkbox("Increased thirst?")
        if q1 or q2:
            st.error("🚨 **Alert:** Please contact your care team immediately.")
        else:
            st.success("✅ Your current symptoms appear stable.")

    # ---------------------------------------------------------
    # TAB 4: DOCTOR PREP
    # ---------------------------------------------------------
    elif tab == "Doctor Prep":
        st.title("🏥 My Visit Planner")
        st.info(f"1. How does my BP of {current_bp} look for my profile?")
        st.info(f"2. Does my activity goal of {exercise_goal} mins help my glucose trend?")
        if doc_risk_alert:
            st.info(f"3. Let's discuss the findings in my report: {uploaded_file.name}")

    # ---------------------------------------------------------
    # TAB 5: MY REPORTS
    # ---------------------------------------------------------
    elif tab == "My Reports":
        st.title("📄 Export My Health Summary")
        st.write("Download a summary of your California health records to share with your doctor.")
        
        report_data = f"""
        MYHEALTH DASHBOARD REPORT
        --------------------------
        Patient: {patient_name}
        Location: {selected_row['CITY']}, CA
        
        LATEST VITALS:
        - Blood Pressure: {current_bp} mmHg
        - Glucose: {current_gl} mg/dL
        
        ACTIVE CHRONIC CONDITIONS:
        {', '.join(chronic_display['DESCRIPTION'].tolist()) if not chronic_display.empty else 'No active chronic conditions.'}
        
        Weekly Exercise Goal: {exercise_goal} minutes
        
        Generated on: 2026-03-16
        """
        
        st.text_area("Preview your report:", report_data, height=250)
        
        st.download_button(
            label="📥 Download as Text File",
            data=report_data,
            file_name=f"{first_name}_Health_Summary.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Dashboard Error: {e}")
