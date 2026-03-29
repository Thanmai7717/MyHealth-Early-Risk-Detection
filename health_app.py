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

# 2. FUNCTIONS (THE TOOLBOX)
@st.cache_data
def load_data():
    p = pd.read_csv('patients.csv')
    # Using the compressed file you just made!
    o = pd.read_csv('observations.csv.gz', compression='gzip') 
    c = pd.read_csv('conditions.csv')
    e = pd.read_csv('encounters.csv')
    return p, o, c, e

def get_latest_vital(desc, user_o):
    # Flexible search for vitals
    res = user_o[user_o['DESCRIPTION'].str.lower().str.contains(desc.lower(), na=False)]
    if not res.empty:
        val = res.iloc[-1]['VALUE']
        try:
            return f"{float(val):.1f}"
        except:
            return val
    return "N/A"

# 3. MAIN APP LOGIC
try:
    df_p, df_o, df_c, df_e = load_data()

    # --- THE HARD CHRONIC FILTER ---
    CHRONIC_LIST = ['Diabetes', 'Hypertension', 'Heart Failure', 'COPD', 'Asthma', 'Kidney Disease', 'Hyperlipidemia', 'Alzheimer', 'Arthritis', 'Prediabetes']
    has_chronic = df_c['DESCRIPTION'].str.contains('|'.join(CHRONIC_LIST), case=False, na=False)
    chronic_patient_ids = df_c[has_chronic]['PATIENT'].unique()
    df_p_chronic = df_p[df_p['Id'].isin(chronic_patient_ids)].copy()
    df_p_chronic['FULL_NAME'] = df_p_chronic['FIRST'] + " " + df_p_chronic['LAST']

    # --- SIDEBAR ---
    st.sidebar.title("👤 MyHealth Dashboard")
    
    with st.sidebar.expander("🔐 System Login (Demo Only)"):
        patient_name = st.sidebar.selectbox("Select Profile", options=df_p_chronic['FULL_NAME'].sort_values())
    
    selected_row = df_p_chronic[df_p_chronic['FULL_NAME'] == patient_name].iloc[0]
    p_id = selected_row['Id']
    first_name = selected_row['FIRST']
    
    st.sidebar.markdown(f"**Logged in as:** {patient_name}")
    st.sidebar.divider()

    exercise_goal = st.sidebar.slider("Weekly Exercise (Minutes)", 0, 300, 150)
    potential_impact = exercise_goal / 30 

    uploaded_file = st.sidebar.file_uploader("Add Hospital Visit Summary", type=['pdf', 'png', 'jpg', 'jpeg'])
    doc_risk_alert = True if uploaded_file else False

    # --- DATA FETCHING FOR SELECTED PATIENT ---
    user_o = df_o[df_o['PATIENT'] == p_id].sort_values('DATE')
    user_e = df_e[df_e['PATIENT'] == p_id].sort_values('START')
    user_c = df_c[df_c['PATIENT'] == p_id]
    chronic_display = user_c[user_c['DESCRIPTION'].str.contains('|'.join(CHRONIC_LIST), case=False, na=False)]

    # Use the function to get vitals
    current_bp = get_latest_vital("Systolic", user_o)
    current_gl = get_latest_vital("Glucose", user_o)

    tab = st.sidebar.radio("My Navigation", ["Home", "My History", "Health Check", "Doctor Prep", "My Reports"])

    # ---------------------------------------------------------
    # TAB 1: HOME
    # ---------------------------------------------------------
    if tab == "Home":
        st.title(f"👋 Hello, {first_name}!")
        
        if exercise_goal > 0:
            st.info(f"✨ **Health Insight:** By aiming for {exercise_goal} minutes of activity, your individual Blood Pressure risk could drop by {potential_impact:.1f}%!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Blood Pressure", f"{current_bp} mmHg")
        m2.metric("Latest Glucose Level", f"{current_gl} mg/dL")
        m3.metric("Conditions Tracked", len(chronic_display))

        st.divider()
        st.subheader("📋 My Tracked Conditions")
        for _, row in chronic_display.iterrows():
            st.success(f"**{row['DESCRIPTION']}**")

    # (Other tabs like 'My History' would go here - keeping it simple for your fix)
    elif tab == "My History":
        st.title("🏥 My Medical Visits")
        st.table(user_e[['START', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].tail(10))

except Exception as e:
    st.error(f"Dashboard Error: {e}")
