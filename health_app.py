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
    df_final = df_final.merge(
