import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import joblib
import json
from datetime import datetime

# Paths
raw_file = "/content/drive/MyDrive/clinical-trial-outcome-prediction/data/clinical_trials_raw.csv"
processed_dir = "/content/drive/MyDrive/clinical-trial-outcome-prediction/data/processed"
os.makedirs(processed_dir, exist_ok=True)

# Load raw data
df = pd.read_csv(raw_file)
print("Loaded raw data:", df.shape)

# 1. Create outcome labels
def outcome_label(status, why=None):
    if pd.isna(status):
        return None
    stat = str(status).lower()
    if any(s in stat for s in ['completed', 'active, not recruiting', 'enrolling by invitation', 'recruiting']):
        return 1  # Success
    if any(s in stat for s in ['terminated', 'withdrawn', 'suspended', 'no longer', 'temporarily']):
        return 0  # Failure
    if pd.notna(why) and str(why).strip():
        return 0  # Any stopped with a reason = likely failure
    return None

df['outcome'] = df.apply(lambda row: outcome_label(row.get('overall_status'), row.get('why_stopped')), axis=1)
df = df.dropna(subset=['outcome'])
df['outcome'] = df['outcome'].astype(int)
print(
    f"Prepared outcome labels (final count: {len(df)} | fail: {(df['outcome']==0).sum()}, success: {(df['outcome']==1).sum()})"
)

# 2. Clean and fuse text fields
def clean(txt):
    if pd.isna(txt):
        return ""
    txt = str(txt).lower()
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = ' '.join(txt.split())
    return txt

for col in ['brief_title', 'official_title', 'brief_summary', 'detailed_description', 'eligibility_criteria']:
    if col in df.columns:
        df[f'{col}_clean'] = df[col].apply(clean)

text_cols = [c for c in df.columns if c.endswith('_clean')]
df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
df['combined_text'] = df['combined_text'].str[:1000]   # Truncate if very long

# 3. Numeric and categorical features
numerical = ['enrollment_count']
for n in numerical:
    if n in df.columns:
        df[n] = pd.to_numeric(df[n], errors='coerce').fillna(0)

date_cols = ['start_date', 'completion_date', 'primary_completion_date']
for d in date_cols:
    if d in df.columns:
        df[f"{d}_year"] = pd.to_datetime(df[d], errors='coerce').dt.year.fillna(2020)

categorical = [
    'study_type', 'phases', 'allocation', 'intervention_model',
    'masking', 'primary_purpose', 'gender', 'lead_sponsor_class'
]
label_encoders = {}
for col in categorical:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# 4. Derived features
if 'start_date_year' in df.columns and 'completion_date_year' in df.columns:
    df['trial_duration_years'] = df['completion_date_year'] - df['start_date_year']
    df['trial_duration_years'] = df['trial_duration_years'].clip(lower=0, upper=20)

if 'phases' in df.columns:
    df['is_multi_phase'] = df['phases'].astype(str).str.contains('\|').astype(int)
if 'location_countries' in df.columns:
    df['is_international'] = df['location_countries'].astype(str).str.contains('\|').astype(int)

# 5. Final features
structured = [
    'enrollment_count', 'start_date_year', 'completion_date_year',
    'primary_completion_date_year', 'trial_duration_years',
    'study_type_encoded', 'phases_encoded', 'allocation_encoded',
    'intervention_model_encoded', 'masking_encoded', 'primary_purpose_encoded',
    'gender_encoded', 'lead_sponsor_class_encoded', 'is_multi_phase', 'is_international'
]
structured = [c for c in structured if c in df.columns]
for s in structured:
    if df[s].dtype in ['float64', 'int64']:
        df[s] = df[s].fillna(df[s].median())
    else:
        df[s] = df[s].fillna(0)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
processed_csv = f"{processed_dir}/processed_clinical_trials_{timestamp}.csv"
df.to_csv(processed_csv, index=False)

# Save feature metadata and encoders
feature_groups = {
    'text': ['combined_text'],
    'structured': structured,
    'target': 'outcome',
    'identifier': 'nct_id'
}
with open(f"{processed_dir}/feature_groups_{timestamp}.json", 'w') as f:
    json.dump(feature_groups, f, indent=2)
joblib.dump(label_encoders, f"{processed_dir}/label_encoders_{timestamp}.pkl")

print("✅ Data preprocessing completed!")
print(f"Processed CSV: {processed_csv}")
print(f"Features Spec: {processed_dir}/feature_groups_{timestamp}.json")
print(f"Encoders:      {processed_dir}/label_encoders_{timestamp}.pkl")
