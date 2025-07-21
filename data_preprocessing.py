import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json

class ClinicalTrialPreprocessor:
    def __init__(self, raw_file="data/raw_clinical_trials.csv", out_dir="data/processed"):
        self.raw_file = Path(raw_file)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}

    def load_raw_data(self):
        df = pd.read_csv(self.raw_file)
        print(f"Loaded raw data: {df.shape}")
        return df

    def create_outcome_labels(self, df):
        successful = ['Completed', 'Active, not recruiting', 'Enrolling by invitation', 'Recruiting']
        failed = ['Terminated', 'Withdrawn', 'Suspended', 'No longer available', 'Temporarily not available']

        def outcome(status, why_stopped):
            if pd.isna(status):
                return None
            status_str = str(status).lower()
            if any(s.lower() in status_str for s in successful):
                return 1
            elif any(s.lower() in status_str for s in failed):
                return 0
            elif pd.notna(why_stopped) and str(why_stopped).strip():
                return 0
            return None

        df['outcome'] = df.apply(lambda row: outcome(row['overall_status'], row.get('why_stopped', None)), axis=1)
        df = df.dropna(subset=['outcome'])
        print(f"Labelled outcomes: {df.shape}")
        return df

    def preprocess_text_fields(self, df):
        def clean(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            return re.sub(r"[^\w\s]", " ", text).strip()
        for field in ['brief_title', 'official_title', 'brief_summary', 'detailed_description', 'eligibility_criteria']:
            if field in df:
                df[f"{field}_clean"] = df[field].fillna("").apply(clean)
        text_fields = [f+"_"+"clean" for f in ['brief_title', 'official_title', 'brief_summary', 'detailed_description', 'eligibility_criteria']]
        df['combined_text'] = df[text_fields].fillna("").agg(" ".join, axis=1)
        print("Processed text fields.")
        return df

    def preprocess_structured(self, df):
        # Example only: Add more features as needed
        if 'enrollment_count' in df:
            df['enrollment_count'] = pd.to_numeric(df['enrollment_count'], errors='coerce').fillna(0)
        for field in ['study_type', 'phases', 'allocation', 'intervention_model', 'masking', 'primary_purpose', 'gender', 'lead_sponsor_class']:
            if field in df:
                df[field] = df[field].fillna('Unknown')
                le = LabelEncoder()
                df[f"{field}_encoded"] = le.fit_transform(df[field].astype(str))
                self.label_encoders[field] = le
        print("Structured features encoded.")
        return df

    def finalize_and_save(self, df):
        out_file = self.out_dir / "processed_clinical_trials.csv"
        df.to_csv(out_file, index=False)
        with open(self.out_dir / "label_encoders.json", 'w') as f:
            json.dump({k: list(le.classes_) for k, le in self.label_encoders.items()}, f)
        print(f"Saved processed data to {out_file}")

def run_preprocessing():
    processor = ClinicalTrialPreprocessor()
    df = processor.load_raw_data()
    df = processor.create_outcome_labels(df)
    df = processor.preprocess_text_fields(df)
    df = processor.preprocess_structured(df)
    processor.finalize_and_save(df)

if __name__ == "__main__":
    run_preprocessing()
