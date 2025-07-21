# data_preprocessing.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib
import os

class ClinicalTrialPreprocessor:
    def __init__(self, input_csv, output_dir="data/processed"):
        self.raw_path = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label_encoders = {}

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def create_outcome_labels(self, df):
        success = ['Completed', 'Active, not recruiting']
        fail = ['Terminated', 'Withdrawn', 'Suspended']
        def label_outcome(status):
            if status in success:
                return 1
            elif status in fail:
                return 0
            else:
                return np.nan
        df['outcome'] = df['statusModule.overallStatus'].apply(label_outcome)
        df = df.dropna(subset=['outcome'])
        return df

    def process_structured(self, df):
        df['enrollment'] = pd.to_numeric(df.get('designModule.enrollmentInfo.count', 0), errors='coerce').fillna(0)

        for col in [
            "designModule.studyType",
            "designModule.designInfo.allocation",
            "designModule.designInfo.primaryPurpose"
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df

    def process_text(self, df):
        text_cols = [
            "identificationModule.briefTitle",
            "descriptionModule.briefSummary",
            "descriptionModule.detailedDescription"
        ]
        for col in text_cols:
            df[col] = df[col].fillna("").apply(self.clean_text)
        df["combined_text"] = df[text_cols].apply(lambda row: " ".join(row.values), axis=1)
        return df

    def run(self):
        df = pd.read_csv(self.raw_path)
        df = self.create_outcome_labels(df)
        df = self.process_structured(df)
        df = self.process_text(df)

        output_path = self.output_dir / f"processed_trials_{self.timestamp}.csv"
        df.to_csv(output_path, index=False)

        encoders_path = self.output_dir / f"label_encoders_{self.timestamp}.pkl"
        joblib.dump(self.label_encoders, encoders_path)

        print(f"✅ Preprocessed data saved: {output_path}")
        print(f"✅ Label encoders saved: {encoders_path}")
        return output_path
