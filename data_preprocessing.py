import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Directory to save processed data (define this in your config or main script)
processed_dir = "data/processed"

class ClinicalTrialPreprocessor:
    """Comprehensive preprocessing for clinical trial data"""

    def __init__(self, output_dir=processed_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"🔧 Clinical Trial Preprocessor initialized")
        print(f"📂 Output directory: {self.output_dir}")

    def create_outcome_labels(self, df):
        """Create binary outcome labels from trial status with enhanced logic"""
        print("🎯 CREATING OUTCOME LABELS")
        print("-" * 40)

        # Define successful/failed outcomes
        successful_statuses = [
            'Completed', 'Active, not recruiting', 'Enrolling by invitation', 'Recruiting'
        ]
        failed_statuses = [
            'Terminated', 'Withdrawn', 'Suspended',
            'No longer available', 'Temporarily not available'
        ]

        def categorize_outcome(status, why_stopped=None):
            """Categorize trial outcome based on status and reason"""
            if pd.isna(status):
                return None
            status_str = str(status).lower()
            if any(success.lower() in status_str for success in successful_statuses):
                return 1
            elif any(fail.lower() in status_str for fail in failed_statuses):
                return 0
            elif pd.notna(why_stopped) and str(why_stopped).strip():
                return 0
            else:
                return None  # Unknown/Ongoing

        df['outcome'] = df.apply(
            lambda row: categorize_outcome(row['overall_status'], row.get('why_stopped')),
            axis=1
        )
        # Remove trials with unknown outcomes
        df_clean = df.dropna(subset=['outcome']).copy()

        # Reporting stats
        success_rate = df_clean['outcome'].mean()
        failure_count = (df_clean['outcome'] == 0).sum()
        success_count = (df_clean['outcome'] == 1).sum()
        removed_count = len(df) - len(df_clean)

        print(f"📊 OUTCOME LABELING RESULTS:")
        print(f"   • Original trials: {len(df):,}")
        print(f"   • Trials with clear outcomes: {len(df_clean):,}")
        print(f"   • Trials removed (unknown status): {removed_count:,}")
        print(f"   • Successful trials: {success_count:,} ({success_rate:.1%})")
        print(f"   • Failed trials: {failure_count:,} ({1-success_rate:.1%})")

        if failure_count < 50:
            print(f"   ⚠️ WARNING: Low number of failure cases ({failure_count}) may impact model performance")

        print(f"\n📈 DETAILED STATUS BREAKDOWN:")
        status_outcome = df_clean.groupby(['overall_status', 'outcome']).size().unstack(fill_value=0)
        status_outcome.columns = ['Failure', 'Success']
        for status in status_outcome.index:
            failures = status_outcome.loc[status, 'Failure']
            successes = status_outcome.loc[status, 'Success']
            total = failures + successes
            print(f"   • {status:<25}: {total:>4,} total ({failures:>3,} fail, {successes:>3,} success)")
        print(f"\n✅ Outcome labels created successfully!")
        return df_clean

    def preprocess_text_fields(self, df):
        """Preprocess and combine text fields for embedding"""
        print("📝 PROCESSING TEXT FIELDS")
        print("-" * 40)

        def clean_text(text):
            """Clean and normalize text data"""
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'[^\w\s.,;:()\-]', ' ', text)
            text = ' '.join(text.split())
            return text

        # Define text fields to process
        text_fields = [
            'brief_title', 'official_title', 'brief_summary',
            'detailed_description', 'eligibility_criteria'
        ]
        # Clean individual text fields
        for field in text_fields:
            if field in df.columns:
                df[f'{field}_clean'] = df[field].apply(clean_text)
                clean_count = df[f'{field}_clean'].str.len().gt(0).sum()
                print(f"   • {field:<20}: {clean_count:>5,}/{len(df):,} trials have content")

        # Combine with field labels
        text_components = [f'{field}_clean' for field in text_fields if f'{field}_clean' in df.columns]
        print(f"\n🔤 COMBINING TEXT FIELDS:")
        print(f"   • Fields to combine: {len(text_components)}")

        def combine_text_fields(row):
            parts = []
            field_mapping = {
                'brief_title_clean': 'Title',
                'official_title_clean': 'Official',
                'brief_summary_clean': 'Summary',
                'detailed_description_clean': 'Description',
                'eligibility_criteria_clean': 'Eligibility'
            }
            for field in text_components:
                if pd.notna(row[field]) and str(row[field]).strip():
                    label = field_mapping.get(field, field)
                    content = str(row[field]).strip()
                    if content:
                        parts.append(f"{label}: {content}")
            return ' | '.join(parts)

        df['combined_text'] = df.apply(combine_text_fields, axis=1)
        df['combined_text'] = df['combined_text'].apply(
            lambda x: ' '.join(str(x).split()[:300])  # Limit to 300 words
        )
        text_lengths = df['combined_text'].str.len()
        word_counts = df['combined_text'].str.split().str.len()
        print(f"📊 COMBINED TEXT STATISTICS:")
        print(f"   • Average length: {text_lengths.mean():.0f} characters")
        print(f"   • Average words: {word_counts.mean():.0f} words")
        print(f"   • Max length: {text_lengths.max():,} characters")
        print(f"   • Trials with text: {(text_lengths > 0).sum():,}/{len(df):,}")
        print(f"✅ Text processing completed!")
        return df

    def preprocess_structured_features(self, df):
        """Process and encode structured features"""
        print("📊 PROCESSING STRUCTURED FEATURES")
        print("-" * 40)

        numerical_features = ['enrollment_count']
        print(f"🔢 NUMERICAL FEATURES:")
        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                median_value = df[feature].median()
                missing_count = df[feature].isna().sum()
                df[feature] = df[feature].fillna(median_value)
                print(f"   • {feature:<20}: {missing_count:>4,} missing filled with {median_value:.0f}")
                print(f"     Range: {df[feature].min():.0f} - {df[feature].max():.0f}, Mean: {df[feature].mean():.0f}")

        print(f"\n📅 DATE FEATURES:")
        date_features = ['start_date', 'completion_date', 'primary_completion_date']
        for feature in date_features:
            if feature in df.columns:
                try:
                    date_series = pd.to_datetime(df[feature], errors='coerce')
                    df[f'{feature}_year'] = date_series.dt.year
                    median_year = df[f'{feature}_year'].median()
                    missing_count = df[f'{feature}_year'].isna().sum()
                    df[f'{feature}_year'] = df[f'{feature}_year'].fillna(median_year)
                    year_range = f"{df[f'{feature}_year'].min():.0f}-{df[f'{feature}_year'].max():.0f}"
                    print(f"   • {feature:<20}: {missing_count:>4,} missing, range {year_range}")
                except Exception as e:
                    df[f'{feature}_year'] = 2020
                    print(f"   • {feature:<20}: Failed to process, set to 2020")

        print(f"\n📊 CATEGORICAL FEATURES:")
        categorical_features = [
            'study_type', 'phases', 'allocation', 'intervention_model',
            'masking', 'primary_purpose', 'gender', 'lead_sponsor_class'
        ]
        for feature in categorical_features:
            if feature in df.columns:
                missing_count = df[feature].isna().sum()
                df[feature] = df[feature].fillna('Unknown')
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                unique_count = len(le.classes_)
                print(f"   • {feature:<20}: {missing_count:>4,} missing, {unique_count:>3} unique values")
                top_categories = df[feature].value_counts().head(3)
                print(f"     Top categories: {dict(top_categories)}")

        print(f"\n🔧 DERIVED FEATURES:")
        if 'start_date_year' in df.columns and 'completion_date_year' in df.columns:
            df['trial_duration_years'] = df['completion_date_year'] - df['start_date_year']
            df['trial_duration_years'] = df['trial_duration_years'].clip(lower=0, upper=20)
            duration_mean = df['trial_duration_years'].mean()
            print(f"   • trial_duration_years   : Mean {duration_mean:.1f} years")
        if 'phases' in df.columns:
            df['is_multi_phase'] = df['phases'].str.contains('\|', na=False).astype(int)
            multi_count = df['is_multi_phase'].sum()
            print(f"   • is_multi_phase         : {multi_count:,} trials have multiple phases")
        if 'location_countries' in df.columns:
            df['is_international'] = df['location_countries'].str.contains('\|', na=False).astype(int)
            intl_count = df['is_international'].sum()
            print(f"   • is_international       : {intl_count:,} trials are international")
        print(f"✅ Structured features processing completed!")
        return df

print("✅ ClinicalTrialPreprocessor class defined successfully!")
