# data_collection.py

import requests
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

OUTPUT_PATH = "/content/drive/MyDrive/clinical-trial-outcome-prediction/data"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.descriptionModule.detailedDescription",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.whyStopped",
    "protocolSection.statusModule.startDateStruct.date",
    "protocolSection.statusModule.completionDateStruct.date",
    "protocolSection.statusModule.primaryCompletionDateStruct.date",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.designInfo.allocation",
    "protocolSection.designModule.designInfo.interventionModel",
    "protocolSection.designModule.designInfo.maskingInfo.masking",
    "protocolSection.designModule.designInfo.primaryPurpose",
    "protocolSection.designModule.enrollmentInfo.count",
    "protocolSection.designModule.enrollmentInfo.type"
]

def fetch_trials(condition: str, max_count: int = 500):
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    studies = []
    print(f"Fetching {condition} trials...")
    params = {
        "format": "json",
        "pageSize": 500,
        "fields": ",".join(FIELDS),
        "query.cond": condition
    }
    collected = 0
    while collected < max_count:
        response = requests.get(base_url, params=params, timeout=60)
        data = response.json()
        chunk = data.get("studies", [])
        if not chunk:
            break
        studies.extend(chunk)
        collected += len(chunk)
        if not data.get("nextPageToken"):
            break
        params["pageToken"] = data["nextPageToken"]
        time.sleep(1)
    print(f" - Collected: {len(studies)} for {condition}")
    return studies

def collect_and_save(conditions=None, max_per_condition=300):
    if conditions is None:
        conditions = [
            "cancer", "diabetes", "alzheimer", "cardiovascular", "hypertension",
            "depression", "arthritis", "covid"
        ]
    all_records = []
    for cond in conditions:
        all_records.extend(fetch_trials(cond, max_per_condition))
    # Flatten nested fields for DataFrame
    records = []
    for study in all_records:
        rec = {}
        for field in FIELDS:
            parts = field.split(".")[1:]  # skip 'protocolSection'
            val = study.get("protocolSection", {})
            try:
                for p in parts:
                    if "[" in p:  # handle field like fields[0]
                        base = p.split("[")[0]
                        idx = int(p[p.find("[")+1:p.find("]")])
                        val = val.get(base, [])[idx]
                    else:
                        val = val.get(p, None)
                rec[field.replace("protocolSection.", "")] = val
            except Exception:
                rec[field.replace("protocolSection.", "")] = None
        records.append(rec)
    df = pd.DataFrame(records)
    output_file = f"{OUTPUT_PATH}/clinical_trials_raw.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Data saved to {output_file}")

if __name__ == "__main__":
    collect_and_save()
