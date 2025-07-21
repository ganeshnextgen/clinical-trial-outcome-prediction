# feature_fusion.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

def fuse_and_split(input_csv, output_dir="data/processed"):
    df = pd.read_csv(input_csv)

    # Extract targets
    y = df["outcome"]
    id_col = df["identificationModule.nctId"] if "identificationModule.nctId" in df.columns else df.index
    X_numeric = df[
        [col for col in df.columns if col.startswith("text_emb_") or col.endswith("_enc") or col in ["enrollment"]]
    ]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    datasets = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
    }

    for name, data in datasets.items():
        np.save(out_dir / f"{name}_{timestamp}.npy", data)

    joblib.dump(scaler, out_dir / f"scaler_{timestamp}.pkl")

    print(f"✅ Fusion complete. Train/Test sets saved to {output_dir}")
    return datasets
