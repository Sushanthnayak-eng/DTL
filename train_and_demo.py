"""
train_and_demo.py

Single-file pipeline for rapid compost-quality MVP (Option A).

What it does:
- Loads "compost_data.csv"
- Cleans numeric columns (median imputation)
- Creates a suitability label using these rules:
    - pH between 6.5 and 8.0
    - MC(%) between 20 and 40
    - C/N Ratio < 25
    - TOC(%) > 8
    - OM(%) > 20
  -> If ALL conditions pass -> "Suitable" else "Not Suitable"
- Trains RandomForestClassifier (80/20 split)
- Prints evaluation (accuracy, classification report)
- Saves scaler and model as:
    - models/scaler.pkl
    - models/compost_model.pkl
- Provides functions:
    - preprocess_row(dict or pd.Series) -> numpy array
    - predict_sample(dict) -> (label, probability)
    - get_suggestions(dict) -> list of suggestions
- Demo: interactive input OR run on a few rows from CSV

Usage:
  python train_and_demo.py           # train (if model missing) and run quick demo on test samples
  python train_and_demo.py --retrain  # force retrain
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------
# Config
# ---------------------------
CSV_FILENAME = "compost_data.csv"   # put your CSV here
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "compost_model.pkl"
RANDOM_STATE = 42

# Columns you've provided:
# Temperature	MC(%)	pH	C/N Ratio	Ammonia(mg/kg)	Nitrate(mg/kg)	TN(%)	TOC(%)	EC(ms/cm)	OM(%)	T Value	GI(%)	Score
# We'll use all columns except 'Score' as features. If your file contains different capitalization/spacing,
# pandas will read exact headers — adjust names below if necessary.
EXPECTED_COLUMNS = [
    "Temperature", "MC(%)", "pH", "C/N Ratio", "Ammonia(mg/kg)", "Nitrate(mg/kg)",
    "TN(%)", "TOC(%)", "EC(ms/cm)", "OM(%)", "T Value", "GI(%)", "Score"
]

# Define feature columns (exclude Score)
FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c != "Score"]

# Suitability rules (Option A)
def is_suitable_row(r):
    """
    r: pandas Series with columns above
    Returns True if all conditions are met.
    Rules:
      - pH between 6.5 and 8.0
      - MC(%) between 20 and 40
      - C/N Ratio < 25
      - TOC(%) > 8
      - OM(%) > 20
    If any of these columns missing in dataset, treat missing as failing the rule.
    """
    try:
        ph_ok = (r.get("pH") is not None) and (6.5 <= float(r["pH"]) <= 8.0)
        mc_ok = (r.get("MC(%)") is not None) and (20.0 <= float(r["MC(%)"]) <= 40.0)
        cn_ok = (r.get("C/N Ratio") is not None) and (float(r["C/N Ratio"]) < 25.0)
        toc_ok = (r.get("TOC(%)") is not None) and (float(r["TOC(%)"]) > 8.0)
        om_ok = (r.get("OM(%)") is not None) and (float(r["OM(%)"]) > 20.0)
    except Exception:
        return False
    return bool(ph_ok and mc_ok and cn_ok and toc_ok and om_ok)

# ---------------------------
# Suggestion engine
# ---------------------------
def get_suggestions(sample):
    """
    sample: dict-like with keys matching FEATURE_COLUMNS
    Returns a short prioritized list of suggestions (strings).
    """
    s = []
    # pH suggestions
    try:
        ph = float(sample.get("pH", np.nan))
        if np.isnan(ph):
            s.append("pH value missing — measure pH to get accurate suggestions.")
        else:
            if ph < 6.5:
                s.append("pH is low: add agricultural lime or wood ash to raise pH.")
            elif ph > 8.0:
                s.append("pH is high: mix in acidic materials (e.g., composted leaves) to lower pH.")
    except Exception:
        s.append("pH reading invalid — check input.")

    # Moisture suggestions
    try:
        mc = float(sample.get("MC(%)", np.nan))
        if np.isnan(mc):
            s.append("Moisture missing — measure MC% for better advice.")
        else:
            if mc < 20:
                s.append("Moisture is low: add water and mix; keep pile covered to retain moisture.")
            elif mc > 40:
                s.append("Moisture is high: turn pile and air-dry; reduce watering or improve drainage.")
    except Exception:
        s.append("Moisture reading invalid — check input.")

    # C/N ratio suggestions
    try:
        cn = float(sample.get("C/N Ratio", np.nan))
        if not np.isnan(cn):
            if cn > 30:
                s.append("C/N ratio is high: add nitrogen-rich materials (kitchen waste, green manure).")
            elif cn > 25:
                s.append("C/N ratio slightly high: add some nitrogen-rich material and mix.")
    except Exception:
        pass

    # TOC / OM suggestions
    try:
        toc = float(sample.get("TOC(%)", np.nan))
        om = float(sample.get("OM(%)", np.nan))
        if not np.isnan(toc) and toc <= 8:
            s.append("TOC is low: add dry leaves, straw, or sawdust and re-compost.")
        if not np.isnan(om) and om <= 20:
            s.append("OM is low: add more organic feedstock (plant residues) and re-compost.")
    except Exception:
        pass

    # EC suggestion (salinity)
    try:
        ec = float(sample.get("EC(ms/cm)", np.nan))
        if not np.isnan(ec):
            if ec > 4.0:
                s.append("Electrical conductivity (EC) is high: salts may be high — consider leaching or dilution.")
    except Exception:
        pass

    # Ammonia / Nitrate quick check
    try:
        ammonia = float(sample.get("Ammonia(mg/kg)", np.nan))
        nitrate = float(sample.get("Nitrate(mg/kg)", np.nan))
        # High ammonia indicates incomplete composting
        if not np.isnan(ammonia) and ammonia > 1000:
            s.append("High ammonia suggests incomplete composting: increase aeration and longer curing.")
        if not np.isnan(nitrate) and nitrate < 10:
            s.append("Low nitrate may mean insufficient nitrification; allow more curing time.")
    except Exception:
        pass

    # If nothing generated
    if len(s) == 0:
        s.append("No major issues detected by rules. For finer advice, run lab tests or consult extension services.")

    # return top 3 suggestions, prioritized by earlier checks
    return s[:5]

# ---------------------------
# Preprocessing & training
# ---------------------------
def load_and_prepare(csv_path=CSV_FILENAME):
    # Read CSV. Use python engine so it can sniff separators (tabs, commas).
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # Display columns read
    print("Columns in CSV:", list(df.columns))

    # Try to standardize column names: strip whitespace
    df.rename(columns=lambda c: c.strip(), inplace=True)

    # Ensure expected columns exist; warn if missing
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            print(f"Warning: expected column '{c}' not found in CSV. Proceeding but results may be impacted.")

    # Keep only relevant columns that exist in file (features)
    existing_features = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Drop rows that are entirely empty in those columns
    df = df.dropna(axis=0, how="all", subset=existing_features)

    # Convert numeric columns to numeric (coerce errors to NaN)
    for c in existing_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Create label column 'suitability_label' using our rules (Option A)
    df["suitability_label"] = df.apply(lambda row: "Suitable" if is_suitable_row(row) else "Not Suitable", axis=1)

    # Drop rows where all features are NaN
    df = df.dropna(axis=0, how="all", subset=existing_features)

    # Fill remaining NaNs with column median (fast imputation)
    for c in existing_features:
        if df[c].isna().any():
            median_val = df[c].median()
            df[c].fillna(median_val, inplace=True)

    # Final features and label
    X = df[existing_features].copy()
    y = df["suitability_label"].copy()

    # Encode label to numeric
    y_num = y.map(lambda v: 1 if v == "Suitable" else 0)

    return df, X, y_num, existing_features

def train_and_save(X, y, feature_names, retrain=False):
    MODELS_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.exists() and SCALER_PATH.exists() and not retrain:
        print("Model and scaler already exist. To retrain, run with --retrain flag.")
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save scaler and model
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(clf, MODEL_PATH)
    # Also save feature order
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved model  -> {MODEL_PATH}")

# ---------------------------
# Prediction helpers
# ---------------------------
def load_model_and_scaler():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler file not found. Run training first.")
    scaler = joblib.load(SCALER_PATH)
    clf = joblib.load(MODEL_PATH)
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return scaler, clf, feature_names

def preprocess_row(sample_dict_or_series, feature_names, scaler=None):
    """
    Accepts a dict-like or pandas Series with keys matching feature_names.
    Returns scaled 1D numpy array (shape (n_features,)).
    If scaler is None, returns unscaled features (numpy array).
    """
    # Build row in order of feature_names
    vals = []
    for f in feature_names:
        v = sample_dict_or_series.get(f) if isinstance(sample_dict_or_series, dict) else sample_dict_or_series.get(f, np.nan)
        try:
            v_num = float(v)
        except Exception:
            v_num = np.nan
        vals.append(v_num)
    arr = np.array(vals, dtype=float).reshape(1, -1)
    # Impute any NaNs with column-wise median (quick). This requires training data medians ideally;
    # here we do a simple replacement with 0s if scaler is None. But normally you shouldn't pass rows with NaN.
    # If scaler provided, assume scaler was fit and arr shape matches.
    if scaler is not None:
        # scaler expects no NaNs; replace NaNs with 0 (since scaler will center)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = 0.0
        arr_scaled = scaler.transform(arr)
        return arr_scaled
    else:
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = 0.0
        return arr

def predict_sample(sample, return_prob=True):
    """
    sample: dict-like with feature keys
    returns: (label_str, probability)
    Loads model & scaler if needed.
    """
    scaler, clf, feature_names = load_model_and_scaler()
    Xs = preprocess_row(sample, feature_names, scaler=scaler)
    prob = None
    label = "Not Suitable"
    if hasattr(clf, "predict_proba") and return_prob:
        proba = clf.predict_proba(Xs)[0]  # [prob_not, prob_yes]
        prob = float(proba[1])  # probability of 'Suitable'
        pred = clf.predict(Xs)[0]
        label = "Suitable" if int(pred) == 1 else "Not Suitable"
    else:
        pred = clf.predict(Xs)[0]
        label = "Suitable" if int(pred) == 1 else "Not Suitable"
    return label, prob

# ---------------------------
# Demo / CLI
# ---------------------------
def run_quick_demo(csv_path=CSV_FILENAME):
    print("Running quick demo...")
    df, X, y, feature_names = load_and_prepare(csv_path)
    # If model doesn't exist, train
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("Model not found — training now...")
        train_and_save(X, y, feature_names, retrain=True)
    else:
        print("Model found — skipping training.")
    scaler, clf, feature_names = load_model_and_scaler()

    # Show a few sample predictions from the dataset (first 3 rows)
    samples_to_show = min(5, len(df))
    print(f"\nShowing predictions for {samples_to_show} sample rows from dataset:")
    for i in range(samples_to_show):
        row = df.iloc[i]
        sample_dict = {f: row[f] for f in feature_names}
        label, prob = predict_sample(sample_dict)
        suggestions = get_suggestions(sample_dict) if label == "Not Suitable" else ["Compost appears suitable by quick rules."]
        print(f"\nSample #{i+1}:")
        print(sample_dict)
        print(f"Prediction: {label}   Confidence(Suitable)={prob:.3f}")
        print("Suggestions:")
        for s in suggestions:
            print(" -", s)

    # Offer interactive input
    print("\nInteractive prediction: enter values or press Enter to skip.")
    user_vals = {}
    for f in feature_names:
        val = input(f"Enter {f} (or blank to use median from dataset): ").strip()
        if val == "":
            # use dataset median
            user_vals[f] = float(X[f].median())
        else:
            try:
                user_vals[f] = float(val)
            except Exception:
                user_vals[f] = float(X[f].median())
    label, prob = predict_sample(user_vals)
    print("\nResult for your input:")
    print(user_vals)
    print(f"Prediction: {label}   Confidence(Suitable)={prob:.3f}")
    if label == "Not Suitable":
        print("Suggestions:")
        for s in get_suggestions(user_vals):
            print(" -", s)
    else:
        print("Suggestion: Compost appears suitable by quick rules. Consider lab testing for final confirmation.")

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and demo compost suitability model (Option A rules).")
    parser.add_argument("--retrain", action="store_true", help="Retrain model even if saved model exists.")
    parser.add_argument("--csv", type=str, default=CSV_FILENAME, help="Path to compost CSV file.")
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).exists():
        print(f"CSV file not found at {csv_path}. Please place your CSV in the same folder and name it '{csv_path}' or pass --csv path.")
        sys.exit(1)

    # Load and prepare dataset
    df, X, y, feature_names = load_and_prepare(csv_path)

    # Train if needed or requested
    if args.retrain or (not MODEL_PATH.exists()) or (not SCALER_PATH.exists()):
        print("Training model...")
        train_and_save(X, y, feature_names, retrain=args.retrain)
    else:
        print("Existing model detected. Use --retrain to force retraining.")

    # Run quick demo / interactive demo
    run_quick_demo(csv_path)

if __name__ == "__main__":
    main()
