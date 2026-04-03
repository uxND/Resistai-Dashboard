import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
import os

def train_pipeline(data_path="data/processed/final_dataset.csv"):
    """
    Executes the full model training pipeline.
    Loads processed data, trains XGBoost and Random Forest, compares them, 
    and saves the best model to disk.
    """
    print("🚀 Starting Model Training Pipeline...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data not found at {data_path}.")
        print("Please run 'python src/preprocess.py' first to generate the dataset.")
        return

    # These must exactly match the output of features.py and the inputs of streamlit_app.py
    feature_cols = [
        'species_enc', 'antibiotic_enc', 'isolation_source_enc', 
        'mic_value', 'multi_drug_count', 'gram_stain_enc'
    ]
    
    # Validation check
    if not all(col in df.columns for col in feature_cols + ['resistance']):
        print("❌ Error: Missing required columns in dataset. Did feature engineering run correctly?")
        return

    X = df[feature_cols]
    y = df['resistance']

    print(f"📊 Dataset loaded. Total samples: {len(X)}. Splitting into train/test sets...")
    
    # Stratify ensures the train and test sets have the same proportion of resistant/susceptible cases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train XGBoost (Primary Model)
    print("🧠 Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    xgb_model.fit(X_train, y_train)

    # 3. Train Random Forest (Baseline for comparison)
    print("🌲 Training Random Forest Classifier (Baseline)...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 4. Evaluation
    print("\n" + "="*30)
    print("📈 Model Evaluation Results")
    print("="*30)
    
    for name, model in [("XGBoost", xgb_model), ("Random Forest", rf_model)]:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
        print(f"AUC-ROC:   {roc_auc_score(y_test, proba):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

    # 5. Save the primary model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_model.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\n✅ Primary model (XGBoost) saved successfully to {model_path}")
    print("You can now run 'python src/explain.py' or launch the UI with 'streamlit run app/streamlit_app.py'")

if __name__ == "__main__":
    # Execute the training pipeline when the script is run directly
    train_pipeline()