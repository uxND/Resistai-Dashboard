import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib

def generate_shap_explanations(model_path="models/xgb_model.pkl", data_path="data/processed/final_dataset.csv"):
    """
    Generates global SHAP explanations and saves a summary beeswarm plot.
    This shows which features drive resistance across the entire dataset.
    """
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Please ensure the model is trained first via src/train.py.")
        return None

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Warning: Data not found at {data_path}. Generating mock data for demonstration.")
        # Fallback mock data if the preprocessing pipeline hasn't been run yet
        df = pd.DataFrame({
            'species_enc': np.random.randint(0, 5, 100),
            'antibiotic_enc': np.random.randint(0, 10, 100),
            'isolation_source_enc': np.random.randint(0, 3, 100),
            'mic_value': np.random.uniform(0.1, 64.0, 100),
            'multi_drug_count': np.random.randint(0, 10, 100),
            'gram_stain_enc': np.random.randint(0, 2, 100)
        })

    feature_cols = [
        'species_enc', 'antibiotic_enc', 'isolation_source_enc',
        'mic_value', 'multi_drug_count', 'gram_stain_enc'
    ]

    # Ensure we only use the specified feature columns
    X = df[feature_cols] if all(c in df.columns for c in feature_cols) else df

    print("Calculating SHAP values... (This might take a moment depending on dataset size)")
    explainer = shap.TreeExplainer(model)
    
    # For global summary, we sample to speed things up if the dataset is massive
    X_sample = shap.sample(X, 500) if len(X) > 500 else X
    
    # Generate SHAP values
    shap_values = explainer(X_sample)

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # --- 1. Global Summary Plot ---
    plt.figure(figsize=(10, 6))
    # Using the standard values for TreeExplainer
    shap.summary_plot(shap_values.values, X_sample, feature_names=feature_cols, show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14, pad=20)
    plt.tight_layout()
    
    summary_path = "outputs/shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"✅ Global SHAP summary plot saved to {summary_path}")
    plt.close()

    return explainer

def explain_single_prediction(explainer, input_data, feature_names, save_path="outputs/shap_waterfall.png"):
    """
    Generates a waterfall plot for a single bacterial isolate to explain 'why' 
    it was classified as resistant or susceptible.
    """
    shap_values = explainer(input_data)
    
    plt.figure(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=explainer.expected_value,
            data=input_data[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.title("Local Prediction Explanation", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Local SHAP waterfall plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # If this script is executed directly via terminal, run the global explanation
    print("--- Running ResistAI Explainability Pipeline ---")
    explainer_obj = generate_shap_explanations()
    if explainer_obj:
        print("Explainability pipeline completed successfully.")