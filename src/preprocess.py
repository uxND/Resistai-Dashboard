import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

# Import the feature engineering function from your other script
# We use a try-except so it doesn't crash if run before features.py is finalized
try:
    from features import engineer_features
except ImportError:
    print("Warning: Could not import features.py. Ensure it is in the same directory.")
    def engineer_features(df): return df

def load_and_merge():
    """
    Loads the Mendeley and Kaggle datasets. 
    If they are not found, it generates a robust mock dataset for testing.
    """
    print("📥 Loading raw datasets...")
    try:
        # Load the first dataset (CSV)
        df1 = pd.read_csv("data/raw/Bacteria_dataset_Multiresictance.csv")
        
        # Load the second dataset
        # It looks like an Excel file based on Windows file type, but we add a fallback just in case
        try:
            df2 = pd.read_excel("data/raw/Dataset.xlsx")
        except FileNotFoundError:
            df2 = pd.read_csv("data/raw/Dataset.csv") 

        df = pd.concat([df1, df2], ignore_index=True)
        print("✅ Raw datasets loaded and merged successfully.")
        return df
        
    except FileNotFoundError as e:
        print(f"⚠️ Raw data not found: {e}")
        print("Generating a high-quality mock dataset for pipeline testing...")
        
        # Realistic mock data generation
        np.random.seed(42)
        species_list = ['Escherichia coli', 'Staphylococcus aureus', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa', 'Acinetobacter baumannii']
        antibiotics_list = ['Ciprofloxacin', 'Meropenem', 'Vancomycin', 'Ceftriaxone', 'Amikacin']
        sources_list = ['Blood', 'Urine', 'Sputum', 'Wound']
        
        n_samples = 2000
        mock_data = {
            'isolate_id': [f"ISO_{i}" for i in range(n_samples)],
            'species': np.random.choice(species_list, n_samples),
            'antibiotic': np.random.choice(antibiotics_list, n_samples),
            'isolation_source': np.random.choice(sources_list, n_samples),
            'susceptibility': np.random.choice(['Resistant', 'Susceptible', 'Intermediate'], n_samples, p=[0.4, 0.5, 0.1]),
            'mic_value': np.random.choice([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], n_samples)
        }
        return pd.DataFrame(mock_data)

def clean_and_encode(df):
    """
    Reshapes the wide data, cleans garbage values, 
    maps susceptibility to binary resistance, and encodes categories.
    """
    print("🧹 Cleaning and reshaping data...")
    if df.empty: return df
    
    # 1. Reshape from Wide to Long format
    abx_columns = [
        'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
        'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 
        'colistine', 'IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN'
    ]
    # Safely select only columns that actually exist in the dataframe
    cols_to_melt = [col for col in abx_columns if col in df.columns]
    
    if 'Souches' in df.columns:
        df = df.melt(
            id_vars=['Souches'], 
            value_vars=cols_to_melt, 
            var_name='antibiotic', 
            value_name='susceptibility'
        )
        df = df.rename(columns={'Souches': 'species'})
    
    # 2. Clean up missing and garbage data discovered in EDA
    df = df.dropna(subset=['susceptibility', 'species'])
    df = df[~df['species'].isin(['?', 'missing', 'Unknown'])] # Remove bad species

    # 3. Standardize resistance labels (Binary Classification)
    res_mapping = {
        'Resistant': 1, 'R': 1, 'r': 1,
        'Susceptible': 0, 'S': 0, 's': 0,
        'Intermediate': 0, 'I': 0, 'i': 0
    }
    df['resistance'] = df['susceptibility'].map(res_mapping)
    df = df.dropna(subset=['resistance'])

    # 4. Encode categorical columns for the ML model
    os.makedirs("models", exist_ok=True)
    encoders = {}
    
    # We add a fake isolation_source since your dataset doesn't have one, 
    # but the Streamlit UI expects it!
    df['isolation_source'] = 'Clinical Sample'
    
    cols_to_encode = ['species', 'antibiotic', 'isolation_source']
    for col in cols_to_encode:
        df[col] = df[col].astype(str).fillna('Unknown')
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le
            
    # Save encoders for the UI input forms
    joblib.dump(encoders, "models/encoders.pkl")
    print(f"✅ Data cleaned! Label encoders saved. New shape: {df.shape}")
    
    return df

def balance_data(X, y):
    """
    Uses SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
    (e.g., if you have 80% susceptible and 20% resistant data points).
    """
    print(f"⚖️ Balancing dataset using SMOTE. Original shape: {X.shape}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"✅ Balanced dataset shape: {X_res.shape}")
    return X_res, y_res

if __name__ == "__main__":
    # --- Execute the Data Pipeline ---
    print("=== Starting Data Preprocessing Pipeline ===")
    
    # 1. Load Data
    raw_df = load_and_merge()
    
    # 2. Clean and Encode
    cleaned_df = clean_and_encode(raw_df)
    
    # 3. Engineer Features (Calls src/features.py)
    featured_df = engineer_features(cleaned_df)
    
    # 4. Save Final Dataset
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/final_dataset.csv"
    featured_df.to_csv(save_path, index=False)
    
    print(f"🎉 Preprocessing complete! Final dataset saved to {save_path}")