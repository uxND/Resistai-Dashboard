import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="ResistAI — Antibiotic Resistance Predictor",
    page_icon="🧬",
    layout="wide"
)

FEATURE_COLS = [
    'species_enc', 'antibiotic_enc', 'isolation_source_enc', 
    'mic_value', 'multi_drug_count', 'gram_stain_enc'
]

# --- Helper Functions & Data Loading ---
@st.cache_resource
def load_model_and_encoders():
    """Loads the trained XGBoost model and the label encoders."""
    try:
        model = joblib.load("models/xgb_model.pkl")
        encoders = joblib.load("models/encoders.pkl")
        return model, encoders
    except Exception:
        # Failsafe: Returns None if the user hasn't trained the model yet
        return None, None

@st.cache_data
def load_data():
    """Loads the processed dataset, or mock data if not found."""
    try:
        return pd.read_csv("data/processed/final_dataset.csv")
    except FileNotFoundError:
        # Mock data to keep the UI from crashing during initial testing
        return pd.DataFrame({
            'species': ['Escherichia coli', 'Staphylococcus aureus', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa'],
            'antibiotic': ['Ciprofloxacin', 'Methicillin', 'Meropenem', 'Vancomycin'],
            'isolation_source': ['Blood', 'Urine', 'Sputum', 'Wound'],
            'resistance': [1, 0, 1, 0]
        })

def encode_input(species, antibiotic, isolation_source, mic_value, mdr_count, gram_stain, encoders):
    """Safely transforms user UI inputs into numeric features for the model."""
    # Transform or default to 0 if the label is unseen
    sp_enc = encoders['species'].transform([species])[0] if species in encoders['species'].classes_ else 0
    ab_enc = encoders['antibiotic'].transform([antibiotic])[0] if antibiotic in encoders['antibiotic'].classes_ else 0
    iso_enc = encoders['isolation_source'].transform([isolation_source])[0] if isolation_source in encoders['isolation_source'].classes_ else 0
    gram_enc = 1 if gram_stain == "Negative" else 0
    
    return np.array([sp_enc, ab_enc, iso_enc, mic_value, mdr_count, gram_enc])

def get_alternatives(species, current_drug, df):
    """Finds alternative antibiotics that this species is typically susceptible to."""
    if df.empty or 'resistance' not in df.columns: 
        return [("Vancomycin", 0.1), ("Linezolid", 0.15)] # Mock fallback
    
    sp_data = df[df['species'] == species]
    if sp_data.empty: return []
    
    # Calculate resistance rate per antibiotic for this specific species
    res_rates = sp_data.groupby('antibiotic')['resistance'].mean().sort_values()
    
    # Filter for drugs with < 50% resistance rate, excluding the current failing drug
    alts = [(drug, rate) for drug, rate in res_rates.items() if drug != current_drug and rate < 0.5]
    return alts[:3]

# --- Initialize Application ---
model, encoders = load_model_and_encoders()
df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🧬 ResistAI")
st.sidebar.markdown("AI-powered AMR prediction and clinical decision support.")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Predict Resistance", "Dataset Overview", "Gene Network", "About"])

# --- Page 1: Predict Resistance ---
if page == "Predict Resistance":
    st.title("🔬 Clinical Decision Support: AMR Prediction")
    st.markdown("Enter bacterial isolate details below to predict resistance probabilities and receive alternative treatment recommendations.")

    if model and encoders:
        # Input Form
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # Use encoder classes if real data isn't loaded yet
                species_list = df['species'].unique() if not df.empty else encoders['species'].classes_
                ab_list = df['antibiotic'].unique() if not df.empty else encoders['antibiotic'].classes_
                iso_list = df['isolation_source'].unique() if not df.empty else encoders['isolation_source'].classes_
                
                species = st.selectbox("Bacterial species", species_list)
                antibiotic = st.selectbox("Antibiotic to test", ab_list)
                isolation_source = st.selectbox("Isolation source", iso_list)
            
            with col2:
                mic_value = st.number_input("MIC value (µg/mL)", min_value=0.0, max_value=256.0, value=1.0, step=0.25)
                multi_drug_count = st.slider("Known resistance to other classes (MDR Count)", 0, 15, 2)
                gram_stain = st.radio("Gram stain", ["Positive", "Negative"], horizontal=True)

        if st.button("Predict Efficacy", type="primary", use_container_width=True):
            # Process Prediction
            input_data = encode_input(species, antibiotic, isolation_source, mic_value, multi_drug_count, gram_stain, encoders)
            
            prediction = model.predict([input_data])[0]
            probability = model.predict_proba([input_data])[0][1]

            st.divider()
            
            # Display Top-Line Results
            col_r1, col_r2, col_r3 = st.columns(3)
            status = "🔴 RESISTANT" if prediction == 1 else "🟢 SUSCEPTIBLE"
            col_r1.metric("Prediction", status)
            col_r2.metric("Resistance Probability", f"{probability*100:.1f}%")
            col_r3.metric("MDR Risk Score", f"{min(multi_drug_count, 10)}/10")

            # Explanation and Recommendations Layout
            col_chart, col_alt = st.columns([1.5, 1])
            
            with col_chart:
                st.subheader("Model Decision Drivers (SHAP)")
                st.markdown("This chart explains *why* the model made this specific prediction. Red pushes towards resistant; blue pushes towards susceptible.")
                
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer(np.array([input_data]))
                
                # Matplotlib figure for Streamlit rendering
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals.values[0], 
                        base_values=explainer.expected_value, 
                        data=input_data, 
                        feature_names=FEATURE_COLS
                    ), 
                    show=False
                )
                st.pyplot(fig)

            with col_alt:
                if prediction == 1:
                    st.subheader("💊 Recommended Alternatives")
                    st.warning(f"High risk of clinical failure for {antibiotic}. Consider these alternatives:")
                    alternatives = get_alternatives(species, antibiotic, df)
                    
                    if alternatives:
                        for ab, rate in alternatives:
                            st.success(f"**{ab}** \n\nHistorical Susceptibility: {(1-rate)*100:.0f}%")
                    else:
                        st.error("No highly susceptible alternatives found in historical data for this species.")
                else:
                    st.subheader("✅ Treatment Viable")
                    st.success(f"**{antibiotic}** is likely to be an effective treatment for this isolate.")

    else:
        st.warning("⚠️ Model not found! Please run the training pipeline (`src/train.py`) first to generate the `xgb_model.pkl` and `encoders.pkl` files in the `models/` directory.")

# --- Page 2: Dataset Overview ---
elif page == "Dataset Overview":
    st.title("📊 Global Resistance Heatmap")
    st.markdown("Explore historical resistance patterns across different bacterial species and antibiotics.")
    
    if not df.empty and 'resistance' in df.columns:
        # Create a pivot table for the heatmap
        pivot = df.groupby(['species', 'antibiotic'])['resistance'].mean().unstack().fillna(0)
        
        fig = px.imshow(
            pivot, 
            color_continuous_scale="RdYlGn_r", # Red (High Resistance) to Green (Low Resistance)
            aspect="auto", 
            labels={"color": "Resistance Rate"}, 
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Key Dataset Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Isolates", len(df))
        m2.metric("Species Tracked", df['species'].nunique())
        m3.metric("Antibiotics Tracked", df['antibiotic'].nunique())
        m4.metric("Overall AMR Rate", f"{df['resistance'].mean()*100:.1f}%")
        
    else:
        st.info("Please run the data preprocessing pipeline to populate real data for the heatmap.")

# --- Page 3: Gene Network ---
elif page == "Gene Network":
    st.title("🕸️ Resistance Gene Network")
    st.markdown("Interactive force-directed graph mapping resistance genes to their targeted antibiotics based on the CARD Database.")
    
    network_path = "outputs/resistance_network.html"
    if os.path.exists(network_path):
        with open(network_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=700, scrolling=True)
    else:
        st.warning("⚠️ Network visualization file not found. Please run `python src/network.py` to generate it.")

# --- Page 4: About ---
elif page == "About":
    st.title("About ResistAI")
    st.markdown("""
    ### Built for CODECURE @ SPIRIT 2026, IIT BHU Varanasi.
    
    **Problem:** Antimicrobial resistance (AMR) kills over a million people annually. Clinicians often lack fast, data-driven tools to choose effective antibiotics for resistant strains.
    
    **Solution:** ResistAI utilizes machine learning (XGBoost) and SHAP explainability to move beyond black-box classification. It provides clinicians with actionable, interpretable data on AMR probabilities, explains the features driving the prediction, and recommends viable alternative treatments.
    
    **Tech Stack:**
    * **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn
    * **Explainability:** SHAP
    * **Network Science:** NetworkX, Pyvis
    * **Frontend:** Streamlit, Plotly
    """)