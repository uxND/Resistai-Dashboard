# 🧬 ResistAI — Antibiotic Resistance Predictor

> **AI-powered prediction of antimicrobial resistance (AMR) with integrated clinical decision support and biological explainability.**
> 
> *Built for CODECURE @ SPIRIT 2026, IIT BHU Varanasi.*

---

## 🛑 The Problem
Antimicrobial resistance (AMR) is a top global public health and development threat, responsible for over 1.2 million deaths annually. When a patient presents with an infection, clinicians often lack fast, data-driven tools to determine which antibiotics will fail and which will succeed, leading to prolonged illness and increased transmission of resistant strains.

## 💡 Our Solution: ResistAI
ResistAI doesn't just act as a "black-box" classifier. It is a complete clinical decision-support tool that predicts resistance based on bacterial isolate data, explains the key drivers behind every prediction, visualizes the underlying resistance gene networks, and recommends effective alternative treatments.

### **Key Features**
* **🎯 High-Accuracy Classification:** Predicts Resistance (R) vs. Susceptibility (S) using an optimized XGBoost model.
* **⚠️ MDR Risk Scoring:** Calculates a Multi-Drug Resistance score per isolate to highlight highly dangerous strains.
* **🔍 SHAP Explainability:** Every prediction generates a waterfall chart showing exactly which biological features pushed the prediction toward resistant or susceptible.
* **🕸️ Gene Network Visualization:** An interactive force-directed graph mapping resistance genes to their targeted antibiotics (using the CARD database).
* **💊 Antibiotic Recommender:** Cross-references historical susceptibility data to suggest the top 3 alternative antibiotics when a strain is predicted to be resistant.
* **📊 Global Heatmap:** An interactive dashboard exploring historical resistance patterns across species.

---

## 🛠️ Tech Stack
* **Machine Learning:** `Python 3.10+`, `scikit-learn`, `XGBoost`, `imbalanced-learn` (SMOTE)
* **Explainability:** `SHAP`
* **Network Science:** `NetworkX`, `Pyvis`
* **Frontend & Visualization:** `Streamlit`, `Plotly`, `Matplotlib`
* **Datasets Utilized:** * Primary: Mendeley AMR Dataset
  * Secondary: Kaggle Multi-Resistance Dataset
  * Contextual: Comprehensive Antibiotic Resistance Database (CARD)

---

## 🚀 How to Run Locally

Follow these steps to reproduce the project on your local machine.

### 1. Clone the repository and navigate to the directory
```bash
git clone [https://github.com/yourusername/resistai.git](https://github.com/yourusername/resistai.git)
cd resistai
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 3. Run the Data & Modeling Pipeline
Execute these scripts in order to build the datasets, train the models, and generate the visualizations.
```bash
# Clean datasets, handle class imbalance, and engineer features
python src/preprocess.py

# Train the XGBoost and Random Forest models
python src/train.py

# Generate the interactive Gene Network HTML file
python src/network.py

# (Optional) Generate global SHAP summary charts
python src/explain.py
```

### 4. Launch the Web App
```bash
streamlit run app/streamlit_app.py
```
The app will automatically open in your default browser at http://localhost:8501.

## 📂 Project Structure
resistai/
├── data/
│   ├── raw/                  # Original downloaded CSVs (Mendeley, Kaggle)
│   ├── processed/            # Cleaned and engineered datasets
│   └── card_annotations.tsv  # CARD gene lookup data
├── notebooks/                # Jupyter notebooks for initial EDA
├── src/
│   ├── preprocess.py         # Data cleaning, encoding, and SMOTE balancing
│   ├── features.py           # Biological feature engineering (MDR, Gram Stain)
│   ├── train.py              # XGBoost/RF training and evaluation
│   ├── explain.py            # SHAP model explainability generation
│   └── network.py            # Pyvis/NetworkX graph generation
├── models/                   
│   ├── xgb_model.pkl         # Saved trained classifier
│   └── encoders.pkl          # Saved label encoders for UI inputs
├── app/
│   └── streamlit_app.py      # Main Streamlit UI frontend
├── outputs/                  # Generated HTML networks and SHAP plots
├── requirements.txt
└── README.md

## 🏆 Hackathon Judging Alignment (CODECURE @ SPIRIT 2026)
**Analyze Data**: Preprocessing pipeline handles merging and cleaning of the primary Mendeley dataset.

**Build Models**: Implements XGBoost and handles class imbalance via SMOTE.

**Explore Feature Importance**: Integrated SHAP values for both global model behavior and local, per-prediction explanations.

**Biological Annotations**: Generates interactive NetworkX graphs linking resistance mechanisms based on the CARD database.

**Suggest Treatment Strategies**: Includes an automated fallback recommender system inside the Streamlit UI.

*Developed with ❤️ for global health*