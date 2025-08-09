import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ------------------- Streamlit Config & Styles -------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# Hide Streamlit's default menu and footer for clean look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem;}
        .stProgress > div > div > div > div {background-image: linear-gradient(90deg, #fa5252, #ffe066);}
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Model Assets -------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    selected_features = pickle.load(open("features.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Could not load model/scaler/features: {e}")
    st.stop()

# ------------------- Load Dataset for Ranges -------------------
try:
    df = pd.read_csv("heart.csv")
    feature_ranges = {col: (float(df[col].min()), float(df[col].max())) for col in selected_features}
except Exception as e:
    st.warning(f"Could not auto-load value ranges, using defaults. Reason: {e}")
    feature_ranges = {f: (0., 100.) for f in selected_features}

# ------------------- Feature Descriptions & Types -------------------
feature_descriptions = {
    "age": "Age of the patient (risk increases with age)",
    "sex": "Gender (0 = Female, 1 = Male), affects risk factors",
    "cp": "Chest pain type (0–3), strong predictor",
    "trestbps": "Resting blood pressure (mm Hg), hypertension risk",
    "chol": "Serum cholesterol level (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)",
    "restecg": "Resting ECG results (0–2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = Yes, 0 = No)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of the peak exercise ST segment (0–2)",
    "ca": "Number of major vessels (0–3) colored by fluoroscopy",
    "thal": "Thalassemia status (0–3)"
}

discrete_options = {
    "sex": [0, 1],
    "cp": [0, 1, 2, 3],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [0, 1, 2],
    "ca": [0, 1, 2, 3],
    "thal": [0, 1, 2, 3],
}

# Organize features into logical groups for UI
feature_groups = {
    "Demographics": ["age", "sex"],
    "Symptoms & Clinical Observations": ["cp", "trestbps", "chol", "fbs", "restecg"],
    "Exercise / Test Results": ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]
}

# Only display those present in selected_features
feature_groups = {k: [f for f in v if f in selected_features] for k, v in feature_groups.items()}

# ------------------- UI Layout -------------------
st.markdown("""
    <h1 style="color:#C70039; text-align:center;">❤️ Heart Disease Predictor</h1>
    <p style='text-align:center; color:#555; font-size:1rem'>
        Quickly assess cardiac risk using clinical and test data.<br>
        Powered by Machine Learning.
    </p>
""", unsafe_allow_html=True)
st.divider()

with st.expander("ℹ️ App Instructions", expanded=False):
    st.write("""
        - Please enter the relevant patient details below.
        - Hover on each field to see a quick explanation and known value range.
        - All predictions are private & run only in your browser.
        - Click **Predict** to get assessed risk and model confidence score.
    """)

# Input collection
user_inputs = []
with st.container():
    for section, features_in_group in feature_groups.items():
        st.subheader(section)
        for feature in features_in_group:
            min_val, max_val = feature_ranges.get(feature, (0, 1))
            desc = feature_descriptions.get(feature, "")
            helptext = f"{desc} | Typical range: {min_val}–{max_val}"

            if feature in discrete_options:
                value = st.selectbox(
                    f"{feature}",
                    options=discrete_options[feature],
                    help=helptext,
                    key=f"input_{feature}"
                )
            else:
                value = st.slider(
                    f"{feature}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    help=helptext,
                    key=f"input_{feature}"
                )
            user_inputs.append(value)

st.divider()
st.markdown("### 🩺 Prediction Results")

# ------------------- Prediction -------------------
if st.button("🔍 Predict", type="primary"):
    try:
        X_input = np.array(user_inputs).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
        bar_color = "#fa5252" if pred == 1 else "#40c057"

        # Display result in styled boxes
        if pred == 1:
            st.markdown(
                f"""
                <div style="background:#ffe3e3; padding:1rem; border-radius:1rem; border:1.5px solid #ff6f60;">
                    <h3 style="color:#C70039; font-weight:800;">⚠️ High Risk</h3>
                    <span style="font-size:1.2rem; color:#B22222;">
                    Model estimates <b>{proba:.0%}</b> chance of heart disease.
                    </span>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background:#e6fcf5; padding:1rem; border-radius:1rem; border:1.5px solid #45b39d;">
                    <h3 style="color:#00897B; font-weight:800;">✅ Low Risk</h3>
                    <span style="font-size:1.2rem; color:#257672;">
                    Model estimates <b>{proba:.0%}</b> chance of heart disease.
                    </span>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown("#### Probability Confidence")
        st.progress(proba, text=f"{proba:.1%} risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Click **Predict** to see risk estimation.")

# ------------------- Model & Feature Info -------------------
with st.expander("📊 Model & Feature Details"):
    st.write(f"**Model used:** {type(model).__name__}")
    st.markdown("**Top Features & Descriptions:**")
    
    # Create a table mapping top features to their descriptions
    top_feat_desc = [
        {
            "Feature": feat,
            "Description": feature_descriptions.get(feat, "No description available")
        }
        for feat in selected_features
    ]
    st.table(pd.DataFrame(top_feat_desc))
    
    st.markdown("**Feature Value Ranges (in dataset):**")
    st.dataframe(
        pd.DataFrame(feature_ranges.items(), columns=["Feature", "Value Range"])
        .set_index("Feature")
    )

    st.markdown(
        "<span style='font-size:0.95rem; color:#888'>This tool is for educational/demo purposes only. Not medical advice.</span>",
        unsafe_allow_html=True
    )
