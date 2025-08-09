import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

hide_menu = """
<style>
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
.css-1rs6os.edgvbvh3 {visibility: hidden;}
.stProgress > div > div > div > div {background-image: linear-gradient(90deg, #ff8787, #ffe066);}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# ---- Load Model Assets ----
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    selected_features = pickle.load(open("features.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Could not load model/scaler/features: {e}")
    st.stop()

# ---- Load dataset to determine feature ranges ----
try:
    df = pd.read_csv("heart.csv")
    feature_ranges = {col: (float(df[col].min()), float(df[col].max())) for col in selected_features}
except Exception as e:
    st.warning(f"⚠️ Could not auto-load value ranges, using default values: {e}")
    feature_ranges = {f: (0., 100.) for f in selected_features}

discrete_options = {
    "sex": [0, 1],
    "cp": [0, 1, 2, 3],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "thal": [0, 1, 2, 3],
}

# ---- HEADER ----
st.markdown(
    """
    <h1 style="color:#C70039; text-align:center; font-weight:700; letter-spacing:1px">
    ❤️ Heart Disease Predictor
    </h1>
    <p style='text-align:center; color:#555; font-size:1.1rem'>
        Predict your <span style="color:#C70039"><b>risk</b></span> of heart disease quickly and privately.
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---- Instructions ----
with st.expander("ℹ️ How to use"):
    st.markdown("""
        1. Update your details for each feature below.
        2. Hover on a label to see the usual value range.
        3. Click <b>'Predict'</b> and get your result instantly.
        <br/><br/>
        <i>All predictions are private & local to your browser.</i>
    """, unsafe_allow_html=True)

# ---- INPUT CARD ----
with st.container():
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("Enter Details")
        st.markdown("Set the values for each feature below. These are the ones that influence risk the most.")

        # Dynamically construct inputs
        user_inputs = []
        for feature in selected_features:
            min_val, max_val = feature_ranges.get(feature, (0, 1))
            range_tooltip = f"Typical range: <b>{min_val}</b> – <b>{max_val}</b>"

            # Use dropdowns for discrete/categorical, sliders for numeric
            if feature in discrete_options:
                value = st.selectbox(
                    f"{feature}",
                    options=discrete_options[feature],
                    help=range_tooltip,
                    key=f"input_{feature}"
                )
            else:
                value = st.slider(
                    f"{feature}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    help=range_tooltip,
                    key=f"input_{feature}"
                )
            user_inputs.append(value)

    # Some visual: show the relevant heart image / emoji in the right column for visual balance
    with col2:
        st.image(
            "https://cdn.pixabay.com/photo/2014/04/03/10/32/heartbeat-312769_1280.png",
            use_container_width="always",
            caption="Your heart data stays private.",
        )

st.divider()
st.markdown("### Results")

# ---- PREDICTION ----
if st.button("🔍 Predict", type="primary"):
    try:
        X_input = np.array(user_inputs).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
        bar_color = "#fa5252" if pred == 1 else "#40c057"

        # Display prediction result in a styled info box
        if pred == 1:
            st.markdown(
                f"""
                <div style="background:#ffe3e3; padding:1.5rem 1rem 0.75rem 1rem; border-radius:1rem; border:1.5px solid #ff6f60">
                <h2 style="color:#C70039; font-weight:800; margin-top:0">⚠️ High Risk</h2>
                <span style="font-size:1.3rem; color:#B22222">
                    Our model estimates a <b>{proba:.0%}</b> chance of heart disease.
                </span>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background:#e6fcf5; padding:1.5rem 1rem 0.75rem 1rem; border-radius:1rem; border:1.5px solid #45b39d">
                <h2 style="color:#00897B; font-weight:800; margin-top:0">✅ Low Risk</h2>
                <span style="font-size:1.2rem; color:#257672">
                    Our model estimates a <b>{proba:.0%}</b> chance of heart disease.
                </span>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown(f"#### Probability")
        st.progress(proba, text=f"{proba:.1%} risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Click **Predict** to see your risk.")

# ---- FOOTER/MODEL DETAILS ----
with st.expander("📊 Model & Data Info"):
    st.write(f"**Model used:** {type(model).__name__}")
    st.write(f"**Top features:** `{', '.join(selected_features)}`")
    st.write("**Feature Value Ranges:**")
    st.dataframe(
        pd.DataFrame(feature_ranges.items(), columns=["Feature", "Value Range"])
        .set_index("Feature")
    )
    st.markdown(
        "<span style='font-size:0.99rem; color:#888'>This tool is for educational/demo use only — not a substitute for professional medical advice.</span>",
        unsafe_allow_html=True
    )
