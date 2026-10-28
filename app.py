import streamlit as st
import pandas as pd
import joblib

# Load the existing trained model and preprocessing objects
package = joblib.load("Final_crop_recommendation_model.pkl")

model = package["model"]
scaler = package["scaler"]
le = package["label_encoder"]
feature_names = package["feature_names"]

# Page configuration
st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 AI Crop Recommendation System")
st.write(
    "Enter your soil nutrient and environmental values "
    "to explore the crops predicted by the trained model."
)

st.info(
    "Predictions are based on the training dataset. "
    "They are not a guarantee of real-world crop suitability."
)

# Soil nutrient inputs
st.subheader("Soil Nutrients")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input(
        "Nitrogen (N)", min_value=0.0, value=90.0
    )

with col2:
    P = st.number_input(
        "Phosphorus (P)", min_value=0.0, value=42.0
    )

with col3:
    K = st.number_input(
        "Potassium (K)", min_value=0.0, value=43.0
    )

# Environmental inputs
st.subheader("Environmental Conditions")

temperature = st.number_input(
    "Temperature (°C)", value=25.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

ph = st.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5
)

rainfall = st.number_input(
    "Rainfall (mm)", min_value=0.0, value=200.0
)

# Predict top 3 crops
if st.button("🌾 Recommend Top 3 Crops", type="primary"):

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=feature_names
    )

    scaled_data = scaler.transform(input_data)

    # Random Forest class probabilities
    probabilities = model.predict_proba(scaled_data)[0]

    # Get indices of the 3 highest probabilities
    top_indices = probabilities.argsort()[-3:][::-1]

    st.subheader("🌿 Top 3 Model Predictions")

    results = []

    for rank, index in enumerate(top_indices, start=1):
        crop_name = le.inverse_transform([index])[0]
        confidence = probabilities[index] * 100

        results.append({
            "Rank": rank,
            "Crop": crop_name.title(),
            "Model confidence score": f"{confidence:.2f}%"
        })

    for result in results:
        st.markdown(f"### {result['Rank']}. {result['Crop']}")
        st.progress(
            int(round(float(result["Model confidence score"][:-1]))),
            text=(
                "Model confidence score: "
                f"{result['Model confidence score']}"
            )
        )

    st.caption(
        "These scores are Random Forest predicted class probabilities. "
        "They are not calibrated guarantees of correctness, and they "
        "do not represent the probability of successful crop growth."
    )

    with st.expander("View prediction table"):
        st.dataframe(
            pd.DataFrame(results),
            hide_index=True,
            use_container_width=True
        )

    st.warning(
        "Before planting, verify the recommendation using a soil test, "
        "local climate, water availability, season, and advice from "
        "a qualified agricultural expert."
    )
