import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


def prediction_interface(df):
    """Provide an interface for predicting water potability based on user inputs."""
    st.header("Water Potability Prediction")

    model_features = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]

    missing_features = [f for f in model_features if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {', '.join(missing_features)}")
        return

    model_path = Path("models/best_random_forest_model.pkl")
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    st.subheader("Enter Water Quality Parameters")
    user_input = {}

    col1, col2 = st.columns(2)
    for i, feature in enumerate(model_features):
        with col1 if i % 2 == 0 else col2:
            user_input[feature] = st.slider(
                f"{feature}:",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean()),
                format="%.4f",
                key=f"slider_{feature}",
            )

    if st.button("Predict Water Potability", type="primary"):
        try:
            input_df = pd.DataFrame([user_input])[model_features]
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(
                    "POTABLE: This water is predicted to be safe for consumption! 🎉"
                )
                st.balloons()
            else:
                st.error(
                    "NOT POTABLE: This water is predicted to be unsafe for consumption! ⚠️"
                )

            potable_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            st.write(f"Probability of being potable: {potable_prob:.4f}")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=potable_prob,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Probability of Potability"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.4], "color": "red"},
                            {"range": [0.4, 0.7], "color": "yellow"},
                            {"range": [0.7, 1], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": 0.5,
                        },
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Contribution")
                feature_imp = pd.DataFrame(
                    {
                        "Feature": model_features,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)

                fig = px.bar(
                    feature_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance for Prediction",
                    color="Importance",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please check your model and input data format.")
