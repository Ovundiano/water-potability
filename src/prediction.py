import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


def prediction_interface(df):
    """Provide an interface for predicting water potability based on user inputs."""
    st.header("Water Potability Prediction")

    # Define the EXACT features the model was trained on
    original_features = [
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

    # Check for required features
    missing_features = [f for f in original_features if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {', '.join(missing_features)}")
        return

    # Load the model
    model_path = Path("models/best_random_forest_model.pkl")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Create input sliders for ONLY the original features
    st.subheader("Enter Water Quality Parameters")
    user_input = {}

    col1, col2 = st.columns(2)
    for i, feature in enumerate(original_features):
        with col1 if i % 2 == 0 else col2:
            # Get feature statistics
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())

            user_input[feature] = st.slider(
                f"{feature}:",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.4f",
                key=f"slider_{feature}",
            )

    # Prediction button
    if st.button("Predict Water Potability", key="predict_button"):
        try:
            # Create input DataFrame with ONLY the original features
            input_df = pd.DataFrame([user_input])[original_features]

            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[:, 1]

            # Display results
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success(
                    "POTABLE: This water is predicted to be safe for consumption! 🎉"
                )
                st.balloons()
            else:
                st.error(
                    "NOT POTABLE: This water is predicted to be unsafe for consumption! ⚠️"
                )

            st.write(f"Probability of being potable: {probability[0][1]:.4f}")

            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability[0][1],
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

            # Feature importance (if available)
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Contribution")
                feature_imp = pd.DataFrame(
                    {
                        "Feature": original_features,
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
            st.error(f"Error making prediction: {str(e)}")

    st.markdown("---")
