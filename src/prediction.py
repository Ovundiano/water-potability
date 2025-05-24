import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path


def prediction_interface(df):
    """Provide an interface for predicting water potability based on user inputs."""
    st.header("6. Water Potability Prediction")

    if "Potability" not in df.columns:
        st.error("Target column 'Potability' not found.")
        return

    X = df.drop("Potability", axis=1)

    # Load pre-trained Random Forest model
    model_path = Path("models/best_random_forest_model.pkl")
    try:
        if not model_path.exists():
            st.error(f"Pre-trained model not found at: {model_path}")
            return
        model = joblib.load(model_path)
        st.success("Pre-trained Random Forest model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Input interface
    st.subheader("Enter Water Quality Parameters")
    col1, col2 = st.columns(2)
    user_input = {}

    try:
        for i, column in enumerate(X.columns):
            min_val = float(X[column].min())
            max_val = float(X[column].max())
            mean_val = float(X[column].mean())
            with col1 if i % 2 == 0 else col2:
                user_input[column] = st.slider(
                    f"{column}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    format="%.4f",
                    key=f"slider_{column}",
                )
    except Exception as e:
        st.error(f"Error creating input sliders: {str(e)}")
        return

    # Prediction
    if st.button("Predict Water Potability", key="predict_button"):
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)

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

            # Feature importance
            st.subheader("Feature Contribution")
            feature_imp = pd.DataFrame(
                {"Feature": X.columns, "Importance": model.feature_importances_}
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
