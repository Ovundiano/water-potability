import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px


def prediction_interface(df):
    """Display prediction interface section"""
    st.header("6. Water Potability Prediction")

    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]

        # Train the best model (Random Forest as default)
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        st.subheader("Enter Water Quality Parameters")

        # Create two columns for input fields
        col1, col2 = st.columns(2)

        # Initialize user inputs with mean values
        user_input = {}

        # First column of inputs
        with col1:
            for column in X.columns[:5]:  # First half of features
                min_val = float(X[column].min())
                max_val = float(X[column].max())
                mean_val = float(X[column].mean())

                user_input[column] = st.slider(
                    f"{column}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    format="%.4f",
                )

        # Second column of inputs
        with col2:
            for column in X.columns[5:]:  # Second half of features
                min_val = float(X[column].min())
                max_val = float(X[column].max())
                mean_val = float(X[column].mean())

                user_input[column] = st.slider(
                    f"{column}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    format="%.4f",
                )

        # Convert inputs to DataFrame
        input_df = pd.DataFrame([user_input])

        # Make prediction
        if st.button("Predict Water Potability"):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)

            st.subheader("Prediction Result")

            # Display prediction with color
            if prediction[0] == 1:
                st.success(
                    "POTABLE: This water is predicted to be safe for consumption!"
                )
                st.balloons()
            else:
                st.error(
                    "NOT POTABLE: This water is predicted to be unsafe for consumption!"
                )

            # Display probability
            st.write(f"Probability of being potable: {probability[0][1]:.4f}")

            # Create a gauge chart for probability
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

            st.plotly_chart(fig)

            # Feature importance for this prediction
            st.subheader("Feature Contribution to Prediction")

            if hasattr(model, "feature_importances_"):
                # General feature importance
                feature_imp = pd.DataFrame(
                    {
                        "Feature": X.columns,
                        "Importance": model.feature_importances_,
                        "Value": [user_input[col] for col in X.columns],
                    }
                ).sort_values("Importance", ascending=False)

                fig = px.bar(
                    feature_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance for Prediction",
                    color="Value",
                    color_continuous_scale="RdBu_r",
                )

                st.plotly_chart(fig)

    st.markdown("---")
