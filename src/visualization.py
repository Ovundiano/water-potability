import streamlit as st


def display_header():
    """Display the app header and introduction."""
    st.title("Water Potability Analysis App")
    st.markdown(
        """
        This app provides a comprehensive analysis of water quality to predict potability using machine learning.

        **Key Features:**
        - **Data Exploration**: Visualize raw data, statistics, and distributions.
        - **Preprocessing**: Handle missing values, outliers, and feature scaling.
        - **EDA**: Analyze feature relationships and correlations.
        - **Modeling**: Train and evaluate multiple machine learning models.
        - **Prediction**: Predict water potability for new samples.
        - **Insights**: Gain actionable recommendations based on analysis.

        **Dataset**: The dataset includes water quality metrics such as pH, Hardness, Solids, and Potability.
        **Libraries**: Pandas, NumPy, Plotly, Scikit-learn, XGBoost.
        """
    )

    st.write(
        f"* For more in depth information, you can check out the associated "
        f"[README](https://github.com/Ovundiano/water-potability/tree/main?tab=readme-ov-file) file."
    )

    st.write(
        f"* You can explore water potability and related datasets on the "
        f"[FAO website](https://www.kaggle.com/code/nimapourmoradi/water-potability)."
    )

    st.markdown("---")


def conclusions_and_recommendations():
    """Display conclusions and recommendations based on the analysis."""
    st.header("Conclusions and Recommendations")
    st.markdown(
        """
        **Summary of Findings:**
        - The dataset reveals complex relationships between water quality parameters and potability.
        - Engineered features (e.g., pH-Hardness interaction) improve model performance.
        - Random Forest and XGBoost models consistently perform well, with AUC scores above 0.7.
        - Key features influencing potability include pH, Hardness, and Solids.

        **Recommendations:**
        - **Monitoring**: Prioritize regular testing of high-impact features identified by the model.
        - **Validation**: Use this tool for preliminary screening, followed by laboratory confirmation.
        - **Data Collection**: Gather more balanced data to enhance model accuracy.
        - **Feature Expansion**: Consider adding new water quality parameters (e.g., microbial content) for better predictions.

        **Next Steps:**
        - Deploy this app in water treatment facilities for real-time analysis.
        - Integrate with IoT devices for automated data collection.
        - Explore ensemble methods to further improve prediction accuracy.
        """
    )

    st.write(
        f"* For more in depth information, you can check out the associated "
        f"[README](https://github.com/Ovundiano/water-potability/tree/main?tab=readme-ov-file) file."
    )

    st.write(
        f"* You can explore water potability and related datasets on the "
        f"[FAO website](https://www.kaggle.com/code/nimapourmoradi/water-potability)."
    )

    st.markdown("---")
