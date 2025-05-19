import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def display_header():
    """Display app header and introduction"""
    st.title("Water Potability Analysis App")
    st.markdown(
        """
    This app analyzes water quality parameters to predict potability using machine learning.
    
    * **Data source**: The dataset contains water quality metrics for determining potability
    * **Goal**: Predict whether water is safe for consumption based on various features
    * **Python libraries**: Pandas, NumPy, Seaborn, Matplotlib, Plotly, Scikit-learn, XGBoost
    """
    )

    st.markdown("---")


def conclusions_and_recommendations():
    """Display conclusions and recommendations section"""
    st.header("7. Conclusions and Recommendations")

    st.write(
        """
    This water potability analysis app allows you to:
    
    1. **Explore water quality data** through comprehensive visual analysis
    2. **Preprocess data** by handling missing values and outliers
    3. **Build and evaluate machine learning models** to predict water potability
    4. **Make predictions** on new water samples to determine potability
    
    **Key Findings:**
    - Water quality parameters have complex relationships that influence potability
    - Several machine learning models can effectively predict water potability
    - Certain parameters have stronger influence on water potability than others
    
    **Recommendations:**
    - Regularly monitor key water quality parameters identified as important by the model
    - Use this tool as a preliminary screening method, but confirm with laboratory testing
    - Collect more balanced data to improve model performance
    - Consider adding more water quality parameters to enhance prediction accuracy
    """
    )

    st.markdown("---")


def display_outlier_boxplots(processed_df):
    """Display boxplots to visualize outliers"""
    fig = make_subplots(rows=3, cols=3, subplot_titles=list(processed_df.columns[:-1]))

    row, col = 1, 1
    for i, column in enumerate(processed_df.columns[:-1]):
        fig.add_trace(go.Box(y=processed_df[column], name=column), row=row, col=col)
        col += 1
        if col > 3:
            col = 1
            row += 1

    fig.update_layout(height=800, width=800, showlegend=False)
    st.plotly_chart(fig)
