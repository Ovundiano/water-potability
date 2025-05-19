import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import data_preprocessing
from src.explatory import (
    data_overview,
    exploratory_data_analysis,
    feature_importance_analysis,
)
from src.modeling import build_model
from src.prediction import prediction_interface
from src.visualization import display_header, conclusions_and_recommendations

# Set page configuration
st.set_page_config(
    page_title="Water Potability Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main function to run the Streamlit app"""
    display_header()

    # Sidebar navigation
    st.sidebar.title("Navigation")

    # Upload data or use example dataset
    st.sidebar.header("1. Data Source")

    data_source = st.sidebar.radio(
        "Select data source:", ["Use example dataset", "Upload your own data"]
    )

    if data_source == "Use example dataset":
        df = load_data()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.sidebar.warning("Please upload a dataset or select the example dataset")
            df = None

    # Only proceed if data is loaded
    if df is not None:
        # Create navigation menu
        page = st.sidebar.radio(
            "Go to:",
            [
                "Data Overview",
                "Data Preprocessing",
                "Exploratory Data Analysis",
                "Feature Importance Analysis",
                "Model Building and Evaluation",
                "Prediction Interface",
                "Conclusions and Recommendations",
            ],
        )

        # Store processed dataframe in session state
        if "processed_df" not in st.session_state:
            st.session_state["processed_df"] = df.copy()

        # Process data if needed
        if page != "Data Overview" and st.sidebar.checkbox("Run preprocessing"):
            processed_df = data_preprocessing(df)
            st.session_state["processed_df"] = processed_df
        else:
            processed_df = st.session_state["processed_df"]

        # Display selected page
        if page == "Data Overview":
            data_overview(df)

        elif page == "Data Preprocessing":
            processed_df = data_preprocessing(df)
            st.session_state["processed_df"] = processed_df

        elif page == "Exploratory Data Analysis":
            exploratory_data_analysis(processed_df)

        elif page == "Feature Importance Analysis":
            feature_importance_analysis(processed_df)

        elif page == "Model Building and Evaluation":
            build_model(processed_df)

        elif page == "Prediction Interface":
            prediction_interface(processed_df)

        elif page == "Conclusions and Recommendations":
            conclusions_and_recommendations()


if __name__ == "__main__":
    main()
