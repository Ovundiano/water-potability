import streamlit as st
import pandas as pd
from pathlib import Path
from src.data_loader import load_data
from src.preprocessing import data_preprocessing
from src.exploratory import (
    data_overview,
    exploratory_data_analysis,
    feature_importance_analysis,
)
from src.modeling import build_model
from src.prediction import prediction_interface
from src.visualization import display_header, conclusions_and_recommendations
import os
import sys

sys.path.append(str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="Water Potability Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "df" not in st.session_state:
    st.session_state["df"] = None
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None


def main():
    """Main function to run the Streamlit app for water potability analysis."""
    display_header()

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a section to explore the water potability analysis.")

    st.sidebar.header("1. Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Use example dataset", "Upload your own dataset"],
        key="data_source",
    )

    if data_source == "Use example dataset":
        try:
            st.session_state["df"] = load_data()
            if st.session_state["df"] is not None:
                st.sidebar.success("Example dataset loaded successfully!")
            else:
                st.sidebar.error("Failed to load example dataset. Check file path.")
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {str(e)}")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file", type=["csv"], key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                st.session_state["df"] = load_data(uploaded_file)
                st.sidebar.success("Uploaded dataset loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading uploaded file: {str(e)}")
        else:
            st.sidebar.warning(
                "Please upload a CSV file or select the example dataset."
            )

    if st.session_state["df"] is not None:
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
            key="page_selector",
        )

        preprocess = st.sidebar.checkbox(
            "Apply preprocessing", value=True, key="preprocess_toggle"
        )

        if preprocess and page != "Data Overview":
            with st.spinner("Preprocessing data..."):
                try:
                    st.session_state["processed_df"] = data_preprocessing(
                        st.session_state["df"]
                    )
                    if st.session_state["processed_df"] is not None:
                        st.sidebar.success("Data preprocessing completed!")
                    else:
                        st.sidebar.error("Preprocessing failed. Check dataset.")
                except Exception as e:
                    st.sidebar.error(f"Preprocessing error: {str(e)}")
        else:
            st.session_state["processed_df"] = st.session_state["df"].copy()

        try:
            if page == "Data Overview":
                data_overview(st.session_state["df"])
            elif page == "Data Preprocessing":
                st.session_state["processed_df"] = data_preprocessing(
                    st.session_state["df"]
                )
            elif page == "Exploratory Data Analysis":
                exploratory_data_analysis(st.session_state["processed_df"])
            elif page == "Feature Importance Analysis":
                feature_importance_analysis(st.session_state["processed_df"])
            elif page == "Model Building and Evaluation":
                build_model(st.session_state["processed_df"])
            elif page == "Prediction Interface":
                prediction_interface(st.session_state["processed_df"])
            elif page == "Conclusions and Recommendations":
                conclusions_and_recommendations()
        except Exception as e:
            st.error(f"An error occurred while rendering the page: {str(e)}")
            st.markdown("Please check the dataset and try again.")
    else:
        st.warning("No dataset loaded. Please select a data source in the sidebar.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    main()
