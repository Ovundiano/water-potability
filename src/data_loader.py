import pandas as pd
import streamlit as st
from pathlib import Path


def load_data(file_path="Data/water_potability.csv"):
    """Load the water potability dataset from a specified path or uploaded file.

    Args:
        file_path (str, optional): Path to the dataset. Defaults to 'Data/water_potability.csv'.

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, None otherwise.
    """
    try:
        if isinstance(file_path, str):
            data_path = Path(file_path)
            if not data_path.exists():
                st.error(f"Dataset file not found at: {data_path}")
                return None
            df = pd.read_csv(data_path)
        else:
            df = pd.read_csv(file_path)

        if df.empty:
            st.error("Loaded dataset is empty.")
            return None
        if "Potability" not in df.columns:
            st.error("Dataset must contain a 'Potability' column.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
