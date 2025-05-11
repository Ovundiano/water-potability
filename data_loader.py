import pandas as pd
import streamlit as st


def load_data():
    """Load and return the water potability dataset"""
    try:
        df = pd.read_csv("data/water_potability.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload your data.")
        return None
