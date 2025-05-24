import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path


def data_preprocessing(df):
    """Handle data preprocessing including missing values, outliers, feature engineering, and scaling.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    st.header("2. Data Preprocessing")

    processed_df = df.copy()

    # Feature Engineering
    st.subheader("Feature Engineering")
    try:
        processed_df["ph_Hardness"] = processed_df["ph"] * processed_df["Hardness"]
        processed_df["Solids_Chloramines"] = processed_df["Solids"] / (
            processed_df["Chloramines"] + 1e-6
        )
        processed_df["TDS_Conductivity"] = processed_df["Solids"] / (
            processed_df["Conductivity"] + 1e-6
        )
        processed_df["Organic_Carbon_ratio"] = processed_df["Organic_carbon"] / (
            processed_df["Trihalomethanes"] + 1e-6
        )
        st.success(
            "Engineered features added: ph_Hardness, Solids_Chloramines, TDS_Conductivity, Organic_Carbon_ratio"
        )
    except Exception as e:
        st.warning(
            f"Feature engineering failed: {str(e)}. Proceeding without new features."
        )

    # Handle Missing Values
    st.subheader("Handling Missing Values")
    impute_method = st.radio(
        "Select imputation method for missing values:",
        ["Mean", "Median", "Drop rows"],
        index=1,
        key="impute_method",
    )

    try:
        if impute_method == "Mean":
            for column in processed_df.columns:
                if processed_df[column].isna().sum() > 0:
                    mean_value = processed_df[column].mean()
                    processed_df[column].fillna(mean_value, inplace=True)
            st.success("Missing values imputed with mean values.")
        elif impute_method == "Median":
            for column in processed_df.columns:
                if processed_df[column].isna().sum() > 0:
                    median_value = processed_df[column].median()
                    processed_df[column].fillna(median_value, inplace=True)
            st.success("Missing values imputed with median values.")
        elif impute_method == "Drop rows":
            old_shape = processed_df.shape[0]
            processed_df.dropna(inplace=True)
            st.success(
                f"Dropped {old_shape - processed_df.shape[0]} rows with missing values."
            )

        if processed_df.isna().sum().sum() > 0:
            st.warning("Some missing values remain in the dataset.")
        else:
            st.success("All missing values handled successfully.")
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")

    # Outlier Handling
    st.subheader("Outlier Handling")
    outlier_method = st.radio(
        "Select method for handling outliers:",
        ["Keep outliers", "Cap outliers (IQR)", "Remove outliers"],
        index=1,
        key="outlier_method",
    )

    try:
        if outlier_method == "Cap outliers (IQR)":
            for column in processed_df.columns[:-1]:  # Exclude Potability
                Q1 = processed_df[column].quantile(0.25)
                Q3 = processed_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_df[column] = np.where(
                    processed_df[column] < lower_bound,
                    lower_bound,
                    processed_df[column],
                )
                processed_df[column] = np.where(
                    processed_df[column] > upper_bound,
                    upper_bound,
                    processed_df[column],
                )
            st.success("Outliers capped using IQR method.")
        elif outlier_method == "Remove outliers":
            old_shape = processed_df.shape[0]
            for column in processed_df.columns[:-1]:  # Exclude Potability
                Q1 = processed_df[column].quantile(0.25)
                Q3 = processed_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_df = processed_df[
                    (processed_df[column] >= lower_bound)
                    & (processed_df[column] <= upper_bound)
                ]
            st.success(
                f"Removed {old_shape - processed_df.shape[0]} rows containing outliers."
            )
    except Exception as e:
        st.error(f"Error handling outliers: {str(e)}")

    # Feature Scaling
    st.subheader("Feature Scaling")
    scaling_method = st.radio(
        "Select scaling method:",
        ["No scaling", "StandardScaler", "MinMaxScaler"],
        index=1,
        key="scaling_method",
    )

    try:
        if scaling_method != "No scaling":
            X = processed_df.drop("Potability", axis=1)
            y = processed_df["Potability"]
            scaler = (
                StandardScaler()
                if scaling_method == "StandardScaler"
                else MinMaxScaler()
            )
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), columns=X.columns, index=X.index
            )
            processed_df = pd.concat([X_scaled, y], axis=1)
            st.success(f"Features scaled using {scaling_method}.")
    except Exception as e:
        st.error(f"Error scaling features: {str(e)}")

    # Save Processed Data
    try:
        output_path = Path("Data/water_potability_processed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        st.success(f"Processed dataset saved to: {output_path}")
    except Exception as e:
        st.error(f"Failed to save processed dataset: {str(e)}")

    return processed_df
