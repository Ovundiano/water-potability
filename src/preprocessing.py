import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def data_preprocessing(df):
    """Handle data preprocessing including missing values, outliers, and scaling"""
    st.header("2. Data Preprocessing")

    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()

    # ----------------------------
    # Handle Missing Values
    # ----------------------------
    st.subheader("Handling Missing Values")

    impute_method = st.radio(
        "Select imputation method for missing values:",
        ["Mean", "Median", "Drop rows with missing values"],
        index=0,
    )

    if impute_method == "Mean":
        for column in processed_df.columns:
            if processed_df[column].isna().sum() > 0:
                mean_value = processed_df[column].mean()
                processed_df[column].fillna(mean_value, inplace=True)
        st.success("Missing values imputed with mean values")

    elif impute_method == "Median":
        for column in processed_df.columns:
            if processed_df[column].isna().sum() > 0:
                median_value = processed_df[column].median()
                processed_df[column].fillna(median_value, inplace=True)
        st.success("Missing values imputed with median values")

    elif impute_method == "Drop rows with missing values":
        old_shape = processed_df.shape[0]
        processed_df.dropna(inplace=True)
        st.success(
            f"Dropped {old_shape - processed_df.shape[0]} rows with missing values"
        )

    if processed_df.isna().sum().sum() > 0:
        st.warning("There are still missing values in the dataset")
    else:
        st.success("No missing values in the processed dataset")

    # ----------------------------
    # Outlier Handling
    # ----------------------------
    st.subheader("Outlier Detection and Handling")

    outlier_method = st.radio(
        "Select method for handling outliers:",
        ["Keep outliers", "Cap outliers (IQR method)", "Remove outliers"],
        index=0,
    )

    if outlier_method == "Cap outliers (IQR method)":
        for column in processed_df.columns[:-1]:  # Exclude the target column
            Q1 = processed_df[column].quantile(0.25)
            Q3 = processed_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            processed_df[column] = np.where(
                processed_df[column] < lower_bound, lower_bound, processed_df[column]
            )
            processed_df[column] = np.where(
                processed_df[column] > upper_bound, upper_bound, processed_df[column]
            )

        st.success("Outliers have been capped using the IQR method")

    elif outlier_method == "Remove outliers":
        old_shape = processed_df.shape[0]
        for column in processed_df.columns[:-1]:  # Exclude the target column
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
            f"Removed {old_shape - processed_df.shape[0]} rows containing outliers"
        )

    # ----------------------------
    # Feature Scaling
    # ----------------------------
    st.subheader("Feature Scaling")

    scaling_method = st.radio(
        "Select scaling method:",
        ["No scaling", "StandardScaler", "MinMaxScaler"],
        index=0,
    )

    if scaling_method != "No scaling":
        X = processed_df.drop("Potability", axis=1)
        y = processed_df["Potability"]

        scaler = (
            StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
        )
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        processed_df = pd.concat([X_scaled, y], axis=1)
        st.success(f"Data scaled using {scaling_method}")

    # ----------------------------
    # Save Processed Data
    # ----------------------------
    output_path = os.path.join("data", "water_potability_processed.csv")
    try:
        processed_df.to_csv(output_path, index=False)
        st.success(f"✅ Processed data saved to: {output_path}")
    except Exception as e:
        st.error(f"❌ Failed to save processed file: {e}")

    return processed_df
