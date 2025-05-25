import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path


def data_preprocessing(df, session_id=""):
    """Handle data preprocessing including missing values, outliers, feature engineering, and scaling.

    Args:
        df (pd.DataFrame): Input DataFrame
        session_id (str): Unique identifier for the session to create unique keys

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    st.header("2. Data Preprocessing")

    # Create a copy of the original dataframe
    processed_df = df.copy()

    # Check if Potability column exists
    target_present = "Potability" in processed_df.columns

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

    # Show missing values summary before processing
    missing_values = processed_df.isna().sum()
    if missing_values.sum() > 0:
        st.write("Missing values before handling:")
        st.write(missing_values[missing_values > 0])
    else:
        st.info("No missing values found in the dataset.")
        return processed_df

    # Create unique keys by combining purpose with session_id
    impute_key = f"missing_values_impute_method_{session_id}"

    impute_method = st.radio(
        "Select imputation method for missing values:",
        ["Mean", "Median", "Drop rows", "Custom value"],
        index=1,
        key=impute_key,
    )

    try:
        if impute_method in ["Mean", "Median"]:
            for column in processed_df.columns:
                if processed_df[column].isna().sum() > 0:
                    if processed_df[column].dtype in ["float64", "int64"]:
                        fill_value = (
                            processed_df[column].mean()
                            if impute_method == "Mean"
                            else processed_df[column].median()
                        )
                        processed_df[column].fillna(fill_value, inplace=True)
                        st.info(
                            f"Column '{column}' filled with {impute_method.lower()}: {fill_value:.2f}"
                        )
                    else:
                        mode_value = processed_df[column].mode()[0]
                        processed_df[column].fillna(mode_value, inplace=True)
                        st.info(
                            f"Non-numeric column '{column}' filled with mode: {mode_value}"
                        )

            st.success(f"Missing values imputed with {impute_method.lower()} values.")

        elif impute_method == "Drop rows":
            old_shape = processed_df.shape[0]
            processed_df.dropna(inplace=True)
            new_shape = processed_df.shape[0]
            rows_dropped = old_shape - new_shape
            if rows_dropped > 0:
                st.success(
                    f"Dropped {rows_dropped} rows with missing values. New shape: {new_shape}"
                )
            else:
                st.info("No rows were dropped (no missing values found).")

        elif impute_method == "Custom value":
            for column in processed_df.columns:
                if processed_df[column].isna().sum() > 0:
                    if processed_df[column].dtype in ["float64", "int64"]:
                        default_val = processed_df[column].median()
                        custom_key = f"custom_num_{column}_{session_id}"
                        custom_val = st.number_input(
                            f"Enter value for '{column}' (default: {default_val:.2f})",
                            value=default_val,
                            key=custom_key,
                        )
                        processed_df[column].fillna(custom_val, inplace=True)
                    else:
                        default_val = processed_df[column].mode()[0]
                        custom_key = f"custom_str_{column}_{session_id}"
                        custom_val = st.text_input(
                            f"Enter value for '{column}' (default: {default_val})",
                            value=default_val,
                            key=custom_key,
                        )
                        processed_df[column].fillna(custom_val, inplace=True)
            st.success("Missing values filled with custom values.")

        # Verify missing values were handled
        remaining_missing = processed_df.isna().sum().sum()
        if remaining_missing > 0:
            st.warning(
                f"Warning: {remaining_missing} missing values remain in the dataset."
            )
            st.write("Columns with remaining missing values:")
            st.write(processed_df.isna().sum()[processed_df.isna().sum() > 0])
        else:
            st.success("All missing values successfully handled.")

    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df

    # Outlier Handling
    st.subheader("Outlier Handling")
    outlier_key = f"outlier_handling_method_{session_id}"

    outlier_method = st.radio(
        "Select method for handling outliers:",
        ["Keep outliers", "Cap outliers (IQR)", "Remove outliers"],
        index=1,
        key=outlier_key,
    )

    try:
        if outlier_method == "Cap outliers (IQR)":
            for column in processed_df.select_dtypes(
                include=["float64", "int64"]
            ).columns:
                if column != "Potability" or not target_present:
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
            for column in processed_df.select_dtypes(
                include=["float64", "int64"]
            ).columns:
                if column != "Potability" or not target_present:
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
    scaling_key = f"feature_scaling_method_{session_id}"

    scaling_method = st.radio(
        "Select scaling method:",
        ["No scaling", "StandardScaler", "MinMaxScaler"],
        index=1,
        key=scaling_key,
    )

    try:
        if scaling_method != "No scaling" and target_present:
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
        elif scaling_method != "No scaling" and not target_present:
            scaler = (
                StandardScaler()
                if scaling_method == "StandardScaler"
                else MinMaxScaler()
            )
            processed_df = pd.DataFrame(
                scaler.fit_transform(processed_df),
                columns=processed_df.columns,
                index=processed_df.index,
            )
            st.success(f"Features scaled using {scaling_method}.")
    except Exception as e:
        st.error(f"Error scaling features: {str(e)}")

    # Save Processed Data
    try:
        output_path = Path("data/water_potability_processed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        st.success(f"Processed dataset saved to: {output_path}")
    except Exception as e:
        st.error(f"Failed to save processed dataset: {str(e)}")

    return processed_df
