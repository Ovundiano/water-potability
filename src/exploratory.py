import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def data_overview(df):
    """Display data overview section"""
    st.header("1. Data Overview")

    # Show raw data explorer
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(df)

    # Display data info
    st.subheader("Data Information")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Shape**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.markdown("**Data Types**")
        buffer = pd.DataFrame(df.dtypes, columns=["Data Type"])
        st.dataframe(buffer)

    with col2:
        st.markdown("**Missing Values**")
        missing_data = pd.DataFrame(df.isna().sum(), columns=["Missing Values"])
        missing_data["Percentage"] = round(
            missing_data["Missing Values"] / len(df) * 100, 2
        )
        st.dataframe(missing_data)

        st.markdown("**Target Distribution**")
        if "Potability" in df.columns:
            potability_counts = df["Potability"].value_counts().reset_index()
            potability_counts.columns = ["Potability", "Count"]
            fig = px.pie(
                potability_counts,
                values="Count",
                names="Potability",
                title="Distribution of Potable vs Non-Potable Water",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            st.plotly_chart(fig)
        else:
            st.warning("Target column 'Potability' not found in the dataset")

    # Data summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.markdown("---")


def exploratory_data_analysis(df):
    """Display exploratory data analysis section"""
    st.header("3. Exploratory Data Analysis")

    # Distribution of features
    st.subheader("Distribution of Features")

    selected_feature = st.selectbox(
        "Select feature to visualize distribution:",
        df.columns[:-1],  # All columns except the target column
    )

    fig = make_subplots(rows=1, cols=2)

    # Histogram
    fig.add_trace(
        go.Histogram(x=df[selected_feature], nbinsx=30, name="Histogram"), row=1, col=1
    )

    # KDE
    kde_y, kde_x = np.histogram(df[selected_feature].dropna(), bins=30, density=True)
    kde_x = (kde_x[:-1] + kde_x[1:]) / 2  # Get bin centers

    fig.add_trace(
        go.Scatter(x=kde_x, y=kde_y, mode="lines", name="Density"), row=1, col=1
    )

    # Box plot
    fig.add_trace(go.Box(y=df[selected_feature], name="Box Plot"), row=1, col=2)

    fig.update_layout(
        title=f"Distribution of {selected_feature}", height=400, width=800
    )

    st.plotly_chart(fig)

    # Correlation analysis
    st.subheader("Correlation Analysis")

    corr_matrix = df.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
    )

    st.plotly_chart(fig)

    # Feature relationships with target
    st.subheader("Feature Relationships with Potability")

    if "Potability" in df.columns:
        selected_features = st.multiselect(
            "Select features to visualize relationship with potability:",
            df.columns[:-1],
            default=df.columns[0],
        )

        if selected_features:
            for feature in selected_features:
                fig = px.box(
                    df,
                    x="Potability",
                    y=feature,
                    title=f"{feature} vs Potability",
                    color="Potability",
                    labels={
                        "Potability": "Water Potability (0: Not Potable, 1: Potable)"
                    },
                )

                st.plotly_chart(fig)

    # Pairwise relationships
    st.subheader("Pairwise Feature Relationships")

    if st.checkbox("Show pairplot (Warning: May be slow for large datasets)"):
        selected_cols = st.multiselect(
            "Select features for pairplot (recommended: max 4):",
            df.columns,
            default=list(df.columns[:3]) + ["Potability"],
        )

        if len(selected_cols) > 1:
            fig = px.scatter_matrix(
                df[selected_cols],
                dimensions=(
                    selected_cols[:-1]
                    if "Potability" in selected_cols
                    else selected_cols
                ),
                color="Potability" if "Potability" in selected_cols else None,
                title="Pairwise Feature Relationships",
            )

            fig.update_layout(height=800, width=800)
            st.plotly_chart(fig)

    st.markdown("---")


def feature_importance_analysis(df):
    """Display feature importance analysis"""
    st.header("4. Feature Importance Analysis")

    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]

        # Random Forest for feature importance
        from sklearn.ensemble import RandomForestClassifier

        st.subheader("Random Forest Feature Importance")

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)

        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": rf.feature_importances_}
        ).sort_values("Importance", ascending=False)

        fig = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance from Random Forest",
            color="Importance",
            color_continuous_scale="viridis",
        )

        st.plotly_chart(fig)

        # Correlation with target
        st.subheader("Feature Correlation with Target")

        target_corr = pd.DataFrame(abs(df.corr()["Potability"])).reset_index()
        target_corr.columns = ["Feature", "Correlation"]
        target_corr = target_corr[target_corr["Feature"] != "Potability"]
        target_corr = target_corr.sort_values("Correlation", ascending=False)

        fig = px.bar(
            target_corr,
            x="Correlation",
            y="Feature",
            orientation="h",
            title="Absolute Correlation with Potability",
            color="Correlation",
            color_continuous_scale="viridis",
        )

        st.plotly_chart(fig)

    st.markdown("---")
