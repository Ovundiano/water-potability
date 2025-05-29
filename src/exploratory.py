import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


def data_overview(df):
    """Display an overview of the dataset with basic statistics and visualizations."""
    st.header("Data Overview")

    with st.expander("View Raw Data", expanded=False):
        st.subheader("Raw Data")
        st.dataframe(df)

    st.subheader("Data Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dataset Shape**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.markdown("**Data Types**")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    with col2:
        st.markdown("**Missing Values**")
        missing_data = pd.DataFrame(df.isna().sum(), columns=["Missing Values"])
        missing_data["Percentage"] = (
            missing_data["Missing Values"] / len(df) * 100
        ).round(2)
        st.dataframe(missing_data)

        if "Potability" in df.columns:
            st.markdown("**Target Distribution**")
            potability_counts = df["Potability"].value_counts().reset_index()
            potability_counts.columns = ["Potability", "Count"]
            fig = px.pie(
                potability_counts,
                values="Count",
                names=["Not Potable", "Potable"],
                title="Distribution of Potable vs Non-Potable Water",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Target column 'Potability' not found.")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    st.markdown("---")


def exploratory_data_analysis(df):
    """Perform exploratory data analysis with visualizations."""
    st.header("Exploratory Data Analysis")

    st.subheader("Feature Distributions")
    selected_feature = st.selectbox(
        "Select feature to visualize:", df.columns[:-1], key="feature_dist_select"
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram & KDE", "Box Plot"])
    fig.add_trace(
        go.Histogram(x=df[selected_feature], nbinsx=30, name="Histogram"), row=1, col=1
    )
    kde_y, kde_x = np.histogram(df[selected_feature].dropna(), bins=30, density=True)
    kde_x = (kde_x[:-1] + kde_x[1:]) / 2
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines", name="KDE"), row=1, col=1)
    fig.add_trace(go.Box(y=df[selected_feature], name="Box Plot"), row=1, col=2)
    fig.update_layout(
        title=f"Distribution of {selected_feature}", height=400, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Analysis")
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

    if "Potability" in df.columns:
        st.subheader("Feature Relationships with Potability")
        selected_features = st.multiselect(
            "Select features to compare with Potability:",
            df.columns[:-1],
            default=[df.columns[0]],
            key="feature_vs_potability",
        )

        for feature in selected_features:
            fig = px.box(
                df,
                x="Potability",
                y=feature,
                color="Potability",
                title=f"{feature} vs Potability",
                labels={"Potability": "Water Potability (0: Not Potable, 1: Potable)"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Statistical Significance (T-tests)", expanded=False):
            results = []
            for col in df.columns[:-1]:
                group0 = df[df["Potability"] == 0][col]
                group1 = df[df["Potability"] == 1][col]
                t_stat, p_val = stats.ttest_ind(group0, group1, nan_policy="omit")
                results.append({"Feature": col, "p-value": round(p_val, 4)})
            st.dataframe(
                pd.DataFrame(results).sort_values("p-value"), use_container_width=True
            )

    with st.expander("Pairwise Feature Relationships", expanded=False):
        selected_cols = st.multiselect(
            "Select features for pairplot (max 4 recommended):",
            df.columns,
            default=list(df.columns[:3]) + ["Potability"],
            key="pairplot_select",
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
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


def feature_importance_analysis(df):
    """Analyze feature importance using Random Forest and correlation."""
    st.header("Feature Importance Analysis")

    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]

        st.subheader("Random Forest Feature Importance")
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
            )
            rf.fit(X, y)
            feature_importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": rf.feature_importances_}
            ).sort_values("Importance", ascending=False)
            fig = px.bar(
                feature_importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (Random Forest)",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Random Forest feature importance: {str(e)}")

        st.subheader("Correlation with Potability")
        target_corr = pd.DataFrame(
            abs(df.corr()["Potability"]), columns=["Correlation"]
        ).reset_index()
        target_corr = target_corr[target_corr["index"] != "Potability"]
        target_corr = target_corr.sort_values("Correlation", ascending=False)
        fig = px.bar(
            target_corr,
            x="Correlation",
            y="index",
            orientation="h",
            title="Absolute Correlation with Potability",
            color="Correlation",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Target column 'Potability' not found.")

    st.markdown("---")
