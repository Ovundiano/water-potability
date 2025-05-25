import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_model(df):
    """Build and evaluate machine learning models for water potability prediction."""
    st.header("5. Model Building and Evaluation")

    if "Potability" not in df.columns:
        st.error("Target column 'Potability' not found.")
        return

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    st.subheader("Train-Test Split")
    test_size = (
        st.slider(
            "Select test set size (%):",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            key="test_size_slider",
        )
        / 100
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        st.write(f"Training set: {X_train.shape[0]} samples")
        st.write(f"Test set: {X_test.shape[0]} samples")
    except Exception as e:
        st.error(f"Error splitting data: {str(e)}")
        return

    st.subheader("Model Selection")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    }

    selected_models = st.multiselect(
        "Select models to evaluate:",
        list(models.keys()),
        default=["Random Forest", "XGBoost"],
        key="model_select",
    )

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    st.subheader("Cross-Validation Results")
    cv_folds = st.slider(
        "Select number of CV folds:",
        min_value=3,
        max_value=10,
        value=5,
        key="cv_folds_slider",
    )

    try:
        results = {}
        for model_name in selected_models:
            model = models[model_name]
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv_folds, scoring="accuracy"
            )
            results[model_name] = {
                "Mean CV Accuracy": round(cv_scores.mean(), 4),
                "CV Std Dev": round(cv_scores.std(), 4),
            }

        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df = results_df.sort_values("Mean CV Accuracy", ascending=False)
        st.dataframe(results_df, use_container_width=True)

        fig = px.bar(
            results_df.reset_index().rename(columns={"index": "Model"}),
            x="Model",
            y="Mean CV Accuracy",
            error_y="CV Std Dev",
            title="Cross-Validation Accuracy by Model",
            color="Mean CV Accuracy",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in cross-validation: {str(e)}")
        return

    st.subheader("Best Model Evaluation")
    best_model_name = st.selectbox(
        "Select model for detailed evaluation:",
        selected_models,
        key="best_model_select",
    )

    try:
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

        st.subheader("Classification Report")
        cr = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(cr).transpose(), use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Potable", "Potable"],
            y=["Not Potable", "Potable"],
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig = px.line(
            x=fpr,
            y=tpr,
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            title=f"ROC Curve (AUC = {roc_auc:.4f})",
        )
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)

        if hasattr(best_model, "feature_importances_"):
            st.subheader("Feature Importance")
            feature_imp = pd.DataFrame(
                {"Feature": X.columns, "Importance": best_model.feature_importances_}
            ).sort_values("Importance", ascending=False)
            fig = px.bar(
                feature_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Feature Importance ({best_model_name})",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif best_model_name == "Logistic Regression":
            st.subheader("Feature Coefficients")
            coef_df = pd.DataFrame(
                {"Feature": X.columns, "Coefficient": best_model.coef_[0]}
            ).sort_values("Coefficient", ascending=False)
            fig = px.bar(
                coef_df,
                x="Coefficient",
                y="Feature",
                orientation="h",
                title="Logistic Regression Coefficients",
                color="Coefficient",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")

    st.markdown("---")
