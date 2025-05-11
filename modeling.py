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
    """Display model building and evaluation section"""
    st.header("5. Model Building and Evaluation")

    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]

        # Train-test split
        st.subheader("Train-Test Split")

        test_size = (
            st.slider(
                "Select test set size (%):",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
            )
            / 100
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        st.write(f"Training set: {X_train.shape[0]} samples")
        st.write(f"Test set: {X_test.shape[0]} samples")

        # Model selection
        st.subheader("Model Selection")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True, random_state=42),
            "XGBoost": XGBClassifier(random_state=42),
        }

        selected_models = st.multiselect(
            "Select models to evaluate:",
            list(models.keys()),
            default=["Logistic Regression", "Random Forest", "XGBoost"],
        )

        if not selected_models:
            st.warning("Please select at least one model for evaluation.")
        else:
            # Model evaluation with cross-validation
            st.subheader("Model Evaluation with Cross-Validation")

            cv_folds = st.slider(
                "Select number of cross-validation folds:",
                min_value=3,
                max_value=10,
                value=5,
            )

            results = {}

            for model_name in selected_models:
                model = models[model_name]

                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv_folds, scoring="accuracy"
                )

                results[model_name] = {
                    "Mean CV Accuracy": round(cv_scores.mean(), 4),
                    "CV Std Dev": round(cv_scores.std(), 4),
                }

            # Display cross-validation results
            results_df = pd.DataFrame.from_dict(results, orient="index")
            results_df = results_df.sort_values("Mean CV Accuracy", ascending=False)

            st.dataframe(results_df)

            # Bar chart of model performance
            fig = px.bar(
                results_df.reset_index().rename(columns={"index": "Model"}),
                x="Model",
                y="Mean CV Accuracy",
                error_y="CV Std Dev",
                title="Cross-Validation Accuracy by Model",
                color="Mean CV Accuracy",
                color_continuous_scale="viridis",
            )

            st.plotly_chart(fig)

            # Best model evaluation on test set
            st.subheader("Best Model Evaluation on Test Set")

            best_model_name = st.selectbox(
                "Select best model for detailed evaluation:",
                selected_models,
                index=0 if selected_models else None,
            )

            if best_model_name:
                best_model = models[best_model_name]

                # Train the model
                best_model.fit(X_train, y_train)

                # Make predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]

                # Model evaluation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Create columns for metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")

                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                    st.metric("F1 Score", f"{f1:.4f}")

                # Classification Report
                st.subheader("Classification Report")
                cr = classification_report(y_test, y_pred, output_dict=True)
                cr_df = pd.DataFrame(cr).transpose()
                st.dataframe(cr_df)

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Not Potable (0)", "Potable (1)"],
                    y=["Not Potable (0)", "Potable (1)"],
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                )

                st.plotly_chart(fig)

                # ROC Curve
                st.subheader("ROC Curve")

                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                fig = px.line(
                    x=fpr,
                    y=tpr,
                    labels=dict(x="False Positive Rate", y="True Positive Rate"),
                    title=f"ROC Curve (AUC = {roc_auc:.4f})",
                )

                fig.add_shape(
                    type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1
                )

                st.plotly_chart(fig)

                # Feature importance (if available)
                if hasattr(best_model, "feature_importances_"):
                    st.subheader("Feature Importance")

                    feature_imp = pd.DataFrame(
                        {
                            "Feature": X.columns,
                            "Importance": best_model.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)

                    fig = px.bar(
                        feature_imp,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Feature Importance ({best_model_name})",
                        color="Importance",
                        color_continuous_scale="viridis",
                    )

                    st.plotly_chart(fig)

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

                    st.plotly_chart(fig)

    st.markdown("---")
