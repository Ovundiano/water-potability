---
name: User Story 1
about: As a data analyst, I want to visualize distributions of pH, turbidity, and
  chemical levels so I can identify contamination patterns.
title: 'USER STORY: <TITLE>'
labels: ''
assignees: ''

---
As a data analyst, I want to visualize distributions of pH, turbidity, and chemical levels so I can identify contamination patterns.

Acceptance Criteria:
- AC1: Interactive histograms/box plots in exploratory.py load within 3 seconds.
- AC2: T-tests for potable vs. non-potable groups (p-values <0.05 trigger warnings).
 
Technical Notes:
AC1: Use Plotly/Dash for interactive visualizations.
AC2: Cache pre-aggregated data to meet performance requirements.
AC3: Statistical tests should run on-demand (lazy evaluation).


---

name: User Story 2
about: Validate Feature Importance 
title: 'water-potability'
labels: ''
assignees: Ovundiano

---

As a model developer, I need to confirm top predictive features (e.g., pH, chloramines) to prioritize monitoring efforts.

Acceptance Criteria:

AC1: modeling.py displays feature importance plots.
AC2: Random Forest highlights turbidity with >20% weight.
Technical Notes:

AC1: Use SHAP values alongside traditional feature importance.
AC2: Ensure plots are exportable (PNG/SVG).

---

name: User Story 3
about: Predict Water Safety
title: 'water-potability'
labels: ''
assignees: Ovundiano

---

As a health inspector, I want to input water metrics via sliders and get instant potability predictions to flag unsafe samples.

Acceptance Criteria:

AC1: prediction.py returns results in <2 seconds with risk probability.
AC2: Threshold alerts appear for pH <6.5 or turbidity >5 NTU.
Technical Notes:

AC1: Deploy model via FastAPI/Flask for low-latency inference.
AC2: Slider inputs should validate ranges (e.g., pH 0-14).

---

name: User Story 4
about: Upload Local Data
title: 'water-potability'
labels: ''
assignees: Ovundiano

---

As a field worker, I want to upload CSV files from rural tests to update predictions.

Acceptance Criteria:

AC1: data_loader.py processes custom files with <5% parsing errors.
AC2: Missing values are auto-imputed (mean/median logged).
Technical Notes:

AC1: Validate CSV schemas with Pandas/Pydantic.
AC1: Log imputation decisions for audit trails.

---

name: User Story 
about: View Simplified Alerts
title: 'water-potability'
labels: ''
assignees: Ovundiano

---

As a non-technical user, I need color-coded alerts (red/yellow/green) for quick assessments.

Acceptance Criteria:

AC1: visualization.py shows traffic-light indicators.
AC2: Tooltips explain metrics in plain language.
Technical Notes:

AC1: Use Bootstrap/CSS for intuitive UI.
AC2: Hover text should avoid technical jargon.

---
