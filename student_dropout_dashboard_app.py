import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# App Configuration
# -------------------------------

st.set_page_config(
    page_title="Student Dropout Early Warning Dashboard",
    page_icon="🎓",
    layout="wide"
)

DATA_PATH = "dataset.csv"
RANDOM_STATE = 42

# -------------------------------
# Feature Groups
# -------------------------------

categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

numerical_features = [
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

# -------------------------------
# Load Data
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Create synthetic student IDs for demo purposes
    df.insert(0, "student_id", [f"STU-{i:04d}" for i in range(1, len(df) + 1)])

    # Create binary target
    df["dropout_risk"] = (df["Target"] == "Dropout").astype(int)

    return df


# -------------------------------
# Train Model
# -------------------------------

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Target", "dropout_risk", "student_id"])
    y = df["dropout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    return model


# -------------------------------
# Risk Categorization
# -------------------------------

def risk_level(prob):
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Moderate"
    return "High"


def recommended_action(prob):
    if prob < 0.30:
        return "Monitor"
    if prob < 0.60:
        return "Advisor Check-in"
    return "Academic + Financial Support Review"


# -------------------------------
# Prepare Dashboard Data
# -------------------------------

def create_dashboard_dataframe(df, model):
    X = df.drop(columns=["Target", "dropout_risk", "student_id"])
    probs = model.predict_proba(X)[:, 1]

    dashboard_df = df.copy()
    dashboard_df["dropout_probability"] = probs
    dashboard_df["risk_level"] = dashboard_df["dropout_probability"].apply(risk_level)
    dashboard_df["recommended_action"] = dashboard_df["dropout_probability"].apply(recommended_action)

    dashboard_df["Tuition fees up to date"] = dashboard_df["Tuition fees up to date"].map({1: "Yes", 0: "No"})
    dashboard_df["risk_score_pct"] = (dashboard_df["dropout_probability"] * 100).round(1)

    return dashboard_df


# -------------------------------
# Explanation Functions
# -------------------------------

def explain_student(row):
    reasons = []

    if row["Curricular units 1st sem (approved)"] <= 2:
        reasons.append("low approved units in the first semester")

    if row["Curricular units 1st sem (grade)"] < 10:
        reasons.append("lower first-semester grades")

    if row["Tuition fees up to date"] == "No":
        reasons.append("tuition fees not up to date")

    if row["Age at enrollment"] >= 25:
        reasons.append("older age at enrollment")

    if row["Scholarship holder"] == 0:
        reasons.append("no scholarship support")

    if len(reasons) == 0:
        return "Risk appears relatively low based on the available indicators."

    if len(reasons) == 1:
        return f"Elevated risk appears linked to {reasons[0]}."

    if len(reasons) == 2:
        return f"Elevated risk appears linked to {reasons[0]} and {reasons[1]}."

    return "Elevated risk appears linked to " + ", ".join(reasons[:-1]) + f", and {reasons[-1]}."


def get_risk_factor_rows(row):
    factor_rows = []

    approved_units = row["Curricular units 1st sem (approved)"]
    if approved_units <= 2:
        factor_rows.append({
            "Factor": "Curricular units 1st sem (approved)",
            "Value": approved_units,
            "Impact": "Higher risk",
            "Meaning": "Very low number of approved units in the first semester"
        })
    elif approved_units >= 5:
        factor_rows.append({
            "Factor": "Curricular units 1st sem (approved)",
            "Value": approved_units,
            "Impact": "Lower risk",
            "Meaning": "Stronger first-semester academic progress"
        })

    first_sem_grade = row["Curricular units 1st sem (grade)"]
    if first_sem_grade < 10:
        factor_rows.append({
            "Factor": "Curricular units 1st sem (grade)",
            "Value": round(float(first_sem_grade), 2),
            "Impact": "Higher risk",
            "Meaning": "Lower first-semester average grade"
        })
    elif first_sem_grade >= 12:
        factor_rows.append({
            "Factor": "Curricular units 1st sem (grade)",
            "Value": round(float(first_sem_grade), 2),
            "Impact": "Lower risk",
            "Meaning": "Solid first-semester academic performance"
        })

    tuition_status = row["Tuition fees up to date"]
    if tuition_status == "No":
        factor_rows.append({
            "Factor": "Tuition fees up to date",
            "Value": tuition_status,
            "Impact": "Higher risk",
            "Meaning": "Tuition fees are not up to date"
        })
    else:
        factor_rows.append({
            "Factor": "Tuition fees up to date",
            "Value": tuition_status,
            "Impact": "Lower risk",
            "Meaning": "Tuition fees are up to date"
        })

    scholarship_status = "Yes" if row["Scholarship holder"] == 1 else "No"
    if scholarship_status == "No":
        factor_rows.append({
            "Factor": "Scholarship holder",
            "Value": scholarship_status,
            "Impact": "Higher risk",
            "Meaning": "No scholarship support recorded"
        })
    else:
        factor_rows.append({
            "Factor": "Scholarship holder",
            "Value": scholarship_status,
            "Impact": "Lower risk",
            "Meaning": "Scholarship support may reduce financial pressure"
        })

    age_at_enrollment = int(row["Age at enrollment"])
    if age_at_enrollment >= 25:
        factor_rows.append({
            "Factor": "Age at enrollment",
            "Value": age_at_enrollment,
            "Impact": "Higher risk",
            "Meaning": "Older students may face additional external responsibilities"
        })
    elif age_at_enrollment <= 21:
        factor_rows.append({
            "Factor": "Age at enrollment",
            "Value": age_at_enrollment,
            "Impact": "Lower risk",
            "Meaning": "Younger age at enrollment is associated with lower risk in this dataset"
        })

    factor_rows.sort(key=lambda x: 0 if x["Impact"] == "Higher risk" else 1)
    return factor_rows


# -------------------------------
# Main Dashboard
# -------------------------------

def main():
    st.title("🎓 Student Dropout Early Warning Dashboard")

    st.markdown(
        """
        This dashboard demonstrates a **machine learning-based early warning system**
        that helps educators identify students who may be at risk of dropping out.

        The predictions are based on academic progress indicators, financial factors,
        and demographic characteristics in the dataset.

        The tool is intended to support **early intervention and student support**,
        not to make automated decisions.
        """
    )

    df = load_data()
    model = train_model(df)
    dashboard_df = create_dashboard_dataframe(df, model)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------
    # Dashboard Overview
    # -------------------------------

    st.subheader("Dashboard Overview")

    left_col, right_col = st.columns([1.1, 1.2])

    with left_col:
        risk_counts = dashboard_df["risk_level"].value_counts().reindex(["Low", "Moderate", "High"])
        colors = ["#4CAF50", "#FFC107", "#F44336"]

        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        risk_counts.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Students by Risk Category")
        ax.set_xlabel("")
        ax.set_ylabel("Students")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with right_col:
        total_students = len(dashboard_df)
        high_risk = (dashboard_df["risk_level"] == "High").sum()
        moderate_risk = (dashboard_df["risk_level"] == "Moderate").sum()
        avg_risk = dashboard_df["dropout_probability"].mean() * 100

        metric_col1, metric_col2 = st.columns(2)
        metric_col3, metric_col4 = st.columns(2)

        metric_col1.metric("Students", total_students)
        metric_col2.metric("High Risk", high_risk)
        metric_col3.metric("Moderate Risk", moderate_risk)
        metric_col4.metric("Average Risk", f"{avg_risk:.1f}%")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------
    # Student Table
    # -------------------------------

    st.subheader("Student Risk Table")

    display_cols = [
        "student_id",
        "risk_score_pct",
        "risk_level",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Tuition fees up to date",
        "Age at enrollment",
        "recommended_action"
    ]

    table_df = dashboard_df.sort_values(
        by="dropout_probability",
        ascending=False
    )

    st.dataframe(table_df[display_cols], use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------
    # Student Detail
    # -------------------------------

    st.subheader("Student Detail View")

    student_id = st.selectbox(
        "Select student",
        dashboard_df["student_id"]
    )

    row = dashboard_df[dashboard_df["student_id"] == student_id].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Risk Score:**", f"{row['risk_score_pct']}%")
        st.write("**Risk Level:**", row["risk_level"])
        st.write("**Recommended Action:**", row["recommended_action"])

    with col2:
        st.write("**Approved Units (1st Sem):**", row["Curricular units 1st sem (approved)"])
        st.write("**Grade (1st Sem):**", round(row["Curricular units 1st sem (grade)"], 2))
        st.write("**Tuition Fees Up to Date:**", row["Tuition fees up to date"])
        st.write("**Age at Enrollment:**", int(row["Age at enrollment"]))

    st.markdown("### Explanation")
    st.write(explain_student(row))

    st.markdown("### Why this student received this risk score")
    factor_rows = get_risk_factor_rows(row)

    if factor_rows:
        factor_df = pd.DataFrame(factor_rows)
        st.dataframe(factor_df, use_container_width=True, hide_index=True)
    else:
        st.info("No clear factor pattern was identified for this student based on the current explanation rules.")

    st.caption(
        "This explanation panel is a simplified decision-support layer based on the capstone's key variables. "
        "It is intended to make the model output easier to understand, not to provide a full causal explanation."
    )

    st.markdown("### Help and Definitions")

    help_col1, help_col2 = st.columns(2)

    with help_col1:
        st.markdown("#### What the labels mean")
        st.markdown(
            """
            **Risk levels**
            - **Low**: lower predicted dropout probability
            - **Moderate**: some warning signals are present
            - **High**: stronger warning signals are present

            **Risk score**
            - Percentage estimate of dropout probability based on the model

            **Recommended action**
            - Suggested next step for advisors or educators
            """
        )

    with help_col2:
        st.markdown("#### About grades and approved units")
        st.markdown(
            """
            **Approved Units (1st Sem)**
            - Number of first-semester units the student passed successfully

            **Grade (1st Sem)**
            - Average first-semester grade in the dataset
            - The dataset comes from Portugal, where grades are typically on a **0–20 scale**
            - In general, **10 is usually the minimum passing grade**
            """
        )


if __name__ == "__main__":
    main()