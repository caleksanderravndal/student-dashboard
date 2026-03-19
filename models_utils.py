import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "dataset.csv"
RANDOM_STATE = 42

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


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.insert(0, "student_id", [f"STU-{i:04d}" for i in range(1, len(df) + 1)])
    df["dropout_risk"] = (df["Target"] == "Dropout").astype(int)
    return df


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


def create_dashboard_dataframe(df, model):
    X = df.drop(columns=["Target", "dropout_risk", "student_id"])
    probs = model.predict_proba(X)[:, 1]

    dashboard_df = df.copy()
    dashboard_df["dropout_probability"] = probs
    dashboard_df["risk_level"] = dashboard_df["dropout_probability"].apply(risk_level)
    dashboard_df["recommended_action"] = dashboard_df["dropout_probability"].apply(recommended_action)

    dashboard_df["Tuition fees up to date"] = dashboard_df["Tuition fees up to date"].map({1: "Yes", 0: "No"})
    dashboard_df["Scholarship holder"] = dashboard_df["Scholarship holder"].map({1: "Yes", 0: "No"})
    dashboard_df["Debtor"] = dashboard_df["Debtor"].map({1: "Yes", 0: "No"})
    dashboard_df["risk_score_pct"] = (dashboard_df["dropout_probability"] * 100).round(1)

    return dashboard_df


def explain_student(row):
    reasons = []

    if row["Curricular units 1st sem (approved)"] <= 2:
        reasons.append("low approved units in the first semester")

    if row["Curricular units 1st sem (grade)"] < 10:
        reasons.append("lower first-semester grades")

    if row["Tuition fees up to date"] == "No":
        reasons.append("tuition fees not up to date")

    if row["Debtor"] == "Yes":
        reasons.append("recorded debtor status")

    if row["Age at enrollment"] >= 25:
        reasons.append("older age at enrollment")

    if row["Scholarship holder"] == "No":
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
            "Factor": "Approved Units (1st Sem)",
            "Value": approved_units,
            "Impact": "Higher risk",
            "Meaning": "Very low number of approved units in the first semester"
        })
    elif approved_units >= 5:
        factor_rows.append({
            "Factor": "Approved Units (1st Sem)",
            "Value": approved_units,
            "Impact": "Lower risk",
            "Meaning": "Stronger first-semester academic progress"
        })

    first_sem_grade = row["Curricular units 1st sem (grade)"]
    if first_sem_grade < 10:
        factor_rows.append({
            "Factor": "Grade (1st Sem)",
            "Value": round(float(first_sem_grade), 2),
            "Impact": "Higher risk",
            "Meaning": "Lower first-semester average grade"
        })
    elif first_sem_grade >= 12:
        factor_rows.append({
            "Factor": "Grade (1st Sem)",
            "Value": round(float(first_sem_grade), 2),
            "Impact": "Lower risk",
            "Meaning": "Solid first-semester academic performance"
        })

    tuition_status = row["Tuition fees up to date"]
    if tuition_status == "No":
        factor_rows.append({
            "Factor": "Tuition Fees Up to Date",
            "Value": tuition_status,
            "Impact": "Higher risk",
            "Meaning": "Tuition fees are not up to date"
        })
    else:
        factor_rows.append({
            "Factor": "Tuition Fees Up to Date",
            "Value": tuition_status,
            "Impact": "Lower risk",
            "Meaning": "Tuition fees are up to date"
        })

    scholarship_status = row["Scholarship holder"]
    if scholarship_status == "No":
        factor_rows.append({
            "Factor": "Scholarship Holder",
            "Value": scholarship_status,
            "Impact": "Higher risk",
            "Meaning": "No scholarship support recorded"
        })
    else:
        factor_rows.append({
            "Factor": "Scholarship Holder",
            "Value": scholarship_status,
            "Impact": "Lower risk",
            "Meaning": "Scholarship support may reduce financial pressure"
        })

    debtor_status = row["Debtor"]
    if debtor_status == "Yes":
        factor_rows.append({
            "Factor": "Debtor",
            "Value": debtor_status,
            "Impact": "Higher risk",
            "Meaning": "Recorded debtor status may indicate financial pressure"
        })
    else:
        factor_rows.append({
            "Factor": "Debtor",
            "Value": debtor_status,
            "Impact": "Lower risk",
            "Meaning": "No debtor status recorded"
        })

    age_at_enrollment = int(row["Age at enrollment"])
    if age_at_enrollment >= 25:
        factor_rows.append({
            "Factor": "Age at Enrollment",
            "Value": age_at_enrollment,
            "Impact": "Higher risk",
            "Meaning": "Older students may face additional external responsibilities"
        })
    elif age_at_enrollment <= 21:
        factor_rows.append({
            "Factor": "Age at Enrollment",
            "Value": age_at_enrollment,
            "Impact": "Lower risk",
            "Meaning": "Younger age at enrollment is associated with lower risk in this dataset"
        })

    factor_rows.sort(key=lambda x: 0 if x["Impact"] == "Higher risk" else 1)
    return factor_rows


def _normalize(value, min_value, max_value):
    if max_value == min_value:
        return 50.0
    scaled = (value - min_value) / (max_value - min_value)
    return float(np.clip(scaled * 100, 0, 100))


def compute_group_scores(row, df):
    """
    These grouped scores are a simplified interpretation layer for visualization.
    They are not direct outputs from the machine learning model.
    Higher scores indicate stronger protective conditions or lower apparent risk
    within that group.
    """

    academic_approved = _normalize(
        row["Curricular units 1st sem (approved)"],
        df["Curricular units 1st sem (approved)"].min(),
        df["Curricular units 1st sem (approved)"].max()
    )

    academic_grade = _normalize(
        row["Curricular units 1st sem (grade)"],
        df["Curricular units 1st sem (grade)"].min(),
        df["Curricular units 1st sem (grade)"].max()
    )

    academic_score = round((academic_approved + academic_grade) / 2, 1)

    tuition_score = 100 if row["Tuition fees up to date"] == "Yes" else 0
    scholarship_score = 100 if row["Scholarship holder"] == "Yes" else 0
    debtor_score = 0 if row["Debtor"] == "Yes" else 100
    financial_score = round((tuition_score + scholarship_score + debtor_score) / 3, 1)

    age_norm = _normalize(
        row["Age at enrollment"],
        df["Age at enrollment"].min(),
        df["Age at enrollment"].max()
    )
    demographic_score = round(100 - age_norm, 1)

    return {
        "Academic": academic_score,
        "Financial": financial_score,
        "Demographic": demographic_score
    }


def make_radar_chart(group_scores):
    labels = list(group_scores.keys())
    values = list(group_scores.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"])
    ax.set_title("Student Profile Overview", pad=20)

    return fig