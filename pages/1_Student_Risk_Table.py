import streamlit as st

from model_utils import load_data, train_model, create_dashboard_dataframe


def main():
    st.title("📋 Student Risk Table")

    df = load_data()
    model = train_model(df)
    dashboard_df = create_dashboard_dataframe(df, model)

    st.markdown(
        """
        This page provides a sortable list of students with their predicted dropout risk,
        selected indicators, and suggested next action.
        """
    )

    st.markdown("---")

    filter_option = st.selectbox(
        "Filter by risk level",
        ["All", "High", "Moderate", "Low"]
    )

    table_df = dashboard_df.copy()

    if filter_option != "All":
        table_df = table_df[table_df["risk_level"] == filter_option]

    table_df = table_df.sort_values(by="dropout_probability", ascending=False)

    display_df = table_df[
        [
            "student_id",
            "risk_score_pct",
            "risk_level",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Tuition fees up to date",
            "Debtor",
            "Age at enrollment",
            "recommended_action"
        ]
    ].rename(
        columns={
            "student_id": "Student ID",
            "risk_score_pct": "Risk Score (%)",
            "risk_level": "Risk Level",
            "Curricular units 1st sem (approved)": "Approved Units (1st Sem)",
            "Curricular units 1st sem (grade)": "Grade (1st Sem)",
            "Tuition fees up to date": "Tuition Current",
            "Age at enrollment": "Age at Enrollment",
            "recommended_action": "Recommended Action"
        }
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()