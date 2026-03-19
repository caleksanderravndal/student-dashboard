import pandas as pd
import streamlit as st

from model_utils import (
    load_data,
    train_model,
    create_dashboard_dataframe,
    explain_student,
    get_risk_factor_rows,
    compute_group_scores,
    make_radar_chart
)


def main():
    st.title("🔍 Student Detail View")

    df = load_data()
    model = train_model(df)
    dashboard_df = create_dashboard_dataframe(df, model)

    student_id = st.selectbox("Select student", dashboard_df["student_id"])

    row = dashboard_df[dashboard_df["student_id"] == student_id].iloc[0]

    st.markdown("---")

    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.subheader("Student Summary")
        st.write("**Risk Score:**", f"{row['risk_score_pct']}%")
        st.write("**Risk Level:**", row["risk_level"])
        st.write("**Recommended Action:**", row["recommended_action"])
        st.write("**Approved Units (1st Sem):**", row["Curricular units 1st sem (approved)"])
        st.write("**Grade (1st Sem):**", round(row["Curricular units 1st sem (grade)"], 2))
        st.write("**Tuition Fees Up to Date:**", row["Tuition fees up to date"])
        st.write("**Debtor:**", row["Debtor"])
        st.write("**Age at Enrollment:**", int(row["Age at enrollment"]))

    with top_right:
        st.subheader("Grouped Student Profile")
        group_scores = compute_group_scores(row, dashboard_df)
        fig = make_radar_chart(group_scores)
        st.pyplot(fig)

        st.caption(
            "This chart is a simplified interpretation layer. It groups selected variables into broader categories "
            "to help explain the student's profile. It is not a direct output from the machine learning model."
        )

    st.markdown("---")
    st.subheader("Explanation")
    st.write(explain_student(row))

    st.subheader("Why this student received this risk score")
    factor_rows = get_risk_factor_rows(row)

    if factor_rows:
        factor_df = pd.DataFrame(factor_rows)
        st.dataframe(factor_df, use_container_width=True, hide_index=True)
    else:
        st.info("No clear factor pattern was identified for this student based on the current explanation rules.")

    st.caption(
        "The dashboard focuses on interpretable and actionable variables. Sensitive or non-actionable variables "
        "are not shown in the explanation layer."
    )

    st.markdown("---")
    st.subheader("Help and Definitions")

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
        st.markdown("#### About the grouped profile")
        st.markdown(
            """
            **Academic**
            - Based on first-semester approved units and grade

            **Financial**
            - Based on tuition status, scholarship support, and debtor status

            **Demographic**
            - Based on age at enrollment as a contextual indicator

            These grouped scores are designed to support interpretation and quick understanding.
            They are not direct model outputs.
            """
        )

    st.markdown("#### About grades and approved units")
    st.markdown(
        """
        - **Approved Units (1st Sem)**: number of first-semester units the student passed successfully  
        - **Grade (1st Sem)**: average first-semester grade in the original dataset  
        - The dataset comes from Portugal, where grades are typically on a **0–20 scale**  
        - In general, **10 is usually the minimum passing grade**
        """
    )


if __name__ == "__main__":
    main()