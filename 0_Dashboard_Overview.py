import streamlit as st
import matplotlib.pyplot as plt

from model_utils import load_data, train_model, create_dashboard_dataframe

st.set_page_config(
    page_title="Student Dropout Early Warning Dashboard",
    page_icon="🎓",
    layout="wide"
)


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

    st.info("Use the sidebar to navigate between the overview, risk table, and student detail pages.")

    df = load_data()
    model = train_model(df)
    dashboard_df = create_dashboard_dataframe(df, model)

    st.markdown("---")
    st.subheader("Dashboard Overview")

    left_col, right_col = st.columns([1.1, 1.2])

    with left_col:
        risk_counts = dashboard_df["risk_level"].value_counts().reindex(["Low", "Moderate", "High"])
        colors = ["#4CAF50", "#FFC107", "#F44336"]

        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        risk_counts.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Students by Risk Category")
        ax.set_xlabel("")
        ax.set_ylabel("Students")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with right_col:
        total_students = len(dashboard_df)
        high_risk = int((dashboard_df["risk_level"] == "High").sum())
        moderate_risk = int((dashboard_df["risk_level"] == "Moderate").sum())
        avg_risk = float(dashboard_df["dropout_probability"].mean() * 100)

        metric_col1, metric_col2 = st.columns(2)
        metric_col3, metric_col4 = st.columns(2)

        metric_col1.metric("Students", total_students)
        metric_col2.metric("High Risk", high_risk)
        metric_col3.metric("Moderate Risk", moderate_risk)
        metric_col4.metric("Average Risk", f"{avg_risk:.1f}%")

    st.markdown("---")
    st.subheader("What this dashboard focuses on")

    st.markdown(
        """
        - **Academic performance**: early signals such as approved units and first-semester grades  
        - **Financial factors**: tuition status, scholarship support, and debtor status  
        - **Demographic context**: age at enrollment as a contextual factor  

        The model uses a broader set of variables, but the dashboard highlights the most **interpretable and actionable**
        features for decision support.
        """
    )


if __name__ == "__main__":
    main()