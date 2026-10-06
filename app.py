import streamlit as st


# ==============================
# 🎨 Data Detective AI Theme
# ==============================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #F5F7FB;
    }

    /* Main title */
    h1 {
        color: #172554;
        font-weight: 800;
    }

    /* Section headings */
    h2, h3 {
        color: #312E81;
        font-weight: 700;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E0E7FF;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: #475569;
    }

    div[data-testid="stMetricValue"] {
        color: #312E81;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #3730A3;
        color: white;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #0F766E;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background-color: #115E59;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #172554;
    }

    section[data-testid="stSidebar"] * {
    color: white !important;
}

    /* AI answer box */
    div[data-testid="stTextInput"] {
        background-color: white;
        border-radius: 10px;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

</style>
""", unsafe_allow_html=True)

from backend.data_processor import (
    load_data,
    get_basic_info,
    get_column_info
)

from ai.analyzer import (
    create_investigation_report,
    generate_recommendations,
    ask_ai
)


st.set_page_config(
    page_title="Data Detective",
    page_icon="🕵️"
)


# Sidebar
with st.sidebar:
    st.header("🕵️ Data Detective AI")
    st.write("AI-powered dataset investigation.")

    st.divider()

    st.subheader("How it works")
    st.write("1. Upload a CSV")
    st.write("2. Investigate the data")
    st.write("3. Find data-quality issues")
    st.write("4. Get recommendations")
    st.write("5. Ask the AI")

    st.divider()

    st.caption("Powered by Llama 3.2 + RAG")


# Main title
st.title("🕵️ Data Detective")
st.write("Upload a dataset and let the AI investigate it.")


# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:

    # Load dataset safely
    try:
        df = load_data(uploaded_file)

    except Exception:
        st.error("❌ Unable to read this CSV file.")
        st.stop()

    # Check empty dataset
    if df.empty:
        st.warning("⚠️ The uploaded dataset is empty.")
        st.stop()


    # Dataset Preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)


    # Basic information
    info = get_basic_info(df)


    # Investigation Results
    st.subheader("🔍 Investigation Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", info["rows"])
    col2.metric("Columns", info["columns"])
    col3.metric("Missing Values", info["missing_values"])
    col4.metric("Duplicate Rows", info["duplicate_rows"])


    # Column Investigation
    st.subheader("🧪 Column Investigation")

    column_info = get_column_info(df)

    st.dataframe(column_info)


    # Statistical Summary
    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose()
    )


    # Visualization
    st.subheader("📊 Data Visualization")

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numerical_columns) > 0:

        selected_column = st.selectbox(
            "Select a numerical column:",
            numerical_columns
        )

        st.line_chart(
            df[selected_column]
        )

    else:
        st.info(
            "No numerical columns available for visualization."
        )


    # AI Investigation
    st.subheader("🕵️ AI Investigation")

    report = create_investigation_report(
        info,
        column_info,
        df
    )

    for finding in report:
        st.write(finding)


    # Recommendations
    st.subheader("💡 Recommendations")

    recommendations = generate_recommendations(
        info,
        df
    )

    for recommendation in recommendations:
        st.write(recommendation)


    # Download Report
    st.subheader("📥 Download Investigation Report")

    report_text = "DATA DETECTIVE AI - INVESTIGATION REPORT\n\n"

    report_text += f"Rows: {info['rows']}\n"
    report_text += f"Columns: {info['columns']}\n"
    report_text += f"Missing Values: {info['missing_values']}\n"
    report_text += f"Duplicate Rows: {info['duplicate_rows']}\n\n"

    report_text += "AI INVESTIGATION\n"
    report_text += "-----------------\n"

    for finding in report:
        report_text += finding + "\n"

    report_text += "\nRECOMMENDATIONS\n"
    report_text += "---------------\n"

    for recommendation in recommendations:
        report_text += recommendation + "\n"

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="data_detective_report.txt",
        mime="text/plain"
    )


    # Ask AI
    st.subheader("🤖 Ask Data Detective AI")

    question = st.text_input(
        "Ask a question about your dataset:"
    )

    if question:

        data_summary = df.describe(
            include="all"
        ).to_string()

        try:

            answer = ask_ai(
                question,
                data_summary
            )

        except Exception:

            answer = ""

        if not answer:

            st.warning(
                "⚠️ I couldn't generate an answer. "
                "Make sure Ollama and Llama 3.2 are running."
            )

        else:

            st.write("### 🧠 AI Answer")
            st.write(answer)