import pandas as pd
import requests

from rag.retriever import retrieve_knowledge


def create_investigation_report(info, column_info, df):
    report = []

    # Missing values
    if info["missing_values"] > 0:
        report.append(
            f"⚠️ The dataset contains {info['missing_values']} missing values."
        )
    else:
        report.append("✅ No missing values were detected.")

    # Duplicate rows
    if info["duplicate_rows"] > 0:
        report.append(
            f"⚠️ The dataset contains {info['duplicate_rows']} duplicate rows."
        )
    else:
        report.append("✅ No duplicate rows were detected.")

    # Dataset size
    report.append(
        f"📊 The dataset contains {info['rows']} rows "
        f"and {info['columns']} columns."
    )

    # Numerical column investigation
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        if df[column].nunique() > 1:

            mean = df[column].mean()
            std = df[column].std()

            if std > 0:
                unusual_values = df[
                    abs(df[column] - mean) > 2 * std
                ]

                if len(unusual_values) > 0:
                    report.append(
                        f"🔎 Column '{column}' contains "
                        f"{len(unusual_values)} unusual value(s)."
                    )

    return report


def generate_recommendations(info, df):
    recommendations = []

    # Missing values
    if info["missing_values"] > 0:
        recommendations.append(
            "💡 Recommendation: Check the missing values and either fill them "
            "with suitable values or remove the affected records."
        )

    # Duplicate rows
    if info["duplicate_rows"] > 0:
        recommendations.append(
            "💡 Recommendation: Review the duplicate rows and remove them "
            "if they represent repeated records."
        )

    # Unusual numerical values
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        if df[column].nunique() > 1:

            mean = df[column].mean()
            std = df[column].std()

            if std > 0:

                unusual_values = df[
                    abs(df[column] - mean) > 2 * std
                ]

                if len(unusual_values) > 0:
                    recommendations.append(
                        f"💡 Recommendation: Investigate unusual values "
                        f"in the '{column}' column before using the data."
                    )

    # No problems found
    if len(recommendations) == 0:
        recommendations.append(
            "✅ No major data quality problems were detected."
        )

    return recommendations


def ask_ai(question, data_summary):

    knowledge = retrieve_knowledge(question)

    retrieved_knowledge = "\n\n".join(knowledge)

    prompt = f"""
You are Data Detective AI.

You are investigating a dataset.

Dataset information:
{data_summary}

Relevant data-quality knowledge:
{retrieved_knowledge}

User question:
{question}

Use the relevant knowledge above to answer the user's question.

Give a simple and clear answer.

Explain:
1. What the problem means.
2. Why it may matter.
3. What the user should do.

Do not invent information that is not supported by the dataset
or the retrieved knowledge.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]

    return "Sorry, I could not connect to the AI model."