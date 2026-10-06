import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./rag/chroma_db"
)

collection = client.get_or_create_collection(
    name="data_knowledge"
)


knowledge = [
    {
        "id": "missing_values",
        "text": """
        Missing values occur when a dataset contains empty or unavailable
        information. Numerical missing values can sometimes be replaced
        using mean or median. Categorical missing values can sometimes
        use the mode. The correct method depends on the dataset.
        """
    },
    {
        "id": "duplicates",
        "text": """
        Duplicate rows are repeated records in a dataset. Duplicates should
        be investigated before analysis because they can affect statistics
        and machine learning models. If they represent accidental repetition,
        they can usually be removed.
        """
    },
    {
        "id": "outliers",
        "text": """
        Outliers are unusually high or low observations compared with the
        rest of the data. They can be detected using methods such as the
        IQR method or standard deviation. Outliers should be investigated
        before removing them because they may represent real events.
        """
    },
    {
        "id": "data_cleaning",
        "text": """
        Data cleaning involves handling missing values, duplicates,
        incorrect data types, inconsistent values and unusual observations.
        Clean data improves the reliability of analysis and machine learning.
        """
    },
    {
        "id": "machine_learning",
        "text": """
        Before using data for machine learning, check missing values,
        duplicates, incorrect data types, categorical variables, outliers
        and inconsistent values. Data preprocessing should be performed
        according to the requirements of the model.
        """
    }
]


def build_knowledge_base():

    for item in knowledge:

        embedding = model.encode(item["text"]).tolist()

        collection.upsert(
            ids=[item["id"]],
            documents=[item["text"]],
            embeddings=[embedding]
        )

    return collection

if __name__ == "__main__":
    build_knowledge_base()
    print("Knowledge base built successfully.")