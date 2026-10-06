from rag.knowledge_base import model, collection


def retrieve_knowledge(question, number_of_results=3):

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results.get("documents", [[]])[0]

    return documents