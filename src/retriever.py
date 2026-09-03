import re

from langchain_chroma import Chroma

from src.embeddings import get_embeddings


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents_collection"

TOP_K = 10
FINAL_K = 3
DISTANCE_THRESHOLD = 1.8


def get_keywords(text):
    words = re.findall(
        r"\b[a-zA-Z0-9#]+\b",
        text.lower()
    )

    stop_words = {
        "what", "is", "the", "a", "an", "about",
        "who", "for", "does", "are", "was", "were",
        "and", "or", "to", "of", "in", "on"
    }

    return {
        word
        for word in words
        if word not in stop_words
    }


def rerank_results(query, results):
    query_lower = query.lower()
    query_keywords = get_keywords(query)

    # Detect a specific problem statement such as PS #08
    ps_match = re.search(
        r"ps\s*#?\s*(\d+)",
        query_lower
    )

    requested_ps = None

    if ps_match:
        requested_ps = f"ps #{ps_match.group(1)}"

    scored_results = []

    for document, distance in results:
        text_lower = document.page_content.lower()

        # If the query specifies a PS number,
        # restrict results to that problem statement.
        if requested_ps:
            document_ps = document.metadata.get("ps_id")

            if (
                document_ps
                and document_ps.lower() != requested_ps
            ):
                continue

        document_keywords = get_keywords(
            document.page_content
        )

        keyword_matches = query_keywords.intersection(
            document_keywords
        )

        keyword_score = len(keyword_matches)

        # Keyword relevance is only a small additional signal.
        phrase_bonus = 0

        if query_lower.strip() in text_lower:
            phrase_bonus = 5

        # Vector similarity remains the primary signal.
        final_score = (
            keyword_score * 0.5
            + phrase_bonus
            - distance
        )

        scored_results.append(
            (
                document,
                distance,
                final_score
            )
        )

    # Highest final score first.
    scored_results.sort(
        key=lambda item: item[2],
        reverse=True
    )

    return [
        (document, distance)
        for document, distance, _ in scored_results[:FINAL_K]
    ]


def get_relevant_documents_with_scores(
    query,
    session_id
):
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    # First stage: vector similarity search.
    results = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K,
        filter={
            "session_id": session_id
        }
    )

    # Remove weak vector matches.
    results = [
        (document, distance)
        for document, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    # Second stage: lightweight reranking.
    results = rerank_results(
        query,
        results
    )

    return results


def get_retriever(session_id):

    def retrieve(query):
        results = get_relevant_documents_with_scores(
            query,
            session_id
        )

        return [
            document
            for document, distance in results
        ]

    return retrieve


if __name__ == "__main__":

    session_id = (
        "b0e94848-fce8-4543-b5c8-bcb370130ffa"
    )

    question = "what is the capital of Japan?"

    results = get_relevant_documents_with_scores(
        question,
        session_id
    )

    print(
        f"Retrieved {len(results)} relevant chunks\n"
    )

    for i, (document, distance) in enumerate(
        results,
        1
    ):
        print(f"--- Chunk {i} ---")

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Page: {document.metadata.get('page', 0) + 1}"
        )

        print(
            f"Source: {document.metadata.get('source')}"
        )

        print(
            document.page_content[:500]
        )

        print()