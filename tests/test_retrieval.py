from src.retriever import get_relevant_documents_with_scores


SESSION_ID = "b0e94848-fce8-4543-b5c8-bcb370130ffa"


queries = [
    "what is PS #08 about",
    "who is the SPOC for PS #08",
    "what is PS #09 about",
    "what is the capital of Japan",
    "what programming languages are mentioned",
]


for query in queries:
    print("\n" + "=" * 70)
    print("QUERY:", query)

    results = get_relevant_documents_with_scores(
        query,
        SESSION_ID
    )

    if not results:
        print("NO RESULTS")
        continue

    for i, (document, distance) in enumerate(results, 1):
        print(
            f"\n{i}. distance={distance:.4f} | "
            f"page={document.metadata.get('page', 0) + 1} | "
            f"source={document.metadata.get('source')}"
        )
        print(document.page_content[:300])