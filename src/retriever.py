from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_PATH = "chroma_db"


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name="resume_collection",
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


if __name__ == "__main__":
    retriever = get_retriever()

    question = "What programming languages does Devanshi know?"

    documents = retriever.invoke(question)

    print(f"Retrieved {len(documents)} chunks\n")

    for i, doc in enumerate(documents):
        print(f"--- Retrieved Chunk {i + 1} ---")
        print(doc.page_content)
        print()