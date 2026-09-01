from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import load_pdf, split_documents


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "resume_collection"


def create_vectorstore(pdf_path):
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove old collection so uploaded PDFs don't get mixed together
    temp_store = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    temp_store.delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

    return vectorstore


if __name__ == "__main__":
    pdf_path = Path("data/documents/DevanshiPatel_Resume__.pdf")
    create_vectorstore(pdf_path)