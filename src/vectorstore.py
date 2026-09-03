from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import get_embeddings
from src.loader import load_pdf, split_documents


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents_collection"


def create_vectorstore(pdf_path, session_id):
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    source_name = Path(pdf_path).name

    # Remove only this PDF from this user's session
    try:
        vectorstore.delete(
            where={
                "$and": [
                    {"source": source_name},
                    {"session_id": session_id}
                ]
            }
        )
    except Exception as e:
        print(
            f"Warning: Could not remove existing chunks "
            f"for '{source_name}': {e}"
        )

    # Add session information to every chunk
    for chunk in chunks:
        chunk.metadata["source"] = source_name
        chunk.metadata["session_id"] = session_id

    vectorstore.add_documents(chunks)

    print(
        f"Stored {len(chunks)} chunks from '{source_name}' "
        f"for session '{session_id}'."
    )

    return vectorstore


if __name__ == "__main__":
    pdf_path = Path("data/documents/DevanshiPatel_Resume__.pdf")

    create_vectorstore(
        pdf_path,
        session_id="test-session"
    )