from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    pdf_path = Path("data/documents/DevanshiPatel_Resume__.pdf")

    documents = load_pdf(pdf_path)

    print("Number of pages:", len(documents))

    chunks = split_documents(documents)

    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content)