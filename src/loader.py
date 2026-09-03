from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise ValueError("The PDF contains no pages.")

        valid_documents = []
        current_ps = None

        for document in documents:
            text = document.page_content.strip()

            if not text:
                continue

            # Detect PS number when it appears on a page
            ps_match = re.search(
                r"\bPS\s*#\s*(\d+)\b",
                text,
                re.IGNORECASE
            )

            if ps_match:
                current_ps = f"PS #{ps_match.group(1)}"

            document.metadata["source"] = Path(file_path).name
            document.metadata["page"] = document.metadata.get(
                "page",
                len(valid_documents)
            )

            # Assign the current problem statement to this page
            if current_ps:
                document.metadata["ps_id"] = current_ps

            valid_documents.append(document)

        if not valid_documents:
            raise ValueError(
                "No extractable text was found in the PDF. "
                "The PDF may be scanned, empty, or image-based."
            )

        total_text = " ".join(
            document.page_content.strip()
            for document in valid_documents
        )

        if len(total_text) < 50:
            raise ValueError(
                "Very little text could be extracted from the PDF. "
                "The PDF may be scanned or have poor text extraction."
            )

        return valid_documents

    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}") from e


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def split_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")

    return chunks