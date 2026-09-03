import streamlit as st
from pathlib import Path
import uuid

from src.chatbot import ask_question
from src.vectorstore import create_vectorstore


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

# Initialize session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


st.title("🤖 RAG Chatbot")
st.write("Upload PDF documents and ask questions about them.")


uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    documents_path = Path("data/documents")
    documents_path.mkdir(parents=True, exist_ok=True)

    pdf_paths = []

    for uploaded_file in uploaded_files:

        pdf_path = documents_path / uploaded_file.name

        try:
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_paths.append(pdf_path)

        except OSError as e:
            st.error(
                f"Could not save '{uploaded_file.name}'. "
                f"Please try again."
            )

    if st.button("Process PDFs"):

        if not pdf_paths:
            st.error("No valid PDF files were available for processing.")

        else:
            try:
                with st.spinner("Processing PDFs..."):

                    for pdf_path in pdf_paths:
                        create_vectorstore(
                            pdf_path,
                            st.session_state.session_id
                        )

                st.session_state.pdf_processed = True
                st.session_state.messages = []

                st.success(
                    f"{len(pdf_paths)} PDF(s) processed successfully!"
                )

            except ValueError as e:
                st.session_state.pdf_processed = False
                st.error(f"Could not process the PDF: {e}")

            except Exception:
                st.session_state.pdf_processed = False
                st.error(
                    "An unexpected error occurred while processing "
                    "the PDF(s). Please try again."
                )


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant" and message.get("pages"):
            st.caption(
                "📄 Sources: " +
                ", ".join(
                    f"Page {page}"
                    for page in message["pages"]
                )
            )


# Chat input
if st.session_state.pdf_processed:

    question = st.chat_input("Ask a question...")

else:

    question = None
    st.info(
        "Please upload and process PDF documents before asking questions."
    )


# Handle new question
if question:

    st.chat_message("user").write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                answer, pages = ask_question(
                    question,
                    st.session_state.session_id,
                    st.session_state.messages
                )

            except Exception:
                answer = (
                    "Sorry, I couldn't process your question right now. "
                    "Please try again."
                )
                pages = []

        st.write(answer)

        if pages:
            st.caption(
                "📄 Sources: " +
                ", ".join(
                    f"Page {page}"
                    for page in pages
                )
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pages": pages
    })