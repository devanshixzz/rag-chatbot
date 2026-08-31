import streamlit as st
from pathlib import Path

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


st.title("🤖 RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    documents_path = Path("data/documents")
    documents_path.mkdir(parents=True, exist_ok=True)

    pdf_path = documents_path / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):
            create_vectorstore(pdf_path)

        st.session_state.pdf_processed = True
        st.session_state.messages = []

        st.success("PDF processed successfully!")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and "pages" in message:
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
    st.info("Please upload and process a PDF before asking questions.")


# Handle new question
if question:

    # Display and store user message
    st.chat_message("user").write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            answer, pages = ask_question(
                question,
                st.session_state.messages
                )

        st.write(answer)
        st.caption(
            "📄 Sources: " +
            ", ".join(f"Page {page}" for page in pages)
            )

    # Store assistant message
    st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "pages": pages
})