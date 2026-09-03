import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.retriever import get_relevant_documents_with_scores


@st.cache_resource
def get_chatbot():
    load_dotenv()

    llm = ChatOpenAI(
        model="minimax/minimax-m3:free",
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    return llm


def rewrite_query(question, chat_history, llm):
    if not chat_history:
        return question

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-6:]
    )

    prompt = f"""
Rewrite the user's latest question into a standalone search query.

Use the conversation history to resolve references such as:
"it", "this", "that", "they", "the company", etc.

If the question is already standalone, return it unchanged.

Do not answer the question.
Return only the rewritten search query.

Conversation history:
{history}

Latest question:
{question}

Standalone search query:
"""

    try:
        response = llm.invoke(prompt)
    except Exception:
        return question

    return response.content.strip()


def ask_question(question, session_id, chat_history=None):
    llm = get_chatbot()

    # Rewrite follow-up question into a standalone query
    search_query = rewrite_query(
        question,
        chat_history,
        llm
    )

    # Retrieve documents with raw distance scores
    results = get_relevant_documents_with_scores(
        search_query,
        session_id
    )

    documents = [
        document
        for document, distance in results
    ]

    # If no sufficiently similar chunks were found
    if not documents:
        return (
            "I couldn't find this information in the provided document.",
            []
        )

    # Add page numbers to each retrieved chunk
    context_parts = []

    for document in documents:
        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"[Page {page}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Prepare conversation history
    history = ""

    if chat_history:
        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in chat_history[-6:]
        )

    prompt = f"""
You are a helpful assistant answering questions based only on the provided context.

Conversation history:
{history}

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- If the answer cannot be found in the context, say:
"I couldn't find this information in the provided document."
- After your answer, write a line in this exact format:
Sources: Page X, Page Y
- Include only pages that directly support your answer.
- Do not include pages that were retrieved but not used.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate response
    try:
        response = llm.invoke(prompt)
    except Exception:
        return (
            "Sorry, I couldn't generate an answer right now. "
            "Please try again.",
            []
        )

    response_text = response.content.strip()

    # Extract source pages from the LLM response
    pages = []

    source_match = re.search(
        r"Sources:\s*(.*)",
        response_text,
        re.IGNORECASE
    )

    if source_match:
        source_text = source_match.group(1)

        pages = [
            int(page)
            for page in re.findall(
                r"Page\s+(\d+)",
                source_text,
                re.IGNORECASE
            )
        ]

        # Remove the Sources line from the displayed answer
        response_text = response_text[
            :source_match.start()
        ].strip()

    return response_text, sorted(set(pages))


if __name__ == "__main__":
    question = "What programming languages does Devanshi know?"

    session_id = "test-session"

    answer, pages = ask_question(
        question,
        session_id
    )

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    print(pages)