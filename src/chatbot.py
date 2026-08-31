from langchain_ollama import ChatOllama

from src.retriever import get_retriever


def get_chatbot():
    retriever = get_retriever()

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return retriever, llm


def ask_question(question, chat_history=None):
    retriever, llm = get_chatbot()

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Combine retrieved chunks into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prepare conversation history
    history = ""

    if chat_history:
        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in chat_history[-6:]
        )

    # Create prompt
    prompt = f"""
You are a helpful assistant answering questions based only on the provided context.

Conversation history:
{history}

If the answer cannot be found in the context, say:
"I couldn't find this information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate response
    response = llm.invoke(prompt)

    # Get source pages
    pages = sorted(
        set(
            document.metadata.get("page", 0) + 1
            for document in documents
        )
    )

    return response.content, pages


if __name__ == "__main__":
    question = "What programming languages does Devanshi know?"

    answer, pages = ask_question(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    print(pages)