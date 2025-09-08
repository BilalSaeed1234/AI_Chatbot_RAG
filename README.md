🤖 AI Chatbot with RAG

This project demonstrates how to build a chat application using Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG).

With this chatbot, you can upload a PDF file and then ask questions — the bot will retrieve information directly from your PDF and provide accurate, contextual answers.

🚀 Features

📄 Upload PDF documents

🔍 Retrieve relevant context from PDF using embeddings

🤖 Ask questions and get answers based on your PDF’s content

🧠 RAG pipeline ensures more meaningful and insightful responses

⚙️ How It Works

Your PDF is converted into text and split into chunks.

Each chunk is embedded and stored in a vector database.

When you ask a question, the bot retrieves the most relevant chunks.

The answer is generated using the LLM with the retrieved context.

🛠️ Tech Stack

Python 3.10+

LangChain

FAISS (Vector Database)

PDFPlumber (for PDF parsing)

LLM APIs (e.g., OpenAI, Gemini, etc.)

Built by [Bilal Saeed](https://github.com/BilalSaeed1234) as part of an AI projects portfolio.