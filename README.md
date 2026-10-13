# Experiment 12: RAG Data Indexing Using FAISS

## Aim
To build a basic data indexing pipeline for Retrieval-Augmented Generation (RAG) by splitting text into chunks, generating embeddings, storing them in a FAISS vector store, and retrieving relevant chunks using similarity search.

## Problem Statement
Create a Python script that processes sample climate-change text, divides it into chunks, generates embeddings using OpenAI, stores the chunks in FAISS, and retrieves the top two relevant chunks for a query.

## Technologies Used
- Python
- LangChain
- OpenAI Embeddings
- FAISS Vector Store

## Algorithm
1. Install the required packages.
2. Set the OpenAI API key.
3. Define sample climate-change text.
4. Split the text using `RecursiveCharacterTextSplitter`.
5. Generate embeddings using `OpenAIEmbeddings`.
6. Create a FAISS vector store using `FAISS.from_documents()`.
7. Search using `similarity_search_with_relevance_scores()`.
8. Display the top two chunks and their relevance scores.

## Installation

```bash
pip install -r requirements.txt
```

## API Key Setup

**Windows Command Prompt:**

```cmd
set OPENAI_API_KEY=your_api_key_here
```

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Replace the placeholder with your API key. Never upload your key to GitHub.

## Execution

```bash
python src/main.py
```

## Expected Output

```text
Number of chunks: [calculated at runtime]

Query: What are the effects of climate change?

Top 2 chunks:

Score: [calculated at runtime]
Chunk: [most relevant text chunk]

--------------------------------------------------

Score: [calculated at runtime]
Chunk: [second most relevant text chunk]
```

The actual chunk count, retrieved text, and relevance scores depend on the text splitter and embedding model.

## Result
The RAG indexing pipeline is implemented using LangChain and FAISS. Text chunks are embedded and stored in a vector store, allowing the two most relevant chunks to be retrieved for a given query.
