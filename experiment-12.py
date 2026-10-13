%pip install -q langchain-community langchain-text-splitters faiss-cpu sentence-transformers
---------------------------------------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
---------------------------------------------------------------------------------------------------------
text = """
Climate change refers to long-term changes in temperatures and weather patterns.
Burning fossil fuels increases greenhouse gases and contributes to global warming.
Climate change causes rising temperatures, melting glaciers, and rising sea levels.

Extreme weather events such as floods, droughts, heatwaves, and powerful storms
are becoming more concerning. These events can damage homes, crops, and roads.
Farmers may experience lower crop yields because of changing rainfall patterns.

Rising sea levels threaten coastal cities and communities. Melting polar ice caps
and the expansion of warmer ocean water contribute to this problem.
Climate change also affects wildlife, forests, and marine ecosystems.

People can reduce climate change by using renewable energy, saving electricity,
planting trees, protecting forests, and using public transportation.
Governments and communities can reduce emissions through sustainable practices.
"""

documents = [Document(page_content=text)]
print("Sample text loaded successfully.")
-------------------------------------------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:\n{chunk.page_content}")
  ----------------------------------------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded successfully.")
------------------------------------------------------------------------------------------------------------
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created successfully.")
---------------------------------------------------------------------------------------------------
query = "What are the effects of climate change?"

results = vectorstore.similarity_search_with_relevance_scores(
    query,
    k=2
)

print("Query:", query)
print("\nTop 2 chunks:\n")

for doc, score in results:
    print(f"Score: {score:.4f}")
    print("Chunk:", doc.page_content)
    print("-" * 60)
  --------------------------------------------------------------------------------------
with open("rag_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Number of chunks: {len(chunks)}\n")
    f.write(f"Query: {query}\n\nTop 2 chunks:\n\n")

    for doc, score in results:
        f.write(f"Score: {score:.4f}\n")
        f.write(f"Chunk: {doc.page_content}\n\n")

print("Output saved to rag_output.txt")
