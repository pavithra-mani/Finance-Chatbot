import chromadb

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_collection("finance_collection")

print("Collections in store:", chroma_client.list_collections())
print("Number of documents in collection:", len(collection.get(include=["documents"])["documents"]))

# Peek at first 5 documents
all_docs = collection.get(include=["documents", "metadatas"])  # remove "ids"
for i in range(min(5, len(all_docs["documents"]))):
    print(f"Source: {all_docs['metadatas'][i].get('source', 'No source')}")
    print(f"Content (first 200 chars): {all_docs['documents'][i][:200]}")
    print("-" * 50)

# Optional: run a manual query to test embeddings
query = "How to file ITR for FY 2024-25?"
results = collection.query(
    query_texts=[query],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Source: {meta.get('source', 'No source')}, Distance: {dist}")
    print(f"Content snippet: {doc[:200]}")
    print("=" * 50)
