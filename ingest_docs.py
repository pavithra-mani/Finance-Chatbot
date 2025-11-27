import os
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------
# CONFIG
# ---------------------------
CHROMA_PATH = "chroma_persistent_storage"
COLLECTION_NAME = "finance_collection"
DATA_FOLDER = "data"

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def clean_text(text):
    """Remove extra whitespace and URLs."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\s+", " ", text)     # normalize whitespace
    return text.strip()

# ---------------------------
# SETUP CHROMA
# ---------------------------
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=st_ef
)

# ---------------------------
# LOAD & SPLIT NEW DATA
# ---------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

chunks = []
metadatas = []
ids = []

for i, file in enumerate(os.listdir(DATA_FOLDER)):
    file_path = os.path.join(DATA_FOLDER, file)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = clean_text(f.read())
            for j, chunk in enumerate(splitter.split_text(content)):
                chunks.append(chunk)
                metadatas.append({"source": f"file://{file_path}"})
                ids.append(f"doc_{i}_{j}")

# ---------------------------
# ADD NEW CHUNKS
# ---------------------------
if chunks:
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"{len(chunks)} new chunks added to Chroma collection '{COLLECTION_NAME}'.")

# ---------------------------
# SANITY CHECK
# ---------------------------
all_docs = collection.get(include=["documents", "metadatas"])
print(f"\nNumber of documents in collection: {len(all_docs['documents'])}\n")

print("Sample documents:")
for i in range(min(5, len(all_docs["documents"]))):
    # Fix for NoneType metadata
    metadata = all_docs["metadatas"][i] if all_docs["metadatas"][i] is not None else {}
    print(f"Source: {metadata.get('source', 'No source')}")
    print(f"Content (first 200 chars): {all_docs['documents'][i][:200]}")
    print("-" * 50)

# ---------------------------
# TEST QUERY
# ---------------------------
query_text = "How to file ITR for FY 2024-25?"
results = collection.query(
    query_texts=[query_text],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\nQuery results:")
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    metadata = meta if meta is not None else {}
    print(f"Source: {metadata.get('source', 'No source')}, Distance: {dist}")
    print(f"Content snippet: {doc[:200]}")
    print("=" * 50)
