import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)

#local embedding function
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="finance_collection",
    embedding_function=st_ef
)


model = genai.GenerativeModel("gemini-2.5-flash")

#hybrid rag+fallback with source urls 
def rag_query(user_query, n_results=3, similarity_threshold=0.65):
    """
    Try to answer using finance_collection first. 
    If no relevant docs, fall back to Gemini.
    """
    results = collection.query(
        query_texts=[user_query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    #document relevance
    relevant_docs = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist <= similarity_threshold:  
            relevant_docs.append((doc, meta))

    if relevant_docs:
        context = "\n\n".join([doc for doc, _ in relevant_docs])
        prompt = f"""
You are a financial advisor chatbot.
Use the following documents to answer user queries accurately.

Context from finance docs:
{context}

User question:
{user_query}

Provide a clear, practical answer.
"""
        response = model.generate_content(prompt)
        return response.text
    else:
        #Fallback: ask Gemini directly
        prompt = f"""
You are a financial advisor chatbot. Answer the question clearly.

User question:
{user_query}
"""
        response = model.generate_content(prompt)
        return response.text


while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer = rag_query(user_input)
    print("Gemini:", answer)
