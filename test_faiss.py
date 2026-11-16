
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading FAISS database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

question = input("Enter your question: ")

results = vectorstore.similarity_search(
    question,
    k=3
)

print("\nRelevant Government Scheme Information:\n")

for i, doc in enumerate(results, start=1):
    print(f"\nRESULT {i}")
    print(doc.page_content[:1000])
    print("Source:", doc.metadata.get("source"))
