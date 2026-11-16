
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP 1: Read website text
file_path = Path("data/schemes/tn_schemes.txt")

if not file_path.exists():
    print("ERROR: tn_schemes.txt not found!")
    raise SystemExit(1)

text = file_path.read_text(encoding="utf-8")

if not text.strip():
    print("ERROR: File is empty!")
    raise SystemExit(1)

print("Step 1: Website text loaded successfully")

# STEP 2: Convert text into LangChain document
document = Document(
    page_content=text,
    metadata={
        "source": "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    }
)

# STEP 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents([document])

print("Step 2: Total chunks:", len(chunks))

# STEP 4: Create embedding model
print("Step 3: Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 5: Create FAISS vector database
print("Step 4: Creating FAISS database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# STEP 6: Save FAISS database
vectorstore.save_local("vectorstore")

print("Step 5: FAISS database saved successfully!")
print("Location: vectorstore/")
