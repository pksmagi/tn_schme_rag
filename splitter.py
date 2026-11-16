
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read the extracted government scheme text
file_path = Path("data/schemes/tn_schemes.txt")

if not file_path.exists():
    print("ERROR: tn_schemes.txt not found.")
    raise SystemExit(1)

text = file_path.read_text(encoding="utf-8")

if not text.strip():
    print("ERROR: Text file is empty.")
    raise SystemExit(1)

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# Split text into chunks
chunks = splitter.split_text(text)

print("Total chunks created:", len(chunks))

# Save chunks for inspection
output = Path("data/chunks.txt")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, start=1):
        f.write(f"\n--- CHUNK {i} ---\n")
        f.write(chunk)
        f.write("\n")

print("Chunks saved to:", output)
