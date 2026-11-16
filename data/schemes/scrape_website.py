
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Tamil Nadu Government scheme URL
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

# Create output folder
folder = Path("data/schemes")
folder.mkdir(parents=True, exist_ok=True)

try:
    # Download webpage
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted content
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError("No readable text found.")

    # Save extracted text
    file_path = folder / "tn_schemes.txt"

    file_path.write_text(text, encoding="utf-8")

    print("Website text extracted successfully!")
    print("Saved to:", file_path)
    print("Total characters:", len(text))
    print("\nFirst 1000 characters:\n")
    print(text[:1000])

except Exception as e:
    print("Error:", e)
