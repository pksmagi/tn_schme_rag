import requests
from bs4 import BeautifulSoup
from pathlib import Path

url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

folder = Path("data/schemes")
folder.mkdir(parents=True, exist_ok=True)

try:
    print("Connecting to Tamil Nadu Government website...")

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError("No text found on website")

    output = folder / "tn_schemes.txt"
    output.write_text(text, encoding="utf-8")

    print("Website text extracted successfully!")
    print("Saved to:", output)
    print(text[:1000])

except Exception as e:
    print("Error:", e)