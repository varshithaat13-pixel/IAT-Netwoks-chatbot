# -*- coding: utf-8 -*-
import pdfplumber
import json

all_pages = []

with pdfplumber.open("IAT Networks.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        all_pages.append({
            "page": i + 1,
            "text": text if text else ""
        })

with open("extracted_text.json", "w", encoding="utf-8") as f:
    json.dump(all_pages, f, ensure_ascii=False, indent=2)

print("Done. Pages extracted:", len(all_pages))
