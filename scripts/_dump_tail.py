"""Fix reversed paragraphs: collect all out-of-order paragraphs, remove, re-insert correctly."""
from docx import Document

doc = Document("Battery_Project_Report.docx")
body = doc.element.body

P = doc.paragraphs
print(f"Total: {len(P)} paragraphs")
# Print tail to understand structure
for i, p in enumerate(P):
    t = p.text.strip()
    if i >= 204:
        safe = t.encode("ascii","replace").decode()
        print(f"[{i}] {safe[:90]}")
