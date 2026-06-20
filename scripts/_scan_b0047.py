"""Scan all .docx files for B0047 mentions."""
from docx import Document
from pathlib import Path

results = {}
for fpath in Path(".").glob("*.docx"):
    try:
        doc = Document(str(fpath))
        hits = []
        for i, p in enumerate(doc.paragraphs):
            if "B0047" in p.text:
                hits.append(f"  para[{i}]: {p.text[:120]}")
        results[fpath.name] = hits
    except Exception as e:
        results[fpath.name] = [f"ERROR: {e}"]

out = Path("paperprep/b0047_scan.txt")
lines = []
for fname, hits in results.items():
    if hits:
        lines.append(f"\n{fname} -- B0047 FOUND:")
        lines.extend(hits)
    else:
        lines.append(f"{fname} -- clean")
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nWritten to {out}")
