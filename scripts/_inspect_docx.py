from docx import Document
import sys

files = [
    'Battery_Project_Report.docx',
    'Dashboard_Report.docx',
    'QA_Document.docx',
    'Streamlit_Output_Explained.docx',
]
keywords = ['NASA','dataset','feature','ensemble','coverage','Gemma','Llama','CALCE','PCoE','model','calce']

for fname in files:
    try:
        doc = Document(fname)
        hits = []
        for i, para in enumerate(doc.paragraphs):
            t = para.text.strip()
            if any(k.lower() in t.lower() for k in keywords) and t:
                hits.append(f'  [{i}] {t[:130]}')
        print(f'=== {fname} ({len(hits)} hits) ===')
        for h in hits[:20]:
            print(h.encode('ascii','replace').decode())
        print()
    except Exception as e:
        print(f'{fname}: ERROR {e}')
