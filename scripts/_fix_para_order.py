"""Fix reversed paragraph order: literature table and limitations section."""
from docx import Document
from docx.oxml.ns import qn
import copy

doc = Document("Battery_Project_Report.docx")
body = doc.element.body
all_paras = doc.paragraphs

# Identify the index of para [214] (start of reversed section) 
# and [223] (16. Key Equations Reference = last para we care about)
# We need to find them by content

def find_idx(paras, fragment):
    for i, p in enumerate(paras):
        if fragment in p.text:
            return i
    return None

lit_heading_idx  = find_idx(all_paras, "14. Comparison with Prior Work")
key_eq_idx       = find_idx(all_paras, "16. Key Equations Reference")

print(f"Literature heading at para index: {lit_heading_idx}")
print(f"Key Equations at para index: {key_eq_idx}")

if lit_heading_idx is None or key_eq_idx is None:
    print("ERROR: could not locate section boundaries")
    exit(1)

# Collect the paragraphs from lit_heading to key_eq (inclusive)
section_paras = all_paras[lit_heading_idx : key_eq_idx + 1]
print(f"Section spans {len(section_paras)} paragraphs")
for i, p in enumerate(section_paras):
    safe = p.text.strip().encode("ascii","replace").decode()
    print(f"  [{i}] {safe[:80]}")

# Separate into groups by content
lit_items  = []  # literature table paragraphs
lim_items  = []  # limitations items (numbered 1-7)
lim_heading = None
key_eq_para = None

for p in section_paras:
    t = p.text.strip()
    if "14. Comparison with Prior Work" in t:
        lit_heading_p = p
    elif t.startswith("IMPORTANT: Direct RMSE"):
        lit_caveat_p = p
    elif t.startswith("Paper | Dataset"):
        lit_header_p = p
    elif t.startswith("Lin et al."):
        lit_lin_p = p
    elif t.startswith("Zhao et al."):
        lit_zhao_p = p
    elif t.startswith("Catelani et al."):
        lit_catelani_p = p
    elif t.startswith("THIS WORK"):
        lit_thiswork_p = p
    elif t.startswith("Key differentiators:"):
        lit_keydiff_p = p
    elif "15. Limitations and Honest Assessment" in t:
        lim_heading = p
    elif "16. Key Equations Reference" in t:
        key_eq_para = p
    elif t and t[0].isdigit() and ". " in t[:4]:
        lim_items.append(p)

# Sort lim_items by leading number
def lim_num(p):
    try:
        return int(p.text.strip().split(".")[0])
    except:
        return 99

lim_items.sort(key=lim_num)
print(f"\nLimitation items found: {len(lim_items)}")
for p in lim_items:
    print(f"  {p.text[:60].encode('ascii','replace').decode()}")

# Remove all paragraphs from the section from the body
for p in section_paras:
    p._p.getparent().remove(p._p)

# Now append in correct order: lit table, then limitations, then key equations
correct_order = [
    lit_heading_p,
    lit_caveat_p,
    lit_header_p,
    lit_lin_p,
    lit_zhao_p,
    lit_catelani_p,
    lit_thiswork_p,
    lit_keydiff_p,
    lim_heading,
] + lim_items + [key_eq_para]

for p in correct_order:
    body.append(p._p)

doc.save("Battery_Project_Report.docx")
print(f"\nRe-inserted {len(correct_order)} paragraphs in correct order.")
