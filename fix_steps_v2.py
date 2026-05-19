with open('report/chapter1/chapter1.tex', 'r') as f:
    text = f.read()

for i in range(1, 9):
    text = text.replace(f"\\subsection{{Bước {i}", f"\\subsubsection*{{Bước {i}")

with open('report/chapter1/chapter1.tex', 'w') as f:
    f.write(text)
