import re

with open('report/chapter1/chapter1.tex', 'r') as f:
    text = f.read()

text = re.sub(r'\\subsection\{(Bước \d+.*?)\}', r'\\subsubsection*{\1}', text)

with open('report/chapter1/chapter1.tex', 'w') as f:
    f.write(text)
