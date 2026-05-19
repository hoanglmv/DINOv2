import re

with open('report/chapter1/chapter1.tex', 'r') as f:
    text = f.read()

def replace_textbf(match):
    return f"\\subsection{{{match.group(1)}}}"

text = re.sub(r'^\\textbf\{(.*?)\}\s*$', replace_textbf, text, flags=re.MULTILINE)

def replace_textit_numbered(match):
    inner = match.group(1)
    inner = re.sub(r'^\d+\)\s*', '', inner)
    return f"\\subsubsection{{{inner}}}"

text = re.sub(r'^\\textit\{(\d+\).*?)\}\s*$', replace_textit_numbered, text, flags=re.MULTILINE)

def replace_textit_lettered(match):
    inner = match.group(1)
    inner = re.sub(r'^[a-z]\)\s*', '', inner)
    return f"\\subsubsection{{{inner}}}"

text = re.sub(r'^\\textit\{([a-z]\).*?)\}\\\\\s*$', replace_textit_lettered, text, flags=re.MULTILINE)

text = text.replace('\\subsection{Kết luận từ Result}', '\\subsection{Kết luận từ kết quả}')
text = text.replace('\\subsection{Tài liệu tham khảo}\\\\', '\\section*{Tài liệu tham khảo}')

with open('report/chapter1/chapter1.tex', 'w') as f:
    f.write(text)

print("Chapter 1 section hierarchy updated.")
