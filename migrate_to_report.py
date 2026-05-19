import re

# Update main.tex
with open('report/main.tex', 'r') as f:
    main_tex = f.read()
main_tex = main_tex.replace('\\documentclass[a4paper,12pt]{article}', '\\documentclass[a4paper,12pt]{report}')
with open('report/main.tex', 'w') as f:
    f.write(main_tex)

# Update chapter1.tex
with open('report/chapter1/chapter1.tex', 'r') as f:
    c1 = f.read()
c1 = c1.replace('\\section{Chương 1: Problem và Motivation của bài báo DINOv2}', '\\chapter{PROBLEM VÀ MOTIVATION CỦA BÀI BÁO DINOV2}')
c1 = re.sub(r'\\subsection\{1\.\d+\s+(.*?)\}', r'\\section{\1}', c1)
with open('report/chapter1/chapter1.tex', 'w') as f:
    f.write(c1)

# Update chapter2.tex
with open('report/chapter2/chapter2.tex', 'r') as f:
    c2 = f.read()
c2 = c2.replace('\\section{Chương 2: Thực nghiệm đánh giá DINOv2}', '\\chapter{THỰC NGHIỆM ĐÁNH GIÁ DINOV2}')
c2 = c2.replace('\\subsection{Bài toán Phân loại ảnh (Image Classification)}', '\\section{Bài toán Phân loại ảnh (Image Classification)}')
c2 = c2.replace('\\subsubsection{Mô tả tập dữ liệu SkinCancerMNIST}', '\\subsection{Mô tả tập dữ liệu SkinCancerMNIST}')
c2 = c2.replace('\\subsubsection{Thiết lập thực nghiệm và Cấu hình so sánh (ViT vs DINOv2)}', '\\subsection{Thiết lập thực nghiệm và Cấu hình so sánh (ViT vs DINOv2)}')
c2 = c2.replace('\\subsubsection{Kết quả và Đánh giá (Results \\& Analysis)}', '\\subsection{Kết quả và Đánh giá (Results \\& Analysis)}')
with open('report/chapter2/chapter2.tex', 'w') as f:
    f.write(c2)

# Update chapter3.tex
with open('report/chapter3/chapter3.tex', 'r') as f:
    c3 = f.read()
c3 = c3.replace('\\section{Chương 3: Đánh giá và Tư duy Tích cực (Critical Thinking)}', '\\chapter{ĐÁNH GIÁ VÀ TƯ DUY TÍCH CỰC (CRITICAL THINKING)}')
c3 = c3.replace('\\subsection{', '\\section{')
with open('report/chapter3/chapter3.tex', 'w') as f:
    f.write(c3)

print("Migration to report class complete.")
