import json

with open("logs/dinov2_history.json", "r") as f:
    d = json.load(f)
d_acc = d["test_acc"] * 100
d_f1 = d["test_f1"] * 100

with open("logs/vit_history.json", "r") as f:
    v = json.load(f)
v_acc = v["test_acc"] * 100
v_f1 = v["test_f1"] * 100

with open("report/chapter2/chapter2.tex", "r") as f:
    content = f.read()

import re
# Replace the table row for DINOv2
content = re.sub(r"\\textbf\{DINOv2\} & Linear Probing & \\textbf\{.*?\} & \\textbf\{.*?\} \\\\",
                 f"\\\\textbf{{DINOv2}} & Linear Probing & \\\\textbf{{{d_acc:.2f}}} & \\\\textbf{{{d_f1:.2f}}} \\\\\\\\", content)

# Replace the paragraph text
content = re.sub(r"DINOv2 đạt .*?\\% Accuracy và .*?\\% F1-Score\.", 
                 f"DINOv2 đạt {d_acc:.2f}\\% Accuracy và {d_f1:.2f}\\% F1-Score.", content)

with open("report/chapter2/chapter2.tex", "w") as f:
    f.write(content)

print(f"Updated chapter2.tex with DINOv2 Acc: {d_acc:.2f}%, F1: {d_f1:.2f}%")
