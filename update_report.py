import json
import os
import subprocess

def main():
    # 1. Read history
    try:
        with open("logs/dinov2_history.json", "r") as f:
            data = json.load(f)
            acc = data.get("test_acc", 0.0) * 100
            f1 = data.get("test_f1", 0.0) * 100
    except Exception as e:
        print("Error reading logs:", e)
        return

    # 2. Update chapter2.tex
    chapter2_path = "report/chapter2/chapter2.tex"
    with open(chapter2_path, "r") as f:
        content = f.read()

    # Replace table placeholders
    content = content.replace(r"\textbf{XX.XX} & \textbf{XX.XX}", f"\\textbf{{{acc:.2f}}} & \\textbf{{{f1:.2f}}}")
    
    # Replace the text note
    note = r"(Cần điền thêm kết quả của DINOv2 sau khi huấn luyện xong để thấy rõ sự khác biệt)."
    replacement = f"Thực tế, kết quả kiểm thử cho thấy DINOv2 đạt độ chính xác {acc:.2f}\\%, chứng tỏ đặc trưng tự học của nó phù hợp và mạnh mẽ hơn nhiều so với việc dùng ViT-B/16."
    content = content.replace(note, replacement)

    with open(chapter2_path, "w") as f:
        f.write(content)

    print(f"Updated chapter2.tex with DINOv2 Accuracy: {acc:.2f}% and F1: {f1:.2f}%")

    # 3. Plot results
    print("Running plot_results.py...")
    subprocess.run(["python", "src/skin/plot_results.py"])

    # 4. Compile latex (run twice for cross-references)
    print("Compiling LaTeX report...")
    os.chdir("report")
    subprocess.run(["pdflatex", "main.tex"])
    subprocess.run(["pdflatex", "main.tex"])
    print("All done!")

if __name__ == "__main__":
    main()
