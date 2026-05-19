import os
from PIL import Image
import shutil

os.chdir("report/figures")
for f in os.listdir("."):
    if f.startswith("ảnh "):
        new_name = f.replace("ảnh ", "fig").replace(" ", "_")
        if new_name.endswith(".webp"):
            try:
                # Convert webp to png
                im = Image.open(f).convert("RGB")
                new_name = new_name.replace(".webp", ".png")
                im.save(new_name, "PNG")
                print(f"Converted and renamed {f} to {new_name}")
            except Exception as e:
                print(f"Error converting {f}: {e}")
        else:
            shutil.copy(f, new_name)
            print(f"Copied {f} to {new_name}")

for i in range(1, 10):
    f = f"t{i}.png"
    if os.path.exists(f):
        new_name = f"table{i}.png"
        shutil.copy(f, new_name)
        print(f"Copied {f} to {new_name}")
