"""缩小截图以便在终端显示"""
import os
from PIL import Image

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
for f in sorted(os.listdir(img_dir)):
    if not f.endswith(".png"):
        continue
    path = os.path.join(img_dir, f)
    img = Image.open(path)
    w, h = img.size
    print(f"{f}: {w}x{h}")
    # 缩小到宽度800
    new_w = 800
    ratio = new_w / w
    new_h = int(h * ratio)
    small = img.resize((new_w, new_h), Image.LANCZOS)
    out_path = os.path.join(img_dir, f.replace(".png", "_sm.png"))
    small.save(out_path, optimize=True)
    print(f"  -> {out_path}")
print("Done")
