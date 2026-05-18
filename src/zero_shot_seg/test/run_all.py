"""Chạy tất cả test trong package này.

Cách dùng:
    cd src
    python -m zero_shot_seg.test.run_all
"""

import importlib

MODULES = [
    "zero_shot_seg.test.test_feature_extractor",
    "zero_shot_seg.test.test_segmenter",
    "zero_shot_seg.test.test_visualize",
]

if __name__ == "__main__":
    for name in MODULES:
        print(f"\n=== {name} ===")
        mod = importlib.import_module(name)
        for attr in dir(mod):
            if attr.startswith("test_"):
                getattr(mod, attr)()
    print("\n✓ All tests passed.")
