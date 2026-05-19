#!/usr/bin/env bash
# Tải Oxford-IIIT Pet (dataset nhẹ cho zero-shot segmentation).
#   - images.tar.gz       ~ 755 MB (7349 ảnh JPEG)
#   - annotations.tar.gz  ~  19 MB (trimap PNG + split list)
#
# Cách dùng:
#   bash scripts/download_oxford_pet.sh                # tải về data/oxford_pet
#   bash scripts/download_oxford_pet.sh /custom/path   # tải về thư mục tuỳ chọn

set -euo pipefail

DATA_ROOT="${1:-data/oxford_pet}"
BASE_URL="https://thor.robots.ox.ac.uk/pets"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "==> Đang tải Oxford-IIIT Pet vào: $(pwd)"

download() {
    local url="$1"
    local file="$2"
    if [ -f "$file" ]; then
        echo "    [skip] $file đã tồn tại"
        return
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -c --show-progress "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$file" "$url"
    else
        echo "Cần wget hoặc curl để tải dataset." >&2
        exit 1
    fi
}

download "${BASE_URL}/images.tar.gz" images.tar.gz
download "${BASE_URL}/annotations.tar.gz" annotations.tar.gz

if [ ! -d images ]; then
    echo "==> Giải nén images.tar.gz"
    tar -xzf images.tar.gz
fi

if [ ! -d annotations ]; then
    echo "==> Giải nén annotations.tar.gz"
    tar -xzf annotations.tar.gz
fi

NUM_IMG=$(find images -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
NUM_TRI=$(find annotations/trimaps -maxdepth 1 -name '*.png' 2>/dev/null | wc -l | tr -d ' ')

echo
echo "==> Hoàn tất. Layout:"
echo "    $(pwd)/images/*.jpg          ($NUM_IMG ảnh)"
echo "    $(pwd)/annotations/trimaps/  ($NUM_TRI mask)"
echo "    $(pwd)/annotations/trainval.txt, test.txt"
echo
echo "Test loader bằng:"
echo "    cd src && python -c 'from zero_shot_seg.data import OxfordPetSegDataset; \\"
echo "        ds = OxfordPetSegDataset(\"../$DATA_ROOT\", max_samples=3); \\"
echo "        print(len(ds), ds[0][0].shape, ds[0][1].shape, ds[0][2])'"
