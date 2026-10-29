#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường và huấn luyện mmFormer trên Google Colab CLI
# Benchmark Protocol: myops380_train304_test76_last_v1 (No-Val, 300 Epochs)
# ==============================================================================
set -e

echo "=== [1/6] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime > Change runtime type > GPU trên Colab!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/6] Mount Google Drive ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Không thể mount Drive tự động (bỏ qua nếu lưu SSD cục bộ):', e)
" || true

DRIVE_DIR="/content/drive/MyDrive/MyoPS_Results/mmFormer"
mkdir -p "$DRIVE_DIR" || true

echo "=== [3/6] Cài đặt Dependencies cho mmFormer ==="
python3 -m pip install --upgrade pip -q
python3 -m pip install -q einops SimpleITK medpy nibabel gdown tensorboard

echo "=== [4/6] Tải Dataset MyoPS-380 ==="
LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
if [ ! -d "$LOCAL_DATASET" ]; then
    echo "Đang tải Dataset MyoPS-380 từ Google Drive..."
    gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O /content/MyoPS380_dataset.zip
    mkdir -p /content/MyoPS380_dataset
    command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip tmux)
    unzip -q -o /content/MyoPS380_dataset.zip -d /content/MyoPS380_dataset/
fi

# Tự động định vị thư mục Processed_data
FOUND_BSSFP=$(find /content/MyoPS380_dataset -type d -name "bSSFP" 2>/dev/null | head -n 1)
if [ -n "$FOUND_BSSFP" ]; then
    LOCAL_DATASET=$(dirname "$FOUND_BSSFP")
fi
echo "✅ Đường dẫn Dataset chuẩn: $LOCAL_DATASET"

echo "=== [5/6] Xác minh mã băm SHA-256 danh sách ca ==="
cd /content/mmFormer
python3 -c "
import hashlib
for fname, expected in [
    ('lists/lists_MyoPS380/train.txt', '16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919'),
    ('lists/lists_MyoPS380/test_vol.txt', '3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc')
]:
    with open(fname, 'rb') as fp:
        actual = hashlib.sha256(fp.read()).hexdigest()
    assert actual == expected, f'Checksum mismatch in {fname}: {actual}'
print('✅ Tất cả mã băm split SHA-256 hoàn toàn hợp lệ!')
"

echo "=== [6/6] Cấp quyền thực thi các script ==="
chmod +x run_s1.sh run_s2.sh evaluate.py train.py

echo ""
echo "=============================================================================="
echo "🎉 HOÀN TẤT CHUẨN BỊ MÔI TRƯỜNG CHO mmFormer!"
echo "👉 Để chạy 3 Seed (1234, 1098, 1024) trên máy s1: bash run_s1.sh"
echo "👉 Để chạy 2 Seed (2026, 6868) trên máy s2: bash run_s2.sh"
echo "=============================================================================="
