#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường và chuẩn bị dữ liệu cho mmFormer trên Colab CLI
# Repository: https://github.com/nguyentranhuynhchi/mmFormer.git
# Protocol: myops380_train304_test76_last_v1 (No-Val, 300 epochs)
# ==============================================================================

set -e

echo "=== [1/5] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

# Tự động liên kết thư viện driver NVIDIA nếu có trên Colab (/usr/lib64-nvidia)
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
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime trên Colab!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/5] Kiểm tra Mount Google Drive ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Drive chưa mount tự động (bỏ qua nếu đã mount qua colab drivemount):', e)
else:
    print('✅ Google Drive đã được mount sẵn tại /content/drive/MyDrive')
" || true

echo "=== [3/5] Cài đặt Dependencies cho mmFormer ==="
python3 -m pip install --upgrade pip -q
python3 -m pip install medpy SimpleITK scipy h5py gdown -q

echo "=== [4/5] Tải Dataset từ Thư mục Google Drive về SSD Cục bộ ==="
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

# Kiểm tra nếu dataset đã có sẵn trên Google Drive để copy trực tiếp, tránh rate limit của gdown
for DRIVE_CANDIDATE in \
    "/content/drive/MyDrive/dataset/MyoPS380/Processed_data" \
    "/content/drive/MyDrive/MyoPS380_dataset/Processed_data" \
    "/content/drive/MyDrive/Processed_data" \
    "/content/drive/MyDrive/NCKH_V2/MyoPS380_dataset/Processed_data" \
    "/content/drive/MyDrive/NCKH/MyoPS380_dataset/Processed_data"; do
    if [ -d "$DRIVE_CANDIDATE" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
        echo "✅ Phát hiện dataset trên Google Drive: $DRIVE_CANDIDATE. Đang sao chép vào SSD cục bộ..."
        mkdir -p "$TARGET_DATASET_DIR/Processed_data"
        cp -r "$DRIVE_CANDIDATE/." "$TARGET_DATASET_DIR/Processed_data/"
        break
    fi
done

# Kiểm tra nếu có tệp zip dataset trên Google Drive để copy trực tiếp
for DRIVE_ZIP in \
    "/content/drive/MyDrive/dataset/MyoPS380/MyoPS380_dataset.zip" \
    "/content/drive/MyDrive/MyoPS380_dataset.zip" \
    "/content/drive/MyDrive/dataset/MyoPS380.zip" \
    "/content/drive/MyDrive/MyoPS380.zip"; do
    if [ -s "$DRIVE_ZIP" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
        echo "✅ Phát hiện tệp zip dataset trên Google Drive: $DRIVE_ZIP. Đang sao chép..."
        cp "$DRIVE_ZIP" "$DOWNLOAD_DIR/"
        break
    fi
done

# Tải toàn bộ thư mục Google Drive nếu chưa tải
if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
    echo "Đang tải toàn bộ thư mục dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
    gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || {
        echo "⚠️ Tải folder qua gdown gặp cảnh báo, tiếp tục kiểm tra các tệp đã tải..."
    }
fi

# Tự động giải nén bất kỳ tệp zip nào có trong thư mục tải về
for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
    echo "Đang giải nén $zipfile vào $TARGET_DATASET_DIR ..."
    command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
    unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/"
done

# Nếu trong thư mục tải về đã là thư mục dữ liệu giải nén sẵn
if [ -d "$DOWNLOAD_DIR/Processed_data" ]; then
    cp -r "$DOWNLOAD_DIR/Processed_data" "$TARGET_DATASET_DIR/"
fi

# Cơ chế tự động định vị thư mục dữ liệu chuẩn (thư mục chứa bSSFP, LGE, T2w)
FOUND_BSSFP=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" -type d -name "bSSFP" 2>/dev/null | head -n 1)
if [ -n "$FOUND_BSSFP" ]; then
    LOCAL_DATASET=$(dirname "$FOUND_BSSFP")
    echo "✅ Đã xác định thư mục dataset chuẩn tại SSD: $LOCAL_DATASET"
    if [ "$LOCAL_DATASET" != "/content/MyoPS380_dataset/Processed_data" ] && [ ! -d "/content/MyoPS380_dataset/Processed_data" ]; then
        mkdir -p "/content/MyoPS380_dataset"
        ln -sfn "$LOCAL_DATASET" "/content/MyoPS380_dataset/Processed_data"
        echo "✅ Đã liên kết $LOCAL_DATASET -> /content/MyoPS380_dataset/Processed_data"
    fi
else
    LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
    echo "⚠️ Chưa phát hiện thư mục bSSFP, đặt mặc định: $LOCAL_DATASET"
fi

echo "=== [5/5] Kiểm tra PyTorch & Môi trường Hoàn tất ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "================================================================================"
echo "✅ THIẾT LẬP MÔI TRƯỜNG MMFORMER HOÀN TẤT 100%!"
echo "Dataset chuẩn sẵn sàng tại: $LOCAL_DATASET"
echo "Splits chuẩn sẵn sàng tại: /content/mmFormer/splits/myops380_no_val"
echo "Huấn luyện từ đầu với Kaiming Normal (Conv) & Xavier Uniform (Linear) — không cần pretrain."
echo "================================================================================"
