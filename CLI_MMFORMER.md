# HƯỚNG DẪN CHẠY TRAIN & ĐÁNH GIÁ MMFORMER TRÊN COLAB (5 SEEDS)
> **Tuân thủ nghiêm ngặt Quy chuẩn Benchmark Không Validation ([QUY_TAC_BENCHMARK_NO_VAL.md](file:///D:/NCKH/QUY_TAC_BENCHMARK_NO_VAL.md)):**
> - **Protocol:** `myops380_train304_test76_last_v1`
> - **304 train patients** (1.972 lát cắt 2D), **76 test volumes** (không validation)
> - Đánh giá duy nhất trên checkpoint `last.pth` ở **Epoch 300**
> - **Golden Threads Allocation:** `export OMP_NUM_THREADS=2`, `--num-workers 2`, `--cpu-threads 2`
> - **Chuẩn kênh & nhãn Canonical:** `num_classes=4` (0: BG, 1: Normal Myo, 2: Edema, 3: Scar)
> - **Hàm Loss:** 100% loss gốc của mmFormer ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fuse}} + \mathcal{L}_{\text{sep}} + \mathcal{L}_{\text{prm}}$)

---

## 🟢 MÁY 1 (s1 - GPU A100): Chạy 3 Seed (1234, 1098, 1024)

### 👉 Bước 1: Khởi tạo môi trường tmux
*(Copy 3 dòng này dán vào Terminal máy s1, nhấn Enter để màn hình hiện thanh ngang màu xanh ở đáy):*
```bash
tmux kill-server 2>/dev/null
tmux new -s train
cd /content
```

### 👉 Bước 2: Chạy toàn bộ Pipeline mmFormer (Chuẩn No-Val) + Tự động treo phiên
*(Sau khi đã ở trong tmux, copy nguyên khối này dán vào và nhấn Enter):*
```bash
rm -rf /content/mmFormer
git clone https://github.com/YaoZhang93/mmFormer.git /content/mmFormer
cd /content/mmFormer
pip install -q einops SimpleITK medpy nibabel gdown tensorboard

# 1. Tải Dataset MyoPS-380
LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
if [ ! -d "$LOCAL_DATASET" ]; then
    echo "Đang tải Dataset MyoPS-380..."
    gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O /content/MyoPS380_dataset.zip
    mkdir -p /content/MyoPS380_dataset
    unzip -q -o /content/MyoPS380_dataset.zip -d /content/MyoPS380_dataset/
fi

# 2. Thiết lập giới hạn luồng CPU tránh nghẽn I/O & CPU thrashing
export OMP_NUM_THREADS=2

echo "🚀 Bắt đầu Train SONG SONG 3 Seed mmFormer: 1234, 1098, 1024 (Chuẩn No-Val 300 Epochs)..."
python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output outputs/runs --seed 1234 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 > train_1234.log 2>&1 &
python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output outputs/runs --seed 1098 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 > train_1098.log 2>&1 &
python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output outputs/runs --seed 1024 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 > train_1024.log 2>&1 &
wait

# Kiểm tra log nếu phát hiện checkpoint chưa được tạo
for s in 1234 1098 1024; do
    if [ ! -f "outputs/runs/mmFormer_MyoPS380_Seed_${s}/checkpoints/last.pth" ]; then
        echo "❌ Lỗi: Không tìm thấy checkpoint của Seed ${s}! 25 dòng cuối của train_${s}.log:"
        tail -n 25 "train_${s}.log" 2>/dev/null || true
    fi
done

echo "✅ Train xong 3 Seed! Bắt đầu đánh giá Test 76 ca từ checkpoint last.pth (Epoch 300)..."
python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir outputs/runs/mmFormer_MyoPS380_Seed_1234 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8
python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir outputs/runs/mmFormer_MyoPS380_Seed_1098 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8
python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir outputs/runs/mmFormer_MyoPS380_Seed_1024 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8

echo "📦 Đóng gói kết quả..."
(cd /content/mmFormer/outputs/runs/mmFormer_MyoPS380_Seed_1234 && zip -r -q /content/mmFormer_Seed_1234_Results.zip .)
(cd /content/mmFormer/outputs/runs/mmFormer_MyoPS380_Seed_1098 && zip -r -q /content/mmFormer_Seed_1098_Results.zip .)
(cd /content/mmFormer/outputs/runs/mmFormer_MyoPS380_Seed_1024 && zip -r -q /content/mmFormer_Seed_1024_Results.zip .)

echo "🎉 HOÀN TẤT SESSION 1!"
echo "🕒 Tự động kích hoạt lệnh treo phiên ngầm để chống Colab tự ngắt..."
nohup bash -c 'while true; do sleep 60; done' >/dev/null 2>&1 &
echo "✅ ĐÃ TREO PHIÊN THÀNH CÔNG! BẠN CÓ THỂ ĐÓNG VS CODE VÀ TẢI ZIP BẤT CỨ LÚC NÀO!"
```

---

## 🟢 MÁY 2 (s2 - GPU A100): Chạy 2 Seed (2026, 6868)

### 👉 Bước 1: Khởi tạo môi trường tmux
*(Copy 3 dòng này dán vào Terminal máy s2, nhấn Enter để màn hình hiện thanh ngang màu xanh ở đáy):*
```bash
tmux kill-server 2>/dev/null
tmux new -s train
cd /content
```

### 👉 Bước 2: Chạy toàn bộ Pipeline mmFormer (Chuẩn No-Val) + Tự động treo phiên
*(Sau khi đã ở trong tmux, copy nguyên khối này dán vào và nhấn Enter):*
```bash
rm -rf /content/mmFormer
git clone https://github.com/YaoZhang93/mmFormer.git /content/mmFormer
cd /content/mmFormer
pip install -q einops SimpleITK medpy nibabel gdown tensorboard

# 1. Tải Dataset MyoPS-380
LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
if [ ! -d "$LOCAL_DATASET" ]; then
    echo "Đang tải Dataset MyoPS-380..."
    gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O /content/MyoPS380_dataset.zip
    mkdir -p /content/MyoPS380_dataset
    unzip -q -o /content/MyoPS380_dataset.zip -d /content/MyoPS380_dataset/
fi

# 2. Thiết lập giới hạn luồng CPU tránh nghẽn I/O & CPU thrashing
export OMP_NUM_THREADS=2

echo "🚀 Bắt đầu Train SONG SONG 2 Seed mmFormer: 2026, 6868 (Chuẩn No-Val 300 Epochs)..."
python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output outputs/runs --seed 2026 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 > train_2026.log 2>&1 &
python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output outputs/runs --seed 6868 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 > train_6868.log 2>&1 &
wait

# Kiểm tra log nếu phát hiện checkpoint chưa được tạo
for s in 2026 6868; do
    if [ ! -f "outputs/runs/mmFormer_MyoPS380_Seed_${s}/checkpoints/last.pth" ]; then
        echo "❌ Lỗi: Không tìm thấy checkpoint của Seed ${s}! 25 dòng cuối của train_${s}.log:"
        tail -n 25 "train_${s}.log" 2>/dev/null || true
    fi
done

echo "✅ Train xong 2 Seed! Bắt đầu đánh giá Test 76 ca từ checkpoint last.pth (Epoch 300)..."
python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir outputs/runs/mmFormer_MyoPS380_Seed_2026 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8
python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir outputs/runs/mmFormer_MyoPS380_Seed_6868 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8

echo "📦 Đóng gói kết quả..."
(cd /content/mmFormer/outputs/runs/mmFormer_MyoPS380_Seed_2026 && zip -r -q /content/mmFormer_Seed_2026_Results.zip .)
(cd /content/mmFormer/outputs/runs/mmFormer_MyoPS380_Seed_6868 && zip -r -q /content/mmFormer_Seed_6868_Results.zip .)

echo "🎉 HOÀN TẤT SESSION 2!"
echo "🕒 Tự động kích hoạt lệnh treo phiên ngầm để chống Colab tự ngắt..."
nohup bash -c 'while true; do sleep 60; done' >/dev/null 2>&1 &
echo "✅ ĐÃ TREO PHIÊN THÀNH CÔNG! BẠN CÓ THỂ ĐÓNG VS CODE VÀ TẢI ZIP BẤT CỨ LÚC NÀO!"
```

---

## 📥 LỆNH TẢI ZIP VỀ MÁY LOCAL (DÙNG POWERSHELL HOẶC GIT BASH TRÊN MÁY TÍNH)
Chạy các lệnh SCP sau trên máy cá nhân để kéo kết quả về `D:/NCKH/ChanhThucTrainTest/mmFormer/`:
```bash
# 3 Seed từ Máy 1 (s1):
scp s1:/content/mmFormer_Seed_1234_Results.zip D:/NCKH/ChanhThucTrainTest/mmFormer/
scp s1:/content/mmFormer_Seed_1098_Results.zip D:/NCKH/ChanhThucTrainTest/mmFormer/
scp s1:/content/mmFormer_Seed_1024_Results.zip D:/NCKH/ChanhThucTrainTest/mmFormer/

# 2 Seed từ Máy 2 (s2):
scp s2:/content/mmFormer_Seed_2026_Results.zip D:/NCKH/ChanhThucTrainTest/mmFormer/
scp s2:/content/mmFormer_Seed_6868_Results.zip D:/NCKH/ChanhThucTrainTest/mmFormer/
```
