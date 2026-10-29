#!/bin/bash
# ==============================================================================
# Script tự động chạy 3 Seed (1234, 1098, 1024) song song trong tmux trên Session s1
# Benchmark Protocol: myops380_train304_test76_last_v1 (No-Val, 300 Epochs)
# Model: mmFormer (Standalone Baseline)
# ==============================================================================
set -e

DRIVE_RUNS="/content/drive/MyDrive/MyoPS_Results/mmFormer/runs"
LOCAL_RUNS="/content/mmFormer/outputs/runs"
OUTPUT_DIR="$DRIVE_RUNS"

# Nếu chưa mount Drive thì lưu trên SSD cục bộ
if [ ! -d "/content/drive/MyDrive" ]; then
    OUTPUT_DIR="$LOCAL_RUNS"
fi
mkdir -p "$OUTPUT_DIR"

SESSION_NAME="train_s1"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "🚀 Đang kích hoạt 3 Seed (1234, 1098, 1024) song song trên máy s1 (tmux: $SESSION_NAME)..."

CMD_1234="cd /content/mmFormer && export OMP_NUM_THREADS=2 && python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output $OUTPUT_DIR --seed 1234 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 && python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir $OUTPUT_DIR/mmFormer_MyoPS380_Seed_1234 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8 && (cd $OUTPUT_DIR/mmFormer_MyoPS380_Seed_1234 && zip -r -q /content/mmFormer_Seed_1234_Results.zip .)"

CMD_1098="cd /content/mmFormer && export OMP_NUM_THREADS=2 && python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output $OUTPUT_DIR --seed 1098 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 && python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir $OUTPUT_DIR/mmFormer_MyoPS380_Seed_1098 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8 && (cd $OUTPUT_DIR/mmFormer_MyoPS380_Seed_1098 && zip -r -q /content/mmFormer_Seed_1098_Results.zip .)"

CMD_1024="cd /content/mmFormer && export OMP_NUM_THREADS=2 && python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output $OUTPUT_DIR --seed 1024 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 && python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir $OUTPUT_DIR/mmFormer_MyoPS380_Seed_1024 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8 && (cd $OUTPUT_DIR/mmFormer_MyoPS380_Seed_1024 && zip -r -q /content/mmFormer_Seed_1024_Results.zip .)"

tmux new-session -d -s "$SESSION_NAME" -n "seed_1234" "$CMD_1234"
tmux new-window -t "$SESSION_NAME" -n "seed_1098" "$CMD_1098"
tmux new-window -t "$SESSION_NAME" -n "seed_1024" "$CMD_1024"

echo ""
echo "=============================================================================="
echo "🎉 ĐÃ KÍCH HOẠT THÀNH CÔNG 3 SEED (1234, 1098, 1024) CHẠY NỀN TRONG TMUX!"
echo "👉 Để vào xem màn hình trực tiếp các seed: tmux attach -t $SESSION_NAME"
echo "👉 Chuyển đổi giữa các seed: Nhấn Ctrl+b rồi ấn 0 (seed 1234), 1 (seed 1098), 2 (seed 1024)"
echo "👉 Thoát ra ngoài an toàn: Nhấn Ctrl+b rồi ấn d"
echo "=============================================================================="
