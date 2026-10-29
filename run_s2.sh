#!/bin/bash
# ==============================================================================
# Script tự động chạy 2 Seed (2026, 6868) song song trong tmux trên Session s2
# Benchmark Protocol: myops380_train304_test76_last_v1 (No-Val, 300 Epochs)
# Model: mmFormer (Standalone Baseline)
# ==============================================================================
set -e

DRIVE_RUNS="/content/drive/MyDrive/MyoPS_Results/mmFormer/runs"
LOCAL_RUNS="/content/mmFormer/outputs/runs"
OUTPUT_DIR="$DRIVE_RUNS"

if [ ! -d "/content/drive/MyDrive" ]; then
    OUTPUT_DIR="$LOCAL_RUNS"
fi
mkdir -p "$OUTPUT_DIR"

SESSION_NAME="train_s2"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "🚀 Đang kích hoạt 2 Seed (2026, 6868) song song trên máy s2 (tmux: $SESSION_NAME)..."

CMD_2026="cd /content/mmFormer && export OMP_NUM_THREADS=2 && python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output $OUTPUT_DIR --seed 2026 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 && python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir $OUTPUT_DIR/mmFormer_MyoPS380_Seed_2026 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8 && (cd $OUTPUT_DIR/mmFormer_MyoPS380_Seed_2026 && zip -r -q /content/mmFormer_Seed_2026_Results.zip .)"

CMD_6868="cd /content/mmFormer && export OMP_NUM_THREADS=2 && python3 train.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output $OUTPUT_DIR --seed 6868 --max_epochs 300 --batch_size 8 --effective_batch_size 16 --num_workers 2 --cpu_threads 2 --base_lr 0.001 --weight_decay 1e-4 --img_size 128 && python3 evaluate.py --root_path /content/MyoPS380_dataset/Processed_data --list_dir lists/lists_MyoPS380 --output_dir $OUTPUT_DIR/mmFormer_MyoPS380_Seed_6868 --img_size 128 --num_classes 4 --cpu_threads 2 --batch_size 8 && (cd $OUTPUT_DIR/mmFormer_MyoPS380_Seed_6868 && zip -r -q /content/mmFormer_Seed_6868_Results.zip .)"

tmux new-session -d -s "$SESSION_NAME" -n "seed_2026" "$CMD_2026"
tmux new-window -t "$SESSION_NAME" -n "seed_6868" "$CMD_6868"

echo ""
echo "=============================================================================="
echo "🎉 ĐÃ KÍCH HOẠT THÀNH CÔNG 2 SEED (2026, 6868) CHẠY NỀN TRONG TMUX!"
echo "👉 Để vào xem màn hình trực tiếp các seed: tmux attach -t $SESSION_NAME"
echo "👉 Chuyển đổi giữa các seed: Nhấn Ctrl+b rồi ấn 0 (seed 2026), 1 (seed 6868)"
echo "👉 Thoát ra ngoài an toàn: Nhấn Ctrl+b rồi ấn d"
echo "=============================================================================="
