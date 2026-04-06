CHECKPOINT_PATH="/home/ubuntu/h100x8/fairchem_omat24_08-04-2025/models/mptrj/equiformer_v3/trained_checkpoints/1110-1_grad-ft_N@7_L@4_merge-ln_pt-reg-dens-strict-max-r@0.75-ft-no-reg_lr@0-5e-5_rand-e/checkpoint_epochs@10.0.pt"
OUTPUT_DIR="/home/ubuntu/h100x8/fairchem_omat24_08-04-2025/experimental/tasks/matbench_discovery/discovery_results/equiformer_v3/1110-1_grad-ft_N@7_L@4_merge-ln_pt-reg-dens-strict-max-r@0.75-ft-no-reg_lr@0-5e-5_rand-e/epochs@10/all"
DATA_PATH="/lambda/nfs/h100x8/datasets/omat24/matbench_discovery/discovery/WBM_IS2RE.aselmdb"

for i in $(seq 0 31); do
    device=$((i % 8))
    CUDA_VISIBLE_DEVICES=$device python experimental/tasks/matbench_discovery/test_discovery.py \
        --checkpoint-path $CHECKPOINT_PATH \
        --output-path $OUTPUT_DIR \
        --data-path $DATA_PATH \
        --num-jobs 32 \
        --job-index $i \
        &
done
# pkill -f "python.*test_discovery.py"