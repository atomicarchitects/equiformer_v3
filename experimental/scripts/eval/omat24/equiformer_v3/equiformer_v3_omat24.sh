MAIN_PATH="my_main.py"
LOG_DIR="models/omat24/equiformer_v3"

CONFIG_PATH="experimental/configs/omat24/omat24/experiments/gradient/1226-2_equiformer_v3_grad-finetune_N@7_L@4_attn-hidden@32_rbf@64_max-neighbors@300_attn-grid@14-8_ffn-grid@14_merge-layer-norm_lr@0-1e-4-epochs@2-bs@512-wd@1e-3-beta2@0.98-eps@1e-6_pt-reg-dens-ft-no-reg.yml"

IDENTIFIER="eval_1226-2_grad-ft_N@7_L@4_lr@0-1e-4-epochs@2-bs@512-beta2@0.98-eps@1e-6_rand-e"
PROJECT="equiformer_v3_omat24"

CHECKPOINT_PATH="/home/ylliao/fairchem_omat24_08-04-2025/models/omat24/equiformer_v3/trained_checkpoints/1226-2_grad-ft_N@7_L@4_lr@0-1e-4-epochs@2-bs@512-beta2@0.98-eps@1e-6_rand-e/checkpoint.pt"


python -u -m torch.distributed.launch --nproc_per_node=4 $MAIN_PATH \
    --num-gpus 4 \
    --mode validate \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --seed 1 \
    --identifier $IDENTIFIER \
    --logger.project=$PROJECT \
    --optim.eval_batch_size=4 \
    --checkpoint $CHECKPOINT_PATH

#     --distributed \
