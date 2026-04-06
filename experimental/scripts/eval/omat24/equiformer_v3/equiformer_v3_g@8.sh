MAIN_PATH="my_main.py"
LOG_DIR="/home/models/omat24/equiformer_v3/"
CONFIG_PATH="experimental/configs/omat24/omat24/experiments/direct/equiformer_v3_N@7_L@4_attn-hidden@32_rbf@64_max-neighbors@300_attn-grid@14-8_ffn-grid@14_use-gate-force-head_merge-layer-norm_epochs@4-bs@512-wd@1e-3-beta2@0.98-eps@1e-6_dens-p@0.5-std@0.025-r@0.5-0.75-w@1-no-stress-max-f@2.5_no-amp.yml"
IDENTIFIER="direct_L@4"

CHECKPOINT="/home/models/omat24/equiformer_v3/checkpoint.pt"


python -u -m torch.distributed.launch --nproc_per_node=8 $MAIN_PATH \
    --num-gpus 8 \
    --mode validate \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --amp \
    --seed 1 \
    --identifier $IDENTIFIER \
    --optim.num_workers=0 \
    --optim.batch_size=8 \
    --optim.eval_batch_size=16 \
    --optim.grad_accumulation_steps=1 \
    --checkpoint $CHECKPOINT


' :
for split in id ood_ads ood_cat ood_both
do
    python -u -m torch.distributed.launch --nproc_per_node=8 $MAIN_PATH \
        --distributed \
        --num-gpus 8 \
        --mode validate \
        --config-yml $CONFIG_PATH \
        --run-dir $LOG_DIR \
        --print-every 200 \
        --amp \
        --seed 1 \
        --identifier $IDENTIFIER-$split \
        --dataset.train.src=$TRAIN_PATH \
        --dataset.val.src=$VAL_PATH/val_$split \
        --optim.eval_every=31250 \
        --optim.num_workers=0 \
        --logger.project=$PROJECT \
        --optim.batch_size=8 \
        --optim.eval_batch_size=16 \
        --optim.grad_accumulation_steps=1 \
        --checkpoint $CHECKPOINT
done
'
