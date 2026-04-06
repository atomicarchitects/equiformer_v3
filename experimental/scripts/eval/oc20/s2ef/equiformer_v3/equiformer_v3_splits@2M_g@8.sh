MAIN_PATH="my_main.py"
LOG_DIR="/home/models/oc20/equiformer_v3/2M"
CONFIG_PATH="experimental/configs/oc20/2M/equiformer_v3/experiments/base_N@8-L@6-C@128-attn-hidden@64-ffn@512-envelope-num-rbf@128_merge-layer-norm_gates2-gridmlp_use-gate-force-head_wd@1e-3-grad-clip@100_lin-ref-e@4.yml"
IDENTIFIER="base"

CHECKPOINT="/home/models/oc20/equiformer_v3/2M/base/checkpoint.pt"
VAL_PATH="/home/datasets/oc20/all/val_id"


python -u -m torch.distributed.launch --nproc_per_node=8 $MAIN_PATH \
    --num-gpus 8 \
    --mode validate \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --amp \
    --seed 1 \
    --identifier $IDENTIFIER \
    --dataset.val.src=$VAL_PATH \
    --optim.eval_every=31250 \
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