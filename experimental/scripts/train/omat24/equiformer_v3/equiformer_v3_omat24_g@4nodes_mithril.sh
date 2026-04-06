#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="${SSH_KEY:-/mnt/data2/projects/yilunliao/ssh_key/yi-lun.pem}"
SSH_USER="${SSH_USER:-ubuntu}"
NNODES=4
NPROC_PER_NODE=8
RDZV_PORT=29500
#REMOTE_PREFIX='export NCCL_SOCKET_IFNAME=enp6s0 && export NCCL_IB_DISABLE=1;'
#REMOTE_PREFIX='export NCCL_DEBUG=INFO && export TORCH_DISTRIBUTED_DEBUG=DETAIL;'

# for spot
REMOTE_PREFIX='export NCCL_NET_GDR_LEVEL=LOC;'
# for reservation
#REMOTE_PREFIX=''

###########################
#       Script
###########################
HOSTLIST="/mnt/data2/projects/yilunliao/ssh_key/hostlist_mp-1b-twelve.txt"

TORCHRUN="/mnt/data2/projects/yilunliao/anaconda3/envs/equiformer_v3_py311/bin/torchrun"
PYTHON="/mnt/data2/projects/yilunliao/anaconda3/envs/equiformer_v3_py311/bin/python"

MAIN_PATH="/mnt/data2/projects/yilunliao/fairchem_omat24_08-04-2025/my_main.py"

LOG_DIR="/mnt/data2/projects/yilunliao/fairchem_omat24_08-04-2025/models/omat24/equiformer_v3"

CONFIG_PATH="/mnt/data2/projects/yilunliao/fairchem_omat24_08-04-2025/experimental/configs/omat24/omat24/experiments/direct/0103-2_equiformer_v3_N@7_L@6_attn-hidden@32_rbf@64_max-neighbors@300_attn-grid@20-8_ffn-grid@20_use-gate-force-head_merge-layer-norm_epochs@4-bs@512-wd@1e-3-beta2@0.98-eps@1e-6_dens-p@0.5-std@0.025-r@0.5-0.75-w@1-no-stress-max-f@2.5_no-amp.yml"

IDENTIFIER="0103-2_direct_N@7_L@6_epochs@4-bs@512-beta2@0.98-eps@1e-6_dens-w@1-r@0.5-0.75-max-f@2.5_no-amp"

CHECKPOINT="/mnt/data2/projects/yilunliao/fairchem_omat24_08-04-2025/models/omat24/equiformer_v3/checkpoints/2026-01-11-23-21-36-0103-2_direct_N@7_L@6_epochs@4-bs@512-beta2@0.98-eps@1e-6_dens-w@1-r@0.5-0.75-max-f@2.5_no-amp/checkpoint.pt"


#TRAIN_PATH="/mnt/data/projects/yilunliao/datasets/omat24/mptrj/aselmdb_uncorrected_total_energy"
#VAL_PATH="/mnt/data/projects/yilunliao/datasets/omat24/salex/val_30k"
LIN_REF_PATH="/mnt/data2/projects/yilunliao/fairchem_omat24_08-04-2025/experimental/configs/omat24/omat24/preprocessing/omat24_energy_element_references.npz"

PROJECT="equiformer_v3_omat24"


REMOTE_SCRIPT="$MAIN_PATH \
    --num-gpus ${NPROC_PER_NODE} \
    --num-nodes ${NNODES} \
    --mode train \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --seed 1 \
    --identifier $IDENTIFIER \
    --checkpoint $CHECKPOINT \
    --optim.num_workers=0 \
    --logger.project=$PROJECT \
    --dataset.train.transforms.element_references.energy.file=$LIN_REF_PATH \
"

# --amp


if [[ ! -f "$HOSTLIST" ]]; then
  echo "Missing $HOSTLIST"; exit 1
fi
if [[ $(wc -l < "$HOSTLIST" | tr -d ' ') -ne $NNODES ]]; then
  echo "Expected $NNODES lines in $HOSTLIST"; exit 1
fi

mapfile -t HOSTS < "$HOSTLIST"
MASTER_IP="${HOSTS[0]}"
#MASTER_PRIVATE_IP=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
#  "${SSH_USER}@${MASTER_IP}" "ip -4 -o addr show enp6s0 | awk '{print \$4}' | cut -d/ -f1")
MASTER_PRIVATE_IP="${HOSTS[0]}"

echo "MASTER PUBLIC IP: ${MASTER_IP}"
echo "RENDEZVOUS ENDPOINT: ${MASTER_PRIVATE_IP}:${RDZV_PORT}"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "SCRIPT: $REMOTE_SCRIPT"
echo

rank=0
for ip in "${HOSTS[@]}"; do
  echo "[launch] node_rank=$rank @ $ip"
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o UserKnownHostsFile=/dev/null \
    "${SSH_USER}@${ip}" \
    "bash -lc 'set -euo pipefail; \
      ulimit -n 65536; \
      ${REMOTE_PREFIX} \
      nohup ${TORCHRUN} \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --rdzv-id=10 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_PRIVATE_IP}:${RDZV_PORT} \
        --node_rank=${rank} \
        ${REMOTE_SCRIPT} \
        > \$HOME/torchrun_${rank}.log 2>&1 & echo \$! > \$HOME/torchrun_${rank}.pid'"
  ((rank+=1))
done

echo
echo "Launched. Logs will be on each node as ~/torchrun_<rank>.log ; PIDs in ~/torchrun_<rank>.pid."
