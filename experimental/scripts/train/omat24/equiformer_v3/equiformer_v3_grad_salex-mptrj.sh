#!/usr/bin/env bash
set -euo pipefail

NNODES=4
NPROC_PER_NODE=8
RDZV_PORT=29500

REMOTE_PREFIX=''

###########################
#       Script
###########################

MAIN_PATH="my_main.py"

LOG_DIR="/home/models/omat24/equiformer_v3"

CONFIG_PATH="experimental/configs/omat24/salex_mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_attn-hidden@32_rbf@64_max-neighbors@300_attn-grid@14-8_ffn-grid@14_attn-eps@1e-8_lr@0-5e-5-warmup@0.1-epochs@2-mptrj-salex-ratio@8-bs@256-wd@1e-3-beta2@0.98-eps@1e-6_pt-reg-dens-ft-no-reg-lr@1e-4.yml"
IDENTIFIER="salex-mptrj_grad-ft_N@7_L@4_no-dens"

PROJECT="equiformer_v3_salex-mptrj"


REMOTE_SCRIPT="$MAIN_PATH \
    --num-gpus ${NPROC_PER_NODE} \
    --num-nodes ${NNODES} \
    --mode train \
    --amp \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --seed 1 \
    --identifier $IDENTIFIER \
    --optim.num_workers=0 \
"


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
