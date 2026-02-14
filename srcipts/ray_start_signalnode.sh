export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网卡名称调整
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

ray start --head --num-gpus 8 --min-worker-port=10002 --max-worker-port=10101 --memory=1246835937280 --temp-dir=/dev/shm/ray_tmp


ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{
        "working_dir": "/your/working/dir/CDAS",
        "pip":  ["latex2sympy2", "timeout_decorator", "word2number"]
    }' -- /bin/bash ./scripts/train_grpo_cdas.sh