source /mnt/bn/audio-visual-llm-data2/yuwenyi/conda_env.sh
conda activate salmonn_tts

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

torchrun --nproc_per_node=8 --master_addr $METIS_WORKER_0_HOST --master_port $port train.py \
    --cfg-path /mnt/bn/audio-visual-llm-data5/wangsiyin/SALMONN/configs/config.yaml