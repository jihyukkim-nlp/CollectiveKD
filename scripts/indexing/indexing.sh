#!/bin/bash
indexing_devices=$1 #TODO: input arg, e.g., "0,1"
faiss_devices=$2 #TODO: input arg, e.g., "0,1"
exp_root=$3 #TODO: input arg, e.g., "experiments/colbert.teacher"
step=$4 #TODO: input arg, e.g., "50000"

DATA_DIR=/workspace/DataCenter/PassageRanking/MSMARCO # sonic
# DATA_DIR=/workspace/DataCenter/MSMARCO # dilab

# sanity check
checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -d "${exp_root}/" ] && echo "${exp_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
collection=${DATA_DIR}/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return

index_root=${exp_root}/MSMARCO-psg/index.py

# index
n_devices=$(echo ${indexing_devices} | awk -F "," '{ print NF }')
CUDA_VISIBLE_DEVICES=${indexing_devices} OMP_NUM_THREADS=${n_devices} \
python -m torch.distributed.launch --nproc_per_node=${n_devices} --master_addr 127.0.0.1 --master_port 30000 \
-m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--checkpoint ${checkpoint} --collection ${collection} \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--root ${exp_root} --experiment MSMARCO-psg

# index_faiss
CUDA_VISIBLE_DEVICES=${faiss_devices} python -m colbert.index_faiss \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 --slices 1 \
--root ${exp_root} --experiment MSMARCO-psg
