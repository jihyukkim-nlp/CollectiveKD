#!/bin/bash
exp_root=$1 #TODO: input arg
[ ! -d "${exp_root}/" ] && echo "${exp_root} does not exist." && return

DATA_DIR=/workspace/DataCenter/PassageRanking/MSMARCO # sonic
# DATA_DIR=/workspace/DataCenter/MSMARCO # dilab

# sanity check
checkpoint=data/checkpoints/colbert.teacher.dnn
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
collection=${DATA_DIR}/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return

index_root=${exp_root}/MSMARCO-psg/index.py

# index
#?@ temporary: uncomment
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.0.0.1 --master_port 30000 \
#?@ temporary
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 \
python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 30000 \
-m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--checkpoint ${checkpoint} --collection ${collection} \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--root ${exp_root} --experiment MSMARCO-psg

# index_faiss
# #?@ temporary: uncomment
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m colbert.index_faiss \
#?@ temporary
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m colbert.index_faiss \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 --slices 1 \
--root ${exp_root} --experiment MSMARCO-psg
