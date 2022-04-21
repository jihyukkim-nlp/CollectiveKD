#!/bin/bash

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# - UP (Unlabeled Positives) from pre-trained ColBERT (self-training)

EXP_ROOT_DIR=experiments/finetuned.b36.lr3e6.hn.up
mkdir -p ${EXP_ROOT_DIR}

checkpoint=data/checkpoints/colbert.teacher.dnn

CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29800 \
-m colbert.train --maxsteps 600000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.train.small.ids.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log
