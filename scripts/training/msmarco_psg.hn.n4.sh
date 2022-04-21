#!/bin/bash

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)

#! --bsize 18, as we use 4 negative samples per query

EXP_ROOT_DIR=experiments/finetuned.b18.lr3e6.hn.n4
mkdir -p ${EXP_ROOT_DIR}

checkpoint=data/checkpoints/colbert.teacher.dnn

CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps 600000 --amp --bsize 18 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn4.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log

#TODO: sonic: gpu45-hn-n4