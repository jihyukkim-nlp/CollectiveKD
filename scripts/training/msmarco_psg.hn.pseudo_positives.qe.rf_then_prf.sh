#!/bin/bash

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# - UP (Unlabeled Positives) from pre-trained ColBERT (self-training)

EXP_ROOT_DIR=experiments/finetuned.b36.lr3e6.hn.pseudo_positives.qe.rf_then_prf
mkdir -p ${EXP_ROOT_DIR}

checkpoint=data/checkpoints/colbert.teacher.dnn

CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29600 \
-m colbert.train --maxsteps 600000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/triples.train.small.ids.hn.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log

#TODO: sonic: gpu45-hn-pseudo_positives-rf_then_prf
