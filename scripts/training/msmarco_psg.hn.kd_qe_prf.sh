#!/bin/bash

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
# - Using PRF instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt)

EXP_ROOT_DIR=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf
mkdir -p ${EXP_ROOT_DIR}

checkpoint=data/checkpoints/colbert.teacher.dnn

CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29600 \
-m colbert.train --maxsteps 600000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion --kd_expansion_pt experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log

#TODO: sonic: gpu45-hn-kd_qe_kmeans-prf