#!/bin/bash

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
# - Using PRF with RF, by appending RF docs in front of PRF docs (beta 0.5) (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_prepend_rf.docs3.k10.beta0.5.clusters24/label.py/2021-10-30_15.08.50/expansion.pt)

EXP_ROOT_DIR=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5
mkdir -p ${EXP_ROOT_DIR}

checkpoint=data/checkpoints/colbert.teacher.dnn

CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps 200000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion --kd_expansion_pt experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_prepend_rf.docs3.k10.beta0.5.clusters24/label.py/2021-10-30_15.08.50/expansion.pt \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log

#TODO: sonic: gpu67-hn-kd_qe_kmeans-prf_prepend_rf-beta5