#!/bin/bash
devices=$1 # e.g., "0,1"
lr=$2 # e.g., "3e-06" (default)
#TODO: sonic: gpu01-hn-kd-prf-2nd_kd

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
# - Using PRF (beta 0.5) instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt)
# The 2nd iteration KD training using better student and teacher
# - train triples (``triples``): hard negatives are obtained by previously fine-tuned ColBERT
# - teacher (``kd_expansion_pt``): expansion embeddings are obtained by previously fine-tuned ColBERT
# - using small learning rate for the n-th fine-tuning (n>=2) 

EXP_ROOT_DIR=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd.lr${lr}
mkdir -p ${EXP_ROOT_DIR}

checkpoint=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn

triples=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/triples.hn1.jsonl
[ -f ${triples} ] && echo "${triples} does not exist"
kd_expansion_pt=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train-kmeans.prf.docs3.k10.beta0.5.clusters24/label.py/2021-10-31_11.12.15/expansion.pt
[ -f ${kd_expansion_pt} ] && echo "${kd_expansion_pt} does not exist"

CUDA_VISIBLE_DEVICES=${devices} \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps 400000 --amp --bsize 36 --lr ${lr} --accum 1 \
--triples ${triples} \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion --kd_expansion_pt ${kd_expansion_pt} \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log
