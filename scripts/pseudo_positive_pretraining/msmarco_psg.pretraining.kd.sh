#!/bin/bash
devices=$1 # e.g., "0,1"
master_port=$2 # e.g., "29500"
pseudo_positive_triple=$3 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv"
exp_root=$4 # e.g., "experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.kd.rf.beta0.5"
kd_expansion_pt=$5 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"

[ ! -f ${pseudo_positive_triple} ] && echo "${pseudo_positive_triple} does not exist" && return
[ ! -f ${kd_expansion_pt} ] && echo "${kd_expansion_pt} does not exist" && return

#* Pre-tuning ColBERT using pseudo-positives
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# - PP (Pseudo Positives) from pre-trained ColBERT using expanded query from RF ("experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.*.tsv")
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query

EXP_ROOT_DIR=${exp_root}
mkdir -p ${EXP_ROOT_DIR}

checkpoint=data/checkpoints/colbert.teacher.dnn
[ ! -f ${checkpoint} ] && echo "${checkpoint} does not exist" && return

CUDA_VISIBLE_DEVICES=${devices} \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port ${master_port} \
-m colbert.train --maxsteps 400000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples ${pseudo_positive_triple} \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--resume_optimizer \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion --kd_expansion_pt ${kd_expansion_pt} \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log
