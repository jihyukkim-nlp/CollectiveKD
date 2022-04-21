#!/bin/bash
devices=$1 # e.g., "0,1"
master_port=$2 # e.g., "29500"
exp_root=$3 # e.g., "experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.kd.rf.beta0.5"
kd_expansion_pt=$4 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"
# 
prevstep=$5 # e.g., "400000"
maxsteps=$6 # e.g., "600000"

checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${prevstep}.dnn

[ ! -f ${checkpoint} ] && echo "${checkpoint} does not exist" && return
[ ! -f ${kd_expansion_pt} ] && echo "${kd_expansion_pt} does not exist" && return

echo;echo;echo "change exp_root: \n\t${exp_root}\n\t--> ${exp_root}.resume";echo;echo
exp_root=${exp_root}.resume

#* Training ColBERT using labeled positive and BM25 negatives
# Train triple: "data/triples.train.small.ids.jsonl"
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - kd_expansion_pt (PRF): "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
# - kd_expansion_pt (RF) : "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"

EXP_ROOT_DIR=${exp_root}
mkdir -p ${EXP_ROOT_DIR}

CUDA_VISIBLE_DEVICES=${devices} \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port ${master_port} \
-m colbert.train --maxsteps ${maxsteps} --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples data/triples.train.small.ids.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint data/checkpoints/colbert.teacher.dnn \
--kd_query_expansion --kd_expansion_pt ${kd_expansion_pt} \
--resume --resume_optimizer --checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log
