#!/bin/bash
devices=$1 # e.g., "0,1"
kd_expansion_pt=$2 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
# kd_expansion_pt=${prev_exp}/$(ls ${prev_exp}/MSMARCO-psg-train-*/label.py/*/expansion.pt)
checkpoint=$3 # e.g., "data/checkpoints/colbert.teacher.dnn", "experiments/colbert.teacher/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn"
# checkpoint=${prev_exp}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-50000.dnn
new_exp=$4 # e.g., "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k"
# prev_exp=$2 # e.g., "experiments/colbert.teacher"
maxsteps=$5 # e.g., ``50000`` or ``100000`` or ``150000``

if [ "${maxsteps}" -eq "50000" ];then
    resume=""
else
    resume="--resume"
fi

[ ! -f ${kd_expansion_pt} ] && echo "${kd_expansion_pt} does not exist" && return
[ ! -f ${checkpoint} ] && echo "${checkpoint} does not exist" && return
[ ! -d ${new_exp} ] && echo "${new_exp} does not exist. \nPlease do this before the execution: \n\n\t mkdir -p ${new_exp}\n" && return

triples=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl
[ ! -f ${triples} ] && echo "${triples} does not exist" && return

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
# - Using PRF (beta 0.5) instead of RF
# Updating expansion embeddings step-wise for every 50k iterations, for mutual learning between the teacher and the student
# 
# - 1. [pre-training   ] (300k iterations) "colbert.teacher.dnn", trained using BM25 negatives (official train triples)
# - update ranking: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.[jsonl/tsv]"
# - update kd_expansion_pt: "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
# - update triples with HN: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
# 
# - 2. [1st fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k", trained using HN from the pre-trained retrieval 
# - update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
# - update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"
# 
# - 3. [2nd fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k", trained using HN from the pre-trained retrieval 
# - update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
# - update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"
# 
# - 4. [3rd fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k", trained using HN from the pre-trained retrieval 
# - update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
# - update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"

EXP_ROOT_DIR=${new_exp}
mkdir -p ${EXP_ROOT_DIR}

# -m colbert.train --maxsteps 50000 --amp --bsize 36 --lr 3e-06 --accum 1 \
CUDA_VISIBLE_DEVICES=${devices} \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps ${maxsteps} --amp --bsize 36 --accum 1 \
--triples ${triples} \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion --kd_expansion_pt ${kd_expansion_pt} \
--resume_optimizer ${resume} \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log

#TODO: sonic: gpu01-50k