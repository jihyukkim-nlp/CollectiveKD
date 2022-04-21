#!/bin/bash
devices=$1 # e.g., "0,1"
master_port=$2 # e.g., "29500"
exp_root=$3 # e.g., "finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5"
kd_expansion_pt1=$4 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
kd_expansion_pt2=$5 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"
kd_expansion_pt3=$6 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"

[ -z "${kd_expansion_pt1}" ] && echo "At least one kd_expansion_pt must be given." && return
[ ! -z "${kd_expansion_pt1}" ] && echo;echo "kd_expansion_pt1 is given: ${kd_expansion_pt1}";echo && [ ! -f ${kd_expansion_pt1} ] && echo "${kd_expansion_pt1} does not exist" && return
[ ! -z "${kd_expansion_pt2}" ] && echo;echo "kd_expansion_pt2 is given: ${kd_expansion_pt2}";echo && [ ! -f ${kd_expansion_pt2} ] && echo "${kd_expansion_pt2} does not exist" && return
[ ! -z "${kd_expansion_pt3}" ] && echo;echo "kd_expansion_pt3 is given: ${kd_expansion_pt3}";echo && [ ! -f ${kd_expansion_pt3} ] && echo "${kd_expansion_pt3} does not exist" && return

kd_expansion_pt_list=""
[ ! -z "${kd_expansion_pt1}" ] && kd_expansion_pt_list=${kd_expansion_pt_list}" ${kd_expansion_pt1}"
[ ! -z "${kd_expansion_pt2}" ] && kd_expansion_pt_list=${kd_expansion_pt_list}" ${kd_expansion_pt2}"
[ ! -z "${kd_expansion_pt3}" ] && kd_expansion_pt_list=${kd_expansion_pt_list}" ${kd_expansion_pt3}"

echo;echo;echo;echo "${kd_expansion_pt_list}";echo;echo;echo

#* Fine-tuning ColBERT using labeled positives and hard negatives
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - Using ensemble of teachers

debugging_dir=experiments/ensemble/debugging
EXP_ROOT_DIR=${debugging_dir}
mkdir -p ${debugging_dir}

checkpoint=data/checkpoints/colbert.teacher.dnn
[ ! -f ${checkpoint} ] && echo "${checkpoint} does not exist" && return

CUDA_VISIBLE_DEVICES=${devices} \
python -m colbert.train --maxsteps 400000 --amp --bsize 8 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion \
--kd_expansion_pt_list ${kd_expansion_pt_list} \
--checkpoint ${checkpoint}
rm -r ${debugging_dir}
# --resume_optimizer \