#!/bin/bash

devices=$1 # e.g., "0,1"
master_port=$2 # e.g., "29500"
exp_root=$3 # e.g., "finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5"
cached_queries=$4 # e.g., "data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.v2.pt"
checkpoint=$5 # e.g., "experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn"
triples=$6 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl" or "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn4.jsonl" or "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.hn8.jsonl"
bsize=$7 # e.g., "18" or "24" or "36"
# 
kd_expansion_pt1=$8 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
kd_expansion_pt2=$9 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"
kd_expansion_pt3=$10 # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"


echo;echo;echo
echo "\nInput args for $(cd $(dirname $0) && pwd)/$(basename $0) ==================\n"
echo "\t devices=${devices}"
echo "\t master_port=${master_port}"
echo "\t exp_root=${exp_root}"
echo "\t cached_queries=${cached_queries}"
echo "\t checkpoint=${checkpoint}"
echo "\t triples=${triples}"
echo "\t bsize=${bsize}"
echo "\t kd_expansion_pt1=${kd_expansion_pt1}"
echo "\t kd_expansion_pt2=${kd_expansion_pt2}"
echo "\t kd_expansion_pt3=${kd_expansion_pt3}"
echo "\nInput args for $(cd $(dirname $0) && pwd)/$(basename $0) ==================\n"
echo;echo;echo

[ ! -f ${cached_queries} ] && echo "${cached_queries} does not exist" && return
[ ! -f ${checkpoint} ] && echo "${checkpoint} does not exist" && return
[ ! -f ${triples} ] && echo "${triples} does not exist" && return

[ -z "${kd_expansion_pt1}" ] && echo "At least one kd_expansion_pt must be given." && return
[ ! -z "${kd_expansion_pt1}" ] && echo;echo "kd_expansion_pt1 is given: ${kd_expansion_pt1}";echo && [ ! -f ${kd_expansion_pt1} ] && echo "${kd_expansion_pt1} does not exist" && return
[ ! -z "${kd_expansion_pt2}" ] && echo;echo "kd_expansion_pt2 is given: ${kd_expansion_pt2}";echo && [ ! -f ${kd_expansion_pt2} ] && echo "${kd_expansion_pt2} does not exist" && return
[ ! -z "${kd_expansion_pt3}" ] && echo;echo "kd_expansion_pt3 is given: ${kd_expansion_pt3}";echo && [ ! -f ${kd_expansion_pt3} ] && echo "${kd_expansion_pt3} does not exist" && return

kd_expansion_pt_list=""
[ ! -z "${kd_expansion_pt1}" ] && kd_expansion_pt_list=${kd_expansion_pt_list}" ${kd_expansion_pt1}"
[ ! -z "${kd_expansion_pt2}" ] && kd_expansion_pt_list=${kd_expansion_pt_list}" ${kd_expansion_pt2}"
[ ! -z "${kd_expansion_pt3}" ] && kd_expansion_pt_list=${kd_expansion_pt_list}" ${kd_expansion_pt3}"

echo;echo;echo;echo "${kd_expansion_pt_list}";echo;echo;echo

collection=/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv
[ ! -f ${collection} ] && collection=/workspace/DataCenter/MSMARCO/collection.tsv
[ ! -f ${collection} ] && echo "${collection} does not exist" && return

# n_devices=$(echo ${devices} | awk -F "," '{ print NF }')
# [ ! ${n_devices} -eq 2 ] && echo "n_devices should be 2, but ${n_devices} (devices=${devices}) is given." && return

#* Fine-tuning ColBERT Query Encoder with expansion tokens from docT5query
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
# Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
# - Using ensemble of teachers
# - Averaging losses from different teachers

debugging_dir=experiments/finetuning_query_encoder/debugging
EXP_ROOT_DIR=${debugging_dir}
mkdir -p ${debugging_dir}

teacher_checkpoint=data/checkpoints/colbert.teacher.dnn
[ ! -f ${teacher_checkpoint} ] && echo "${teacher_checkpoint} does not exist" && return

CUDA_VISIBLE_DEVICES=${devices} \
python -m query_encoder.train --maxsteps 400000 --amp --bsize ${bsize} --lr 3e-06 --accum 1 \
--triples ${triples} \
--queries data/queries.train.reduced.tsv --cached_queries ${cached_queries} --collection ${collection} \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${teacher_checkpoint} \
--kd_query_expansion \
--kd_expansion_pt_list ${kd_expansion_pt_list} \
--checkpoint ${checkpoint}
rm -r ${debugging_dir}