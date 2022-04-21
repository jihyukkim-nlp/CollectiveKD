#!/bin/bash
device=$1 # e.g., "0"
exp_root=$2 # e.g., "experiments/kd_on_bm25_negatives/static_kd/ce_ensemble"
step=$3 # e.g., "100000"

[ ! -d "${exp_root}/" ] && echo "${exp_root} does not exist." && return
checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return

#! hard coded paths
collection=data/msmarco-pass/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return
queries=data/msmarco-pass/queries.dev.small.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/msmarco-pass/qrels.dev.small.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
topk=data/msmarco-pass/top1000.dev
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return

mkdir -p ${exp_root}/MSMARCO-psg/test.py

if [ ! -f ${exp_root}/MSMARCO-psg/test.py/${step}.log ];then
    CUDA_VISIBLE_DEVICES=${device} \
    python -m colbert.test --checkpoint ${checkpoint} \
    --amp --doc_maxlen 180 --mask-punctuation \
    --collection ${collection} --queries ${queries} --qrels ${qrels} --topk ${topk} \
    --root ${exp_root} --experiment MSMARCO-psg > ${exp_root}/MSMARCO-psg/test.py/${step}.log
else
    tail -15 ${exp_root}/MSMARCO-psg/test.py/${step}.log
fi