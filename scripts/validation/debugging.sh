#!/bin/bash
device=$1 #TODO: input arg
exp=$2 #TODO: input arg
step=$3 #TODO: input arg

# DATA_DIR=/workspace/DataCenter/PassageRanking/MSMARCO # sonic
DATA_DIR=/workspace/DataCenter/MSMARCO # dilab

# sanity check
checkpoint=experiments/${exp}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -d "experiments/${exp}/" ] && echo "experiments/${exp} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
collection=${DATA_DIR}/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return
queries=data/queries.dev.small.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/qrels.dev.small.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
topk=${DATA_DIR}/top1000.dev
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return

CUDA_VISIBLE_DEVICES=${device} \
python -m colbert.test --checkpoint ${checkpoint} \
--amp --doc_maxlen 180 --mask-punctuation \
--collection ${collection} --queries ${queries} --qrels ${qrels} --topk ${topk} \
--root experiments/${exp} --experiment MSMARCO-psg
