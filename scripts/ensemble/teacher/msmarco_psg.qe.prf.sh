#!/bin/bash
# Retrieve Unlabeled Positives using Expanded queries


device=$1 #TODO: input arg, e.g., "0"
fb_k=$2 #TODO: input arg, e.g., "10"
beta=$3 #TODO: input arg, e.g., "0.5"
fb_clusters=$4 #TODO: input arg, e.g., "24"
fb_docs=$5 #TODO: input arg, e.g., "3"
expansion_exp=$6 #TODO: input arg, e.g., "MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24"
fb_ranking=$7 #TODO: input arg, e.g., "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl"


# 0. Remove queries that are not included in qrels
org_queries=data/queries.train.tsv #TODO: custom path
qrels=data/qrels.train.tsv #TODO: custom path
queries=data/queries.train.reduced.tsv #TODO: custom path
[ ! -f "${queries}" ] && python -m preprocessing.utils.reduce_train_queries --queries ${org_queries} --qrels ${qrels} --out ${queries}


index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/colbert.teacher #TODO: custom path
# 
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -d "${experiment_root}" ] && echo "${experiment_root} does not exist." && return
[ ! -f "${fb_ranking}" ] && echo "${fb_ranking} does not exist." && return
CUDA_VISIBLE_DEVICES=${device} python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--root ${experiment_root} --experiment ${expansion_exp} \
\
--expansion_only \
--prf \
--fb_ranking ${fb_ranking} --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
\
--index_root ${index_root} --index_name MSMARCO.L2.32x200k  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
--batch --log-scores \
--queries ${queries} \
--checkpoint ${checkpoint} \
--qrels data/qrels.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv
