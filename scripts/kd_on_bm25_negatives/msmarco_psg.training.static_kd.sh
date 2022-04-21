#!/bin/bash
devices=$1 # e.g., "0,1"
master_port=$2 # e.g., "29500"
exp_root=$3 # e.g., "experiments/kd_on_bm25_negatives/dual_supervision/lambda0.5.ce_single.prf.beta1.0.b36.lr3e6.bm25n"
maxsteps=$4 # e.g., "600000"
triples=$5 # e.g., "data/msmarco-pass/triples.bm25n.sebastian_ensemble.jsonl" or "triples.bm25n.sebastian_single.jsonl"
static_supervision=$6 # e.g., "data/msmarco-pass/cross_encoder_scores/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv" or "data/msmarco-pass/cross_encoder_scores/bertbase_cat_msmarcopassage_train_scores_ids.tsv"

echo;echo;echo "Input arguments"
echo "devices                   :${devices}"
echo "master_port               :${master_port}"
echo "exp_root                  :${exp_root}"
echo "maxsteps                  :${maxsteps}"
echo "triples                   :${triples}"
echo "static_supervision        :${static_supervision}"
echo;echo;echo

# #?@ debugging ####################################################################################
# # head -1000 data/msmarco-pass/triples.bm25n.sebastian_ensemble.jsonl > data/msmarco-pass/triples.minimal_examples.jsonl
# # head -1000 data/msmarco-pass/cross_encoder_scores/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv > data/msmarco-pass/cross_encoder_scores/bert_single.minimal_examples.tsv
# CUDA_VISIBLE_DEVICES=7 \
# python -m colbert.train --maxsteps 10 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples data/msmarco-pass/triples.minimal_examples.jsonl \
# --queries data/msmarco-pass/queries.train.tsv --collection data/msmarco-pass/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root experiments/kd_on_bm25_negatives/debugging --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --static_supervision data/msmarco-pass/cross_encoder_scores/bert_single.minimal_examples.tsv --dual_supervision_lambda 0.0
# rm -r experiments/kd_on_bm25_negatives/debugging
# return
# #?@ debugging ####################################################################################

[ ! -f ${triples} ] && echo "${triples} does not exist" && return
[ ! -f ${static_supervision} ] && echo "${static_supervision} does not exist" && return

#! hard coded paths
queries=data/msmarco-pass/queries.train.tsv
[ ! -f ${queries} ] && echo "${queries} does not exist" && return
collection=data/msmarco-pass/collection.tsv
[ ! -f ${collection} ] && echo "${collection} does not exist" && return

n_devices=$(echo ${devices} | awk -F "," '{ print NF }')
[ ! ${n_devices} -eq 2 ] && echo "n_devices should be 2, but ${n_devices} (devices=${devices}) is given." && return

mkdir -p ${exp_root}

CUDA_VISIBLE_DEVICES=${devices} \
python -m torch.distributed.launch --nproc_per_node=${n_devices} --master_addr 127.0.0.1 --master_port ${master_port} \
-m colbert.train --maxsteps ${maxsteps} --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples ${triples} --queries ${queries} --collection ${collection} \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${exp_root} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--kd_temperature 0.25 \
--static_supervision ${static_supervision} --dual_supervision_lambda 0.0 \
> ${exp_root}/train.log
