#!/bin/bash

#* Fine-tuning ColBERT
# Training using new train triples: 
# - HN (Hard Negatives) from pre-trained ColBERT 
# The 2nd iteration fine-tuning without KD
# - train triples (``triples``): hard negatives are obtained by previously fine-tuned ColBERT

#TODO: ###########################################################################
#?@ debugging
echo;echo;echo
echo "check CUDA_VISIBLE_DEVICES"
echo "check checkpoint: previously fine-tuned ColBERT is now pre-trained ColBERT"
echo "check EXP_ROOT_DIR: explicitly denote that this is 2nd fine-tuning"
echo "check train triples: hard negatives are obtained by previously fine-tuned ColBERT"
echo "(return) check this code: \t\"$(cd $(dirname $0) && pwd)/msmarco_psg.training.sh\"" && return
#?@ debugging
#TODO: ###########################################################################

#TODO: Done: check EXP_ROOT_DIR
EXP_ROOT_DIR=experiments/finetuned.b36.lr3e6.hn.2nd
mkdir -p ${EXP_ROOT_DIR}

#TODO: Done: check checkpoint: previously fine-tuned ColBERT is now pre-trained ColBERT
checkpoint=experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-50000.dnn

#TODO: check train triples: hard negatives are obtained by previously fine-tuned ColBERT
#TODO: check "scripts/wo_kd_2nd_iter/msmarco_psg.meta_script.sh"
triples=experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg-train/label.py/triples.hn1.jsonl
[ -f ${triples} ] && echo "${triples} does not exist"

#TODO: check CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps 200000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples ${triples} \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--checkpoint ${checkpoint} > ${EXP_ROOT_DIR}/train.log

#TODO: sonic: gpu01-hn-2nd