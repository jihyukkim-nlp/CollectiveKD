#!/bin/bash
exp_root=$1 #TODO: input arg
step=$2 #TODO: input arg
device=$3 #TODO: input arg

echo;echo;echo

qrels=data/msmarco-pass/qrels.dev.small.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return

ranking=${exp_root}/MSMARCO-psg/rerank.py/$(ls ${exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
if [ -f "${ranking}" ];then
    python -m utility.evaluate.msmarco_passages --qrels ${qrels} --ranking ${ranking}
    return
fi


checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -d "${exp_root}/" ] && echo "${exp_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
index_root=${exp_root}/MSMARCO-psg/index.py
[ ! -d "${index_root}/" ] && echo "${index_root} does not exist." && return
queries=data/msmarco-pass/queries.dev.small.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return

# ANN search
topk=${exp_root}/MSMARCO-psg/retrieve.py/$(ls ${exp_root}/MSMARCO-psg/retrieve.py)/unordered.tsv
if [ ! -f ${topk} ];then
    echo "ANN Search using retrieve.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${exp_root} --experiment MSMARCO-psg
else
    echo "We have ANN search result at: \"${topk}\""
fi
echo;echo;echo

# Exact-NN search
topk=${exp_root}/MSMARCO-psg/retrieve.py/$(ls ${exp_root}/MSMARCO-psg/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
ranking=${exp_root}/MSMARCO-psg/rerank.py/$(ls ${exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
if [ ! -f ${ranking} ];then
    echo "Exact-NN Search using rerank.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    echo "topk      : ${topk}"
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.rerank --topk ${topk} --batch --log-scores --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${exp_root} --experiment MSMARCO-psg
else
    echo "We have Exact-NN search result at: \"${ranking}\""
fi
echo;echo;echo

# evaluate
ranking=${exp_root}/MSMARCO-psg/rerank.py/$(ls ${exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m utility.evaluate.msmarco_passages --qrels ${qrels} --ranking ${ranking}
echo;echo;echo
