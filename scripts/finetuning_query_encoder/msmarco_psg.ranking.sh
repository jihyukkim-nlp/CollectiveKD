#!/bin/bash
cached_queries=$1 #TODO: e.g, "data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k10.tensor.cache.v2.pt"
index_root=$2 #TODO: e.g, "experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/index.py"
exp_root=$3 #TODO: e.g., "experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0"
step=$4 #TODO: e.g., "100000" 
device=$5 #TODO: e.g., "0"

echo;echo;echo
echo "\nInput args for $(cd $(dirname $0) && pwd)/$(basename $0) ==================\n"
echo "\t cached_queries=${cached_queries}"
echo "\t index_root=${index_root}"
echo "\t exp_root=${exp_root}"
echo "\t step=${step}"
echo "\t device=${device}"
echo "\nInput args for $(cd $(dirname $0) && pwd)/$(basename $0) ==================\n"
echo;echo;echo

[ ! -f "${cached_queries}" ] && echo "${cached_queries} does not exist." && return

checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
ckp_exp_root=${exp_root}/colbert-${step}.dnn
mkdir -p ${ckp_exp_root}

# sanity check
[ ! -d "${exp_root}/" ] && echo "${exp_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -d "${index_root}/" ] && echo "${index_root} does not exist." && return
# 
queries=data/queries.dev.small.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/qrels.dev.small.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return

# ANN search
topk=${ckp_exp_root}/MSMARCO-psg/retrieve.py/$(ls ${ckp_exp_root}/MSMARCO-psg/retrieve.py)/unordered.tsv
if [ ! -f ${topk} ];then
    echo "ANN Search using retrieve.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    CUDA_VISIBLE_DEVICES=${device} python -m query_encoder.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} --cached_queries ${cached_queries} --query_term_weight_act sigmoid \
    --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${ckp_exp_root} --experiment MSMARCO-psg
else
    echo "We have ANN search result at: \"${topk}\""
fi
echo;echo;echo



# Exact-NN search
topk=${ckp_exp_root}/MSMARCO-psg/retrieve.py/$(ls ${ckp_exp_root}/MSMARCO-psg/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
ranking=${ckp_exp_root}/MSMARCO-psg/rerank.py/$(ls ${ckp_exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
if [ ! -f ${ranking} ];then
    echo "Exact-NN Search using rerank.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    echo "topk      : ${topk}"
    CUDA_VISIBLE_DEVICES=${device} python -m query_encoder.rerank --topk ${topk} --batch --log-scores --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} --cached_queries ${cached_queries} --query_term_weight_act sigmoid \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${ckp_exp_root} --experiment MSMARCO-psg
else
    echo "We have Exact-NN search result at: \"${ranking}\""
fi
echo;echo;echo

# evaluate
ranking=${ckp_exp_root}/MSMARCO-psg/rerank.py/$(ls ${ckp_exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
result=${ckp_exp_root}/MSMARCO-psg/e2e.metrics
python -m utility.evaluate.msmarco_passages --qrels ${qrels} --ranking ${ranking} > ${result}
echo;echo;echo
cat ${result}
echo;echo;echo
