#!/bin/bash
exp_root=$1 #TODO: input arg
step=$2 #TODO: input arg
device=$3 #TODO: input arg
# 
fb_docs=$4 #TODO: input arg, e.g., "3"
fb_clusters=$5 #TODO: input arg, e.g., "24"
fb_k=$6 #TODO: input arg, e.g., "10"
beta=$7 #TODO: input arg, e.g., "0.5"

qrels=data/msmarco-pass/qrels.dev.small.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return

prf_tag=docs${fb_docs}.clusters${fb_clusters}.k${fb_k}.beta${beta}
prf_exp_root=${exp_root}/ColBERT-PRF-Ranking/${prf_tag}
ranking=${prf_exp_root}/MSMARCO-psg/label.py/$(ls ${prf_exp_root}/MSMARCO-psg/label.py/ | grep 20*)/ranking.tsv
if [ -f ${ranking} ];then
    python -m utility.evaluate.msmarco_passages --qrels ${qrels} --ranking ${ranking}
    echo;echo;echo
    return
fi

echo;echo;echo
# sanity check
[ ! -d "${exp_root}/" ] && echo "${exp_root} does not exist." && return
checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
index_root=${exp_root}/MSMARCO-psg/index.py
[ ! -d "${index_root}/" ] && echo "${index_root} does not exist." && return
queries=data/msmarco-pass/queries.dev.small.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/msmarco-pass/qrels.dev.small.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
collection=data/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return

# ANN search
topk=${exp_root}/MSMARCO-psg/retrieve.py/$(ls ${exp_root}/MSMARCO-psg/retrieve.py)/unordered.tsv
if [ ! -f "${topk}" ];then
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
fb_ranking=${exp_root}/MSMARCO-psg/rerank.py/$(ls ${exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
if [ ! -f "${fb_ranking}" ];then
    echo "Exact-NN Search using rerank.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    echo "topk      : ${topk}"
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.rerank --topk ${topk} --batch --log-scores --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${exp_root} --experiment MSMARCO-psg
else
    echo "We have Exact-NN search result at: \"${fb_ranking}\""
fi
echo;echo;echo

# 0-2. Preprocessing: Prepare tokenids for ColBERT-PRF
echo;echo;echo
index_path=${index_root}/MSMARCO.L2.32x200k
tokenids_path=experiments/metadata.tokenids
if [ ! -f "${tokenids_path}/tokenids.docfreq" ]; then
    echo "Preprocessing: Prepare tokenids for ColBERT-PRF"
    python -m preprocessing.pseudo_labeling.prepare_tokenids_for_colbert_prf --collection ${collection} \
    --mask-punctuation --doc_maxlen 180 --index_path ${index_path} --output_path ${tokenids_path}
    cp --verbose ${index_path}/tokenids.docfreq ${tokenids_path}/tokenids.docfreq
    echo;echo;echo
fi
if [ ! -f "${index_path}/tokenids.docfreq" ]; then
    echo "Copy tokenids from the previous index (tokenids are same for all retrievers):"
    cp --verbose ${tokenids_path}/tokenids.docfreq ${index_path}/
    cp --verbose ${tokenids_path}/*.tokenids ${index_path}/
    echo;echo;echo
fi

# Exact-NN search using PRF
topk=${exp_root}/MSMARCO-psg/retrieve.py/$(ls ${exp_root}/MSMARCO-psg/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
fb_ranking=${exp_root}/MSMARCO-psg/rerank.py/$(ls ${exp_root}/MSMARCO-psg/rerank.py)/ranking.tsv
[ ! -f "${fb_ranking}" ] && echo "${fb_ranking} does not exist." && return
prf_tag=docs${fb_docs}.clusters${fb_clusters}.k${fb_k}.beta${beta}
prf_exp_root=${exp_root}/ColBERT-PRF-Ranking/${prf_tag}
ranking=${prf_exp_root}/MSMARCO-psg/label.py/$(ls ${prf_exp_root}/MSMARCO-psg/label.py/ | grep 20*)/ranking.tsv
if [ ! -f ${ranking} ];then
    echo "Exact-NN Search using PRF / label.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    echo "topk      : ${topk}"
    echo "fb_ranking: ${fb_ranking}"
    echo "prf_tag   : ${prf_tag}"
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.label \
    \
    --prf --fb_ranking ${fb_ranking} --topk ${topk} \
    --fb_docs ${fb_docs} --fb_clusters ${fb_clusters} --fb_k ${fb_k} --beta ${beta} \
    --depth 1000 \
    \
    --batch --log-scores --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --qrels ${qrels} --collection ${collection} \
    --checkpoint ${checkpoint} --root ${prf_exp_root} --experiment MSMARCO-psg
else
    echo "We have Exact-NN search using PRF result at: \"${ranking}\""
fi
echo;echo;echo

# evaluate
ranking=${prf_exp_root}/MSMARCO-psg/label.py/$(ls ${prf_exp_root}/MSMARCO-psg/label.py/ | grep 20*)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m utility.evaluate.msmarco_passages --qrels ${qrels} --ranking ${ranking}
echo;echo;echo
