#!/bin/bash
device=$1 # e.g., "0"
exp_root=$2 # e.g., "experiments/colbert.teacher"
checkpoint=$3 # e.g., "data/checkpoints/colbert.teacher.dnn" or "experiments/colbert.teacher/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-300000.dnn"
fb_ranking=$4 # nearest neighbor passages for each query, e.g., "experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl"
fb_docs=$5 # The number of feedback documents, e.g., "3"
fb_clusters=$6 # The number of clusters on feedback terms, e.g., "24"
fb_k=$7 # The number of feedback terms, e.g., "10"
beta=$8 # The weights for the feedback terms, e.g., "0.5" or "1.0"

echo;echo;echo "Input arguments"
echo "device            :${device}"
echo "exp_root          :${exp_root}"
echo "checkpoint        :${checkpoint}"
echo "fb_ranking        :${fb_ranking}"
echo "fb_docs           :${fb_docs}"
echo "fb_clusters       :${fb_clusters}"
echo "fb_k              :${fb_k}"
echo "beta              :${beta}"
echo;echo;echo

[ ! -d "${exp_root}" ] && echo "${exp_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${fb_ranking}" ] && echo "${fb_ranking} does not exist." && return

#! hard coded data paths
queries=data/msmarco-pass/queries.train.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/msmarco-pass/qrels.train.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
collection=data/msmarco-pass/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return
index_root=${exp_root}/MSMARCO-psg/index.py
[ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return

#! hard coded experiment path
collective_feedback_exp_root=${exp_root}/MSMARCO-psg-CollectiveFeedback/
mkdir -p ${collective_feedback_exp_root}
collective_feedback_experiment=docs${fb_docs}.clusters${fb_clusters}.k${fb_k}.beta${beta}


# Preprocessing: Prepare tokenids for documents in the collection
echo;echo;echo
index_path=${index_root}/MSMARCO.L2.32x200k
tokenids_path=experiments/metadata.tokenids
if [ ! -f "${tokenids_path}/0.tokenids" ]; then
    echo "Preprocessing: Prepare tokenids for documents in the collection"
    python -m preprocessing.prepare_tokenids_for_colbert_prf --collection ${collection} \
    --mask-punctuation --doc_maxlen 180 --index_path ${index_path} --output_path ${tokenids_path}
    echo;echo;echo
fi
if [ ! -f "${index_path}/0.tokenids" ]; then
    echo "Copy tokenids from the previous index (tokenids are same for all retrievers):"
    cp --verbose ${tokenids_path}/*.tokenids ${index_path}/
    if [ -f "${tokenids_path}/tokenids.docfreq" ]; then
        cp --verbose ${tokenids_path}/tokenids.docfreq ${index_path}/tokenids.docfreq
    fi
    echo;echo;echo
fi


CUDA_VISIBLE_DEVICES=${device} python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--root ${collective_feedback_exp_root} --experiment ${collective_feedback_experiment} \
\
--expansion_only \
--prf \
--fb_ranking ${fb_ranking} --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
\
--index_root ${index_root} --index_name MSMARCO.L2.32x200k  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
--batch --log-scores \
--queries ${queries} --checkpoint ${checkpoint} --qrels ${qrels} --collection ${collection}


# Copy generated ``tokenids.docfreq`` to ``experiments/metadata.tokenids/tokenids.docfreq``
cp --verbose ${index_path}/tokenids.docfreq ${tokenids_path}/tokenids.docfreq
# Remove ``tokenids.docfreq`` and ``*.tokenids``
rm -v ${index_path}/*.tokenids
rm -v ${index_path}/tokenids.docfreq