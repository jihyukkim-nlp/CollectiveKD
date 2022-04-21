#!/bin/bash
# Retrieve Unlabeled Positives using Expanded queries


device=$1 #TODO: input arg
# 
fb_k=$2 #TODO: input arg
beta=$3 #TODO: input arg
fb_clusters=$4 #TODO: input arg
fb_docs=$5 #TODO: input arg
# 
# fb_k=10 #TODO: custom arg
# beta=0.5 #TODO: custom arg
# fb_clusters=24 #TODO: custom arg
# fb_docs=3 #TODO: custom arg
# 
exp_root=$6 # e.g., "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k"
step=$7 # e.g., "50000" for "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-50000.dnn"

collection=/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv #TODO: custom path
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return

[ ! -d ${exp_root} ] && echo "${exp_root} does not exist" && return

checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -f ${checkpoint} ] && echo "${checkpoint} does not exist" && return

index_root=${exp_root}/MSMARCO-psg/index.py # e.g., "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/index.py"
[ ! -d ${index_root} ] && echo "${index_root} does not exist" && return

fb_ranking=${exp_root}/MSMARCO-psg-train/label.py/ranking.jsonl
[ ! -f ${fb_ranking} ] && echo "${fb_ranking} does not exist." && return



# 0. Remove queries that are not included in qrels
org_queries=data/queries.train.tsv #TODO: custom path
qrels=data/qrels.train.tsv #TODO: custom path
queries=data/queries.train.reduced.tsv #TODO: custom path
[ ! -f "${queries}" ] && python -m preprocessing.utils.reduce_train_queries --queries ${org_queries} --qrels ${qrels} --out ${queries}
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return



expansion_experiment=MSMARCO-psg-train-prf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
# resulting path for expansion.pt: 
#   ${exp_root}/${expansion_experiment}/label.py/*/expansion.pt
#   e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
[ -d ${exp_root}/${expansion_experiment}/label.py/ ] && echo "We already have \n\n\t${exp_root}/${expansion_experiment}/label.py\n" && return



echo;echo;echo
# 0. Preprocessing: prepare tokenids for ColBERT-PRF
# check if 
# index_path=${index_root}/MSMARCO.L2.32x200k
prev_index_path=experiments/colbert.teacher/MSMARCO-psg/index.py/MSMARCO.L2.32x200k
new_index_path=${index_root}/MSMARCO.L2.32x200k
[ ! -d ${prev_index_path} ] && echo "${prev_index_path} does not exist" && return
if [ ! -f ${prev_index_path}/tokenids.docfreq ]; then
    echo "0. Preprocessing: prepare tokenids for ColBERT-PRF"
    python -m preprocessing.pseudo_labeling.prepare_tokenids_for_colbert_prf --collection ${collection} \
    --mask-punctuation --doc_maxlen 180 --index_path ${prev_index_path}
    echo;echo;echo
fi
echo "Copy tokenids from the previous index (tokenids are same for all retrievers):"
cp --verbose ${prev_index_path}/tokenids.docfreq ${new_index_path}/
cp --verbose ${prev_index_path}/*.tokenids ${new_index_path}/
echo;echo;echo




echo;echo;echo
# Expanding query (but not ranking documents using the expanded query)
echo "Expanding query (but not ranking documents using the expanded query)"
CUDA_VISIBLE_DEVICES=${device} python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--root ${exp_root} --experiment ${expansion_experiment} \
\
--expansion_only \
--prf \
--fb_ranking ${fb_ranking} --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
\
--index_root ${index_root} --index_name MSMARCO.L2.32x200k  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
--batch --log-scores \
--queries ${queries} \
--checkpoint ${checkpoint} \
--qrels data/qrels.train.tsv --collection ${collection}
echo;echo;echo
