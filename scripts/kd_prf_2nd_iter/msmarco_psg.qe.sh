#!/bin/bash
# Retrieve Unlabeled Positives using Expanded queries


device=$1 #TODO: input arg
# 
# fb_k=$2 #TODO: input arg
# beta=$3 #TODO: input arg
# fb_clusters=$4 #TODO: input arg
# fb_docs=$5 #TODO: input arg
# 
fb_k=10 #TODO: custom arg
beta=0.5 #TODO: custom arg
fb_clusters=24 #TODO: custom arg
fb_docs=3 #TODO: custom arg



# 0. Remove queries that are not included in qrels
org_queries=data/queries.train.tsv #TODO: custom path
qrels=data/qrels.train.tsv #TODO: custom path
queries=data/queries.train.reduced.tsv #TODO: custom path
[ ! -f "${queries}" ] && python -m preprocessing.utils.reduce_train_queries --queries ${org_queries} --qrels ${qrels} --out ${queries}



experiment_root=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5 #TODO: custom path
index_root=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index.py #TODO: custom path
checkpoint=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn #TODO: custom path
expansion_experiment=MSMARCO-psg-train-kmeans.prf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
fb_ranking=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/ranking.jsonl #TODO: custom path
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -d "${experiment_root}" ] && echo "${experiment_root} does not exist." && return
[ ! -f "${fb_ranking}" ] && echo "${fb_ranking} does not exist." && return

# #TODO: ###########################################################################
# #?@ debugging
# echo "(return) check this code: \t\"$(cd $(dirname $0) && pwd)/msmarco_psg.qe.sh\"" && return
# #?@ debugging
# #TODO: ###########################################################################

echo;echo;echo
# 0. Preprocessing: prepare tokenids for ColBERT-PRF
echo "0. Preprocessing: prepare tokenids for ColBERT-PRF"
python -m preprocessing.pseudo_labeling.prepare_tokenids_for_colbert_prf --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--mask-punctuation --doc_maxlen 180 --index_path ${index_root}/MSMARCO.L2.32x200k
echo;echo;echo


echo;echo;echo
# Expanding query (but not ranking documents using the expanded query)
echo "Expanding query (but not ranking documents using the expanded query)"
CUDA_VISIBLE_DEVICES=${device} python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--root ${experiment_root} --experiment ${expansion_experiment} \
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
echo;echo;echo
