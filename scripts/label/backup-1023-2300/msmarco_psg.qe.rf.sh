#!/bin/bash
# Retrieve Unlabeled Positives using Expanded queries


device=$1 #TODO: input arg
fb_k=$2 #TODO: input arg
beta=$3 #TODO: input arg
fb_clusters=$4 #TODO: input arg


# 0. Remove queries that are not included in qrels
org_queries=data/queries.train.tsv #TODO: custom path
qrels=data/qrels.train.tsv #TODO: custom path
queries=data/queries.train.reduced.tsv #TODO: custom path
[ ! -f "${queries}" ] && python -m preprocessing.utils.reduce_train_queries --queries ${org_queries} --qrels ${qrels} --out ${queries}



# # 1. Split the large query file into small files, to prevent out-of-memory
# [ ! -f "${queries}" ] && echo "${queries} does not exist." && return
# queries_split=data/queries.train.reduced.splits #TODO: custom path
# if [ ! -d ${queries_split} ];then
#     echo "mkdir ${queries_split}"
#     mkdir -p ${queries_split}
# fi
# sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_split} # Check if queries are already splitted
# if [ ! $? -eq 0 ];then
#     echo "split \"${queries}\" into multiple queries with 100000 lines each, to prevent out-of-memory"
#     split -d -l 100000 ${queries} ${queries_split}/queries.tsv.
#     echo "$(ls ${queries_split})"
#     echo
# fi
# #?@ debugging: Sanity check
# sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_split}
# if [ ! $? -eq 0 ];then 
#     echo "# of queries in original and split files are different"
#     wc -l ${queries}
#     wc -l ${queries_split}/*
#     return
# fi


index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/colbert.teacher #TODO: custom path
expansion_experiment=MSMARCO-psg-train-kmeans.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -d "${experiment_root}" ] && echo "${experiment_root} does not exist." && return
CUDA_VISIBLE_DEVICES=${device} python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--root ${experiment_root} --experiment ${expansion_experiment} \
--expansion_only \
--fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
--batch --log-scores \
--queries ${queries} \
--checkpoint ${checkpoint} \
--qrels data/qrels.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv



# # 4. Exact-NN search, using expanded queries
# rerank_experiment=MSMARCO-psg-train-exp_embs10-exp_beta1.0 #TODO: custom path
# for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
#     small_queries=${queries_split}/queries.tsv.${i}
#     small_topk=${topk_split}/unordered.${i}.tsv
#     [ ! -f "${small_queries}" ] && echo "${small_queries} does not exist." && return
#     [ ! -f "${small_topk}" ] && echo "${small_topk} does not exist." && return
#     [ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
#     [ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
#     [ ! -d "${experiment_root}" ] && echo "${experiment_root} does not exist." && return
#     CUDA_VISIBLE_DEVICES=0 python \
#     -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
#     --topk ${small_topk} --batch --log-scores \
#     --queries ${small_queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
#     --qrels data/qrels.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
#     --root ${experiment_root} --experiment ${rerank_experiment} --exp_embs 10 --exp_beta 1.0 --depth 1000
# done
# #?@ 4-1. sanity check
# n_ranking_json=0
# for timelog in $(ls experiments/colbert.teacher/${rerank_experiment}/label.py | grep 2021-);do
#     echo
#     echo "timelog: "$timelog
#     # 
#     small_ranking_json=experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/ranking.jsonl
#     echo "small_ranking_json: "${small_ranking_json}
#     n_lines=$(wc -l ${small_ranking_json} | awk -F ' ' '{print $1}')
#     n_ranking_json=$(expr ${n_ranking_json} + ${n_lines})
# done
# n_queries=$(wc -l data/queries.train.reduced.tsv | awk -F ' ' '{print $1}')
# if [ ! ${n_ranking_json} -eq ${n_queries} ];then
#     echo "# of lines in queries (${n_queries}) and ranking (${n_ranking_json}) are different"
#     return
# fi



# # 5. Merge results
# ranking=experiments/colbert.teacher/${rerank_experiment}/label.py/ranking.tsv
# ranking_jsonl=experiments/colbert.teacher/${rerank_experiment}/label.py/ranking.jsonl
# echo -n "" > ${ranking}
# for timelog in $(ls experiments/colbert.teacher/${rerank_experiment}/label.py | grep 2021-);do
#     small_ranking=experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/ranking.tsv
#     cat ${small_ranking} >> ${ranking}
#     small_ranking_jsonl=experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/ranking.jsonl
#     cat ${small_ranking_jsonl} >> ${ranking_jsonl}
# done
# # 5-1. sanity check
# sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${ranking_jsonl}
# if [ ! $? -eq 0 ];then 
#     echo "# of lines in queries file and ranking file are different"
#     wc -l ${queries}
#     wc -l ${ranking_jsonl}
#     return
# fi
# # 5-2. Merge expansion.pt results
# pts=""
# for timelog in $(ls experiments/colbert.teacher/${rerank_experiment}/label.py | grep 2021-);do
#     pts=${pts}" "experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/expansion.pt
# done
# echo 
# python -m preprocessing.pseudo_labeling.merge_expansion_result --pts ${pts} --output experiments/colbert.teacher/${rerank_experiment}/label.py/expansion.pt
# #?@ debugging
# # # 5-2. delete splited file results
# # for timelog in $(ls experiments/colbert.teacher/${experiment}/label.py | grep 2021-);do
# #     jsonfile=$(ls experiments/colbert.teacher/${experiment}/label.py/${timelog} | grep jsonl)
# #     small_ranking=experiments/colbert.teacher/${experiment}/label.py/${timelog}/${jsonfile}
# #     rm -v ${small_ranking}
# # done
