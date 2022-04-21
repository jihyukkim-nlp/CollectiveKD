#!/bin/bash

# Retrieve Hard Negatives using original query

# 0. Remove queries that are not included in qrels
org_queries=data/queries.train.tsv #TODO: custom path
qrels=data/qrels.train.tsv #TODO: custom path
queries=data/queries.train.reduced.tsv #TODO: custom path
# python -m preprocessing.utils.reduce_train_queries --queries ${org_queries} --qrels ${qrels} --out ${queries}



# 1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/colbert.teacher #TODO: custom path
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return
experiment=MSMARCO-psg-train #TODO: custom path
# CUDA_VISIBLE_DEVICES=1 python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
# --queries ${queries} \
# --nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
# --checkpoint ${checkpoint} --root ${experiment_root} --experiment ${experiment}



# 2. Split the large query file into small files, to prevent out-of-memory
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
echo "Split the large query file into small files, to prevent out-of-memory"
queries_split=data/queries.train.reduced.splits #TODO: custom path
# echo "mkdir ${queries_split}"
# mkdir -p ${queries_split}
# echo "split \"${queries}\" into multiple queries with 100000 lines each"
# split -d -l 100000 ${queries} ${queries_split}/queries.tsv.
echo "$(ls ${queries_split})"
echo
#?@ debugging: Sanity check
sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_split}
if [ ! $? -eq 0 ];then 
    echo "# of queries in original and split files are different"
    wc -l ${queries}
    wc -l ${queries_split}/*
    return
fi




# 3. Filter ANN search result (top-K pids in ``unordered.tsv``), using each split queries
topk_dir=${experiment_root}/${experiment}/retrieve.py/$(ls ${experiment_root}/${experiment}/retrieve.py)
topk=${topk_dir}/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
# echo "Split the large unordered.tsv file into small files, to prevent out-of-memory"
topk_split=${topk_dir}/queries.train.reduced.splits #TODO: custom path
# echo "mkdir ${topk_split}"
# mkdir -p ${topk_split}
n_splits=$(ls ${queries_split} | wc -l)
# for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
#     small_queries=${queries_split}/queries.tsv.${i}
#     #TODO: all at once
#     python -m preprocessing.utils.filter_topK_pids --queries ${small_queries} --topk ${topk} \
#     --filtered_topk ${topk_split}/unordered.${i}.tsv
# done



# 4. Exact-NN search, using the original query
rerank_experiment=MSMARCO-psg-train-exp_embs0-exp_beta0 #TODO: custom path
for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    small_queries=${queries_split}/queries.tsv.${i}
    small_topk=${topk_split}/unordered.${i}.tsv
    [ ! -f "${small_queries}" ] && echo "${small_queries} does not exist." && return
    [ ! -f "${small_topk}" ] && echo "${small_topk} does not exist." && return
    [ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
    [ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
    [ ! -d "${experiment_root}" ] && echo "${experiment_root} does not exist." && return
    CUDA_VISIBLE_DEVICES=1 python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${small_topk} --batch --log-scores \
    --queries ${small_queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    --qrels data/qrels.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --root ${experiment_root} --experiment ${rerank_experiment} --exp_embs 0 --exp_beta 0.0 --depth 1000
done
#?@ 4-1. sanity check
n_ranking_json=0
for timelog in $(ls experiments/colbert.teacher/${rerank_experiment}/label.py | grep 2021-);do
    echo
    echo "timelog: "$timelog
    # 
    small_ranking_json=experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/ranking.jsonl
    echo "small_ranking_json: "${small_ranking_json}
    n_lines=$(wc -l ${small_ranking_json} | awk -F ' ' '{print $1}')
    n_ranking_json=$(expr ${n_ranking_json} + ${n_lines})
done
n_queries=$(wc -l data/queries.train.reduced.tsv | awk -F ' ' '{print $1}')
if [ ! ${n_ranking_json} -eq ${n_queries} ];then
    echo "# of lines in queries (${n_queries}) and ranking (${n_ranking_json}) are different"
    return
fi



# 5. Merge results
ranking=experiments/colbert.teacher/${rerank_experiment}/label.py/ranking.tsv
ranking_jsonl=experiments/colbert.teacher/${rerank_experiment}/label.py/ranking.jsonl
echo -n "" > ${ranking}
for timelog in $(ls experiments/colbert.teacher/${rerank_experiment}/label.py | grep 2021-);do
    small_ranking=experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/ranking.tsv
    cat ${small_ranking} >> ${ranking}
    small_ranking_jsonl=experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/ranking.jsonl
    cat ${small_ranking_jsonl} >> ${ranking_jsonl}
done
# 5-1. sanity check
sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${ranking_jsonl}
if [ ! $? -eq 0 ];then 
    echo "# of lines in queries file and ranking file are different"
    wc -l ${queries}
    wc -l ${ranking_jsonl}
    return
fi
#?@ debugging
# # 5-2. delete splited file results
# for timelog in $(ls experiments/colbert.teacher/${experiment}/label.py | grep 2021-);do
#     jsonfile=$(ls experiments/colbert.teacher/${experiment}/label.py/${timelog} | grep jsonl)
#     small_ranking=experiments/colbert.teacher/${experiment}/label.py/${timelog}/${jsonfile}
#     rm -v ${small_ranking}
# done
