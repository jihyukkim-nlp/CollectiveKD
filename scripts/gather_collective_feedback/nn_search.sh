#!/bin/bash
# Retrieve Hard Negatives using original query
device=$1 # e.g., "0"
exp_root=$2 # e.g., "experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf"
checkpoint=$3 # e.g., "data/checkpoints/colbert.teacher.dnn" or "experiments/colbert.teacher/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-300000.dnn"

echo;echo;echo "Input arguments"
echo "device            :${device}"
echo "exp_root          :${exp_root}"
echo "checkpoint        :${checkpoint}"
echo;echo;echo

[ ! -d "${exp_root}" ] && echo "${exp_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return

#! hard coded data paths
queries=data/msmarco-pass/queries.train.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/msmarco-pass/qrels.train.tsv
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
collection=data/msmarco-pass/collection.tsv
[ ! -f "${collection}" ] && echo "${collection} does not exist." && return
index_root=${exp_root}/MSMARCO-psg/index.py
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return





echo;echo;echo
# 1. Split the large query file into small files, to prevent out-of-memory
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
chunk_size=50000
queries_chunks=data/msmarco-pass/queries.train.chunks${chunk_size} #! hard coded path
if [ ! -d ${queries_chunks} ];then
    do_split=1
else
    sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_chunks}
    if [ ! $? -eq 0 ];then
        do_split=1
    else
        do_split=0
    fi
fi
if [ ${do_split} -eq 1 ];then
    echo "1. Split the large query file into small files, to prevent out-of-memory"
    echo "mkdir ${queries_chunks}"
    mkdir -p ${queries_chunks}
    echo "split \"${queries}\" into multiple queries with 100000 lines each"
    split -d -l ${chunk_size} ${queries} ${queries_chunks}/queries.tsv.
fi
echo "Splitted query files"
wc -l ${queries_chunks}/*
n_splits=$(ls ${queries_chunks} | wc -l)
echo
# sanity check: make sure the original query file and split queries have the same number of lines
sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_chunks}
if [ ! $? -eq 0 ];then 
    echo "# of queries in original and split files are different"
    wc -l ${queries}
    wc -l ${queries_chunks}/*
    return
fi



echo;echo;echo
# 2. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
experiment=MSMARCO-psg-HN
# 
topk_dir=${exp_root}/${experiment}/retrieve.py/$(ls ${exp_root}/${experiment}/retrieve.py)
topk=${topk_dir}/unordered.tsv
if [ ! -f ${topk} ];then
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${exp_root} --experiment ${experiment}
    # 
    topk_dir=${exp_root}/${experiment}/retrieve.py/$(ls ${exp_root}/${experiment}/retrieve.py)
    topk=${topk_dir}/unordered.tsv
else
    echo "We have ANN search result at: \"${topk}\""
fi



echo;echo;echo
# 3. Filter ANN search result (top-K pids in ``unordered.tsv``), using each split queries
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
topk_split=${topk_dir}/queries.train.chunks${chunk_size} #! hard coded path
if [ ! -d ${topk_split} ];then
    do_filter=1
else
    _nonexist_split=0
    for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
        if [ ! -f ${topk_split}/unordered.${i}.tsv ];then
            _nonexist_split=$(expr ${_nonexist_split} + 1)
        fi
    done
    if [ ! ${_nonexist_split} -eq 0 ];then
        do_filter=1
    else
        do_filter=0
    fi
fi
if [ ${do_filter} -eq 1 ];then
    echo "3. Split the large unordered.tsv file into small files, to prevent out-of-memory"
    echo "mkdir ${topk_split}"
    mkdir -p ${topk_split}
    small_queries=""
    filtered_topk=""
    for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
        small_queries="${small_queries} ${queries_chunks}/queries.tsv.${i}"
        filtered_topk="${filtered_topk} ${topk_split}/unordered.${i}.tsv"
    done
    python -m preprocessing.utils.filter_topK_pids --topk ${topk} \
    --queries ${small_queries} \
    --filtered_topk ${filtered_topk}
fi



echo;echo;echo
# 4. Exact-NN search, using the original query
echo "4. Exact-NN search, using expanded queries"
# 
ranking=${exp_root}/${experiment}/label.py/ranking.tsv
ranking_jsonl=${exp_root}/${experiment}/label.py/ranking.jsonl
if [ -f ${ranking} ] && [ -f ${ranking_jsonl} ];then
    echo "We have Exact-NN search results:"
    echo "      $(du -hs ${ranking})"
    echo "      $(du -hs ${ranking_jsonl})"
    # echo "      $(wc -l ${ranking})"
    # echo "      $(wc -l ${ranking_jsonl})"
    return
fi
# 
for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    small_queries=${queries_chunks}/queries.tsv.${i}
    small_topk=${topk_split}/unordered.${i}.tsv
    [ ! -f "${small_queries}" ] && echo "${small_queries} does not exist." && return
    [ ! -f "${small_topk}" ] && echo "${small_topk} does not exist." && return
    [ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
    [ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
    [ ! -d "${exp_root}" ] && echo "${exp_root} does not exist." && return

    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${small_topk} --batch --log-scores \
    --queries ${small_queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    --qrels ${qrels} --collection ${collection} \
    --root ${exp_root} --experiment ${experiment} --fb_k 0 --beta 0.0 --depth 200 \
    --score_by_range
done
echo "sanity check: # of lines in small ranking results && total # of queries"
n_ranking_json=0
for timelog in $(ls ${exp_root}/${experiment}/label.py | grep 2022-);do
    echo
    echo "timelog: "$timelog
    # 
    small_ranking_json=${exp_root}/${experiment}/label.py/${timelog}/ranking.jsonl
    echo "small_ranking_json: "${small_ranking_json}
    n_lines=$(wc -l ${small_ranking_json} | awk -F ' ' '{print $1}')
    n_ranking_json=$(expr ${n_ranking_json} + ${n_lines})
done
n_queries=$(wc -l ${queries} | awk -F ' ' '{print $1}')
if [ ! ${n_ranking_json} -eq ${n_queries} ];then
    echo "# of lines in queries (${n_queries}) and ranking (${n_ranking_json}) are different"
    return
fi


echo;echo;echo
# 5. Merge results
echo "5. Merge results"
echo -n "" > ${ranking}
for timelog in $(ls ${exp_root}/${experiment}/label.py | grep 2022-);do
    small_ranking=${exp_root}/${experiment}/label.py/${timelog}/ranking.tsv
    cat ${small_ranking} >> ${ranking}
    small_ranking_jsonl=${exp_root}/${experiment}/label.py/${timelog}/ranking.jsonl
    cat ${small_ranking_jsonl} >> ${ranking_jsonl}
done
# 5-1. sanity check
echo "sanity check: # of lines in queries file and ranking file"
sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${ranking_jsonl}
if [ ! $? -eq 0 ];then 
    echo "# of lines in queries file and ranking file are different"
    wc -l ${queries}
    wc -l ${ranking_jsonl}
    return
else
    experiment=MSMARCO-psg-HN
    # 5-2. delete splited file results
    for timelog in $(ls ${exp_root}/${experiment}/label.py | grep 2022-);do
        small_ranking=${exp_root}/${experiment}/label.py/${timelog}/ranking.*
        rm -v ${small_ranking}
    done
    # 5-3. delete unordered.tsv
    topk=${topk_dir}/unordered.tsv
    rm -v ${topk}
    # 5-4. delete queries split
    rm -r -v ${queries_chunks}
    rm -r -v ${topk_split}
fi