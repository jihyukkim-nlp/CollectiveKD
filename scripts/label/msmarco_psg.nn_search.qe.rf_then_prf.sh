#!/bin/bash
# Retrieve Unlabeled Positives using Expanded queries from RF (relevance feedback)

device=$1 #TODO: input arg
# 
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg
# 
rf_fb_k=$6 #TODO: input arg
rf_fb_clusters=$7 #TODO: input arg
rf_beta=$8 #TODO: input arg



# 0. Remove queries that are not included in qrels
org_queries=data/queries.train.tsv #TODO: custom path
qrels=data/qrels.train.tsv #TODO: custom path
queries=data/queries.train.reduced.tsv #TODO: custom path
[ ! -f "${queries}" ] && python -m preprocessing.utils.reduce_train_queries --queries ${org_queries} --qrels ${qrels} --out ${queries}



echo;echo;echo
# 1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
echo "1. ANN search (FAISS), using original queries without query expansion"
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/colbert.teacher #TODO: custom path
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return
experiment=MSMARCO-psg-train #TODO: custom path
# 
topk_dir=${experiment_root}/${experiment}/retrieve.py/$(ls ${experiment_root}/${experiment}/retrieve.py)
topk=${topk_dir}/unordered.tsv
if [ ! -f ${topk} ];then
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${experiment_root} --experiment ${experiment}
else
    echo "We have ANN search result at: \"${topk}\""
fi



echo;echo;echo
# 2. Split the large query file into small files, to prevent out-of-memory
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
queries_split=data/queries.train.reduced.splits #TODO: custom path
if [ ! -d ${queries_split} ];then
    do_split=1
else
    sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_split}
    if [ ! $? -eq 0 ];then
        do_split=1
    else
        do_split=0
    fi
fi
if [ ${do_split} -eq 1 ];then
    echo "Split the large query file into small files, to prevent out-of-memory"
    echo "mkdir ${queries_split}"
    mkdir -p ${queries_split}
    echo "split \"${queries}\" into multiple queries with 100000 lines each"
    # split -d -l 100000 ${queries} ${queries_split}/queries.tsv.
    split -d -l 50000 ${queries} ${queries_split}/queries.tsv.
fi
echo "Splitted query files"
wc -l ${queries_split}/*
n_splits=$(ls ${queries_split} | wc -l)
echo
# sanity check: make sure the original query file and split queries have the same number of lines
sh scripts/utils/sanity_check/equal_n_lines.sh ${queries} ${queries_split}
if [ ! $? -eq 0 ];then 
    echo "# of queries in original and split files are different"
    wc -l ${queries}
    wc -l ${queries_split}/*
    return
fi



echo;echo;echo
# 3. Filter ANN search result (top-K pids in ``unordered.tsv``), using each split queries
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
topk_split=${topk_dir}/queries.train.reduced.splits #TODO: custom path
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
    echo "Split the large unordered.tsv file into small files, to prevent out-of-memory"
    echo "mkdir ${topk_split}"
    mkdir -p ${topk_split}
    # for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    #     small_queries=${queries_split}/queries.tsv.${i}
    #     #TODO: all at once
    #     python -m preprocessing.utils.filter_topK_pids \
    #     --queries ${small_queries} \
    #     --topk ${topk} --filtered_topk ${topk_split}/unordered.${i}.tsv
    # done
    small_queries=""
    filtered_topk=""
    for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
        small_queries="${small_queries} ${queries_split}/queries.tsv.${i}"
        filtered_topk="${filtered_topk} ${topk_split}/unordered.${i}.tsv"
    done
    
    python -m preprocessing.utils.filter_topK_pids --topk ${topk} \
    --queries ${small_queries} \
    --filtered_topk ${filtered_topk}
fi



# #TODO: ###########################################################################
# #?@ debugging
# return
# #?@ debugging
# #TODO: ###########################################################################



echo;echo;echo
# 4. Exact-NN search, using expanded queries
echo "4. Exact-NN search, using expanded queries"
# 
# rerank_experiment=MSMARCO-psg-train-exp_embs10-exp_beta1.0 # previous version
# rerank_experiment=MSMARCO-psg-train-kmeans.rf.k${fb_k}.beta${beta}.clusters${fb_clusters} # rf
rerank_experiment=kmeans.rf_then_prf.rf-k${rf_fb_k}.beta${rf_beta}.clusters${rf_fb_clusters}.prf-docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
# 
expansion_pt=experiments/colbert.teacher/${rerank_experiment}/label.py/expansion.pt
ranking=experiments/colbert.teacher/${rerank_experiment}/label.py/ranking.tsv
ranking_jsonl=experiments/colbert.teacher/${rerank_experiment}/label.py/ranking.jsonl
if [ -f ${expansion_pt} ] && [ -f ${ranking} ] && [ -f ${ranking_jsonl} ];then
    echo "We have Exact-NN search results:"
    echo "      $(du -hs ${expansion_pt})"
    echo "      $(du -hs ${ranking})"
    echo "      $(du -hs ${ranking_jsonl})"
    # echo "      $(wc -l ${ranking})"
    # echo "      $(wc -l ${ranking_jsonl})"
    return
fi
# 
echo "NN-Search start ... "
# 
rf_exp=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k${rf_fb_k}.beta${rf_beta}.clusters${rf_fb_clusters}
if [ ! -d "${rf_exp}" ];then
    echo "${rf_exp} does not exist!" && return
    # echo "sh scripts/label/msmarco_psg.nn_search.rf.sh ${device} ${rf_fb_k} ${rf_beta} ${rf_fb_clusters}"
    # sh scripts/label/msmarco_psg.nn_search.rf.sh ${device} ${rf_fb_k} ${rf_beta} ${rf_fb_clusters}
fi
[ ! -d "${rf_exp}" ] && echo "${rf_exp} does not exist." && return
rf_fb_ranking=${rf_exp}/label.py/ranking.jsonl
rf_expansion_pt=${rf_exp}/label.py/expansion.pt
[ ! -f "${rf_fb_ranking}" ] && echo "${rf_fb_ranking} does not exist." && return
[ ! -f "${rf_expansion_pt}" ] && echo "${rf_expansion_pt} does not exist." && return
# 
[ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -d "${experiment_root}" ] && echo "${experiment_root} does not exist." && return
for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    
    small_queries=${queries_split}/queries.tsv.${i}
    small_topk=${topk_split}/unordered.${i}.tsv
    [ ! -f "${small_queries}" ] && echo "${small_queries} does not exist." && return
    [ ! -f "${small_topk}" ] && echo "${small_topk} does not exist." && return

    # #TODO: ###########################################################################
    # #?@ debugging
    # return
    # #?@ debugging
    # #TODO: ###########################################################################
    
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${small_topk} --batch --log-scores \
    --queries ${small_queries} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --checkpoint ${checkpoint} \
    --qrels data/qrels.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    \
    --prf \
    --expansion_pt ${rf_expansion_pt} --fb_ranking ${rf_fb_ranking} \
    --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
    --depth 1000

done

# #TODO: ###########################################################################
# #?@ debugging
# return
# #?@ debugging
# #TODO: ###########################################################################

echo "sanity check: # of lines in small ranking results && total # of queries"
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
echo "5. Merge results"
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
# 5-2. Merge expansion.pt results
pts=""
for timelog in $(ls experiments/colbert.teacher/${rerank_experiment}/label.py | grep 2021-);do
    pts=${pts}" "experiments/colbert.teacher/${rerank_experiment}/label.py/${timelog}/expansion.pt
done
echo 
python -m preprocessing.pseudo_labeling.merge_expansion_result --pts ${pts} --output ${expansion_pt}
#?@ debugging
# # 5-2. delete splited file results
# for timelog in $(ls experiments/colbert.teacher/${experiment}/label.py | grep 2021-);do
#     jsonfile=$(ls experiments/colbert.teacher/${experiment}/label.py/${timelog} | grep jsonl)
#     small_ranking=experiments/colbert.teacher/${experiment}/label.py/${timelog}/${jsonfile}
#     rm -v ${small_ranking}
# done
