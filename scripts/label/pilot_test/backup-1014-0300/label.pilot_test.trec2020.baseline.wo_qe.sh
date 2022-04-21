#!/bin/bash
device=$1 #TODO: input arg

# 1. ANN Search
queries=data/queries.trec2020.tsv #TODO: custom path
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/pilot_test/trec2020 #TODO: custom path
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return
ann_experiment=wo_qe #TODO: custom path
topk=${experiment_root}/${ann_experiment}/retrieve.py/$(ls ${experiment_root}/${ann_experiment}/retrieve.py)/unordered.tsv
if [ ! -f ${topk} ];then
    echo "1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small."
    # 1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${experiment_root} --experiment ${ann_experiment}
else
    echo "We have ANN search result at: \"${topk}\""
fi



# 2. Exact-NN Search, using query expansion.
topk=${experiment_root}/${ann_experiment}/retrieve.py/$(ls ${experiment_root}/${ann_experiment}/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
rerank_experiment=wo_qe #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --exp_embs 0 --exp_beta 0.0 --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --ranking ${ranking} > ${result_path}
# python -m utility.evaluate.trec_passages --qrels data/trec2020/2020qrels-pass.txt --ranking ${ranking} > ${result_path} #?@ debugging
cat ${result_path} | tail -1
# 3-2. After filtering
qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
for thr in 0 26 28 29 30;do
    python -m pilot_test.label.evaluate_labels \
    --k 9999 --thr ${thr} --ranking ${ranking} --qrels ${qrels} --qrels_exclude ${qrels_exclude}
    echo
done
# data/trec2020/2020qrels-pass.txt
# {'ndcg_cut_10': 0.6746415559920877, 'ndcg_cut_200': 0.6482406927534247, 'map_cut_1000': 0.4645343330170458, 'recall_100': 0.5772633316464172, 'recall_200': 0.6417267537573986, 'recall_500': 0.7063759957941926, 'recall_1000': 0.7416689882254505}
# data/pilot_test/label/2020qrels-pass.test.tsv
# {'ndcg_cut_10': 0.6251265897711206, 'ndcg_cut_200': 0.6198468752918634, 'map_cut_1000': 0.42696921265247495, 'recall_100': 0.5710588061000155, 'recall_200': 0.6365763090383149, 'recall_500': 0.7014673654426932, 'recall_1000': 0.7374973327690123}
# === k 9999, thr 0.0 ===
#          ranking       = experiments/pilot_test/trec2020/wo_qe/label.py/2021-10-08_11.31.27/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 53951 (999.09 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.041, recall 0.629, F1 0.078

# === k 9999, thr 26.0 ===
#          ranking       = experiments/pilot_test/trec2020/wo_qe/label.py/2021-10-08_11.31.27/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 1041 (19.28 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.478, recall 0.140, F1 0.217

# === k 9999, thr 28.0 ===
#          ranking       = experiments/pilot_test/trec2020/wo_qe/label.py/2021-10-08_11.31.27/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 170 (3.15 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.629, recall 0.030, F1 0.057

# === k 9999, thr 29.0 ===
#          ranking       = experiments/pilot_test/trec2020/wo_qe/label.py/2021-10-08_11.31.27/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 42 (0.78 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.881, recall 0.010, F1 0.021

# === k 9999, thr 30.0 ===
#          ranking       = experiments/pilot_test/trec2020/wo_qe/label.py/2021-10-08_11.31.27/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 5 (0.09 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 1.000, recall 0.001, F1 0.003

