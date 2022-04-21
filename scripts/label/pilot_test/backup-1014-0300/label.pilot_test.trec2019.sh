#!/bin/bash
device=$1 #TODO: input arg
exp_embs=$2 #TODO: input arg
exp_beta=$3 #TODO: input arg

# 1. ANN Search
queries=data/queries.trec2019.tsv #TODO: custom path
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/pilot_test/trec2019 #TODO: custom path
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
rerank_experiment=maxsim.exp_embs${exp_embs}.exp_beta${exp_beta} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --exp_embs ${exp_embs} --exp_beta ${exp_beta} \
    --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --ranking ${ranking} > ${result_path}
# python -m utility.evaluate.trec_passages --qrels data/trec2019/2019qrels-pass.txt --ranking ${ranking} > ${result_path} #?@ debugging
cat ${result_path} | tail -1
# 3-2. After filtering
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
qrels_exclude=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
for thr in 0 30 31 32 33 34 35 36;do
    python -m pilot_test.label.evaluate_labels \
    --k 9999 --thr ${thr} --ranking ${ranking} --qrels ${qrels} --qrels_exclude ${qrels_exclude}
    echo
done
# data/trec2019/2019qrels-pass.txt
# {'ndcg_cut_10': 0.7536132253598935, 'ndcg_cut_200': 0.7007095419215157, 'map_cut_1000': 0.5155817214834795, 'recall_100': 0.5643196130161435, 'recall_200': 0.6533570306727404, 'recall_500': 0.7271932753883089, 'recall_1000': 0.7538050907436037}
# data/pilot_test/label/2019qrels-pass.test.tsv
# {'ndcg_cut_10': 0.6805611088245922, 'ndcg_cut_200': 0.6622118552120205, 'map_cut_1000': 0.4735306587178119, 'recall_100': 0.5590749398368665, 'recall_200': 0.6492189368303629, 'recall_500': 0.7237083971913443, 'recall_1000': 0.7506573655887914}
# === k 9999, thr 0.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 42962 (999.12 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.065, recall 0.688, F1 0.119

# === k 9999, thr 30.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 4956 (115.26 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.355, recall 0.433, F1 0.390

# === k 9999, thr 31.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 3174 (73.81 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.474, recall 0.371, F1 0.416

# === k 9999, thr 32.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 2050 (47.67 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.617, recall 0.312, F1 0.414

# === k 9999, thr 33.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 1371 (31.88 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.744, recall 0.251, F1 0.376

# === k 9999, thr 34.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 915 (21.28 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.842, recall 0.190, F1 0.310

# === k 9999, thr 35.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 555 (12.91 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.895, recall 0.122, F1 0.215

# === k 9999, thr 36.0 ===
#          ranking       = experiments/pilot_test/trec2019/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-06_13.47.39/ranking.tsv
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv
#          # of queries = 43
#         #> The # of predicted unlabeled positives = 292 (6.79 per query)
#         #> The # of labeled gold positives = 4059 (94.40 per query)
#         #> precision 0.928, recall 0.067, F1 0.125