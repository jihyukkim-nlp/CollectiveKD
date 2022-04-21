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
# 
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
#* {'nDCG@10': 0.6591721188838581, 'nDCG@25': 0.6264421274544607, 'nDCG@50': 0.6169454391385506, 'nDCG@100': 0.6192425491183937, 'nDCG@200': 0.6390633045917826, 'nDCG@500': 0.6647762769372917, 'nDCG@1000': 0.6773923877486366, 'R(rel=2)@3': 0.1721369010424624, 'R(rel=2)@5': 0.25133817767116956, 'R(rel=2)@10': 0.3817402959277569, 'R(rel=2)@25': 0.5280019698725538, 'R(rel=2)@50': 0.6453652260677059, 'R(rel=2)@100': 0.7233729843700203, 'R(rel=2)@200': 0.7728589156028156, 'R(rel=2)@1000': 0.8392102404811852, 'P(rel=2)@1': 0.7037037037037037, 'P(rel=2)@3': 0.6234567901234569, 'P(rel=2)@5': 0.5666666666666667, 'P(rel=2)@10': 0.48888888888888893, 'P(rel=2)@25': 0.3392592592592592, 'P(rel=2)@50': 0.23740740740740734, 'P(rel=2)@100': 0.14814814814814814, 'P(rel=2)@200': 0.08416666666666667, 'AP(rel=2)@1000': 0.44760906897749636, 'RR(rel=2)@10': 0.8068415637860082, 'NumRet': 53951.0, 'num_q': 54.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
#* {'nDCG@10': 0.6746415559920877, 'nDCG@25': 0.6408579361254387, 'nDCG@50': 0.6284356710611297, 'nDCG@100': 0.6296520709973429, 'nDCG@200': 0.6482406927534247, 'nDCG@500': 0.6738405847608164, 'nDCG@1000': 0.6862852044096319, 'R(rel=2)@3': 0.16003239577496967, 'R(rel=2)@5': 0.2476104839708578, 'R(rel=2)@10': 0.38692970913678476, 'R(rel=2)@25': 0.5430444561689066, 'R(rel=2)@50': 0.65296216952532, 'R(rel=2)@100': 0.7291550798979732, 'R(rel=2)@200': 0.7767769930691577, 'R(rel=2)@1000': 0.8425068725529834, 'P(rel=2)@1': 0.7037037037037037, 'P(rel=2)@3': 0.6543209876543209, 'P(rel=2)@5': 0.6111111111111113, 'P(rel=2)@10': 0.5296296296296296, 'P(rel=2)@25': 0.3651851851851853, 'P(rel=2)@50': 0.2522222222222222, 'P(rel=2)@100': 0.1562962962962963, 'P(rel=2)@200': 0.08833333333333335, 'AP(rel=2)@1000': 0.4637476631143032, 'RR(rel=2)@10': 0.8140432098765432, 'NumRet': 54000.0, 'num_q': 54.0}
