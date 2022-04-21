#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg

# 1. ANN Search
queries=data/queries.trec2020.tsv #TODO: custom path
index_root=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index.py #TODO: custom path
checkpoint=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn #TODO: custom path
experiment_root=experiments/pilot_test/kd_prf_2nd_iter/trec2020 #TODO: custom path
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
    --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${experiment_root} --experiment ${ann_experiment}
else
    echo "We have ANN search result at: \"${topk}\""
fi



# 2. Exact-NN Search, using query expansion.
topk=${experiment_root}/${ann_experiment}/retrieve.py/$(ls ${experiment_root}/${ann_experiment}/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
rerank_experiment=kmeans.prf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
fb_ranking=experiments/pilot_test/kd_prf_2nd_iter/trec2020/wo_qe/label.py/$(ls experiments/pilot_test/kd_prf_2nd_iter/trec2020/wo_qe/label.py/)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    \
    --prf \
    --fb_ranking ${fb_ranking} \
    --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta}  --fb_clusters ${fb_clusters} \
    --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
# 
# # Evaluation on ``2020qrels-pass.test.tsv``
# result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
# qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
# qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
# python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
# echo;tail -1 ${result_all_path};echo
# 
# Evaluation on ``data/trec2020/2020qrels-pass.txt``
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;tail -1 ${result_all_path};echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7490645632516797, 'nDCG@25': 0.6948051182464804, 'nDCG@50': 0.679704987170137, 'nDCG@100': 0.6794812764738369, 'nDCG@200': 0.7038348725311904, 'nDCG@500': 0.7344385714173035, 'nDCG@1000': 0.7494062456857071, 'R(rel=2)@3': 0.18264212758176496, 'R(rel=2)@5': 0.2786180479475292, 'R(rel=2)@10': 0.40534596559278907, 'R(rel=2)@25': 0.5580293438777234, 'R(rel=2)@50': 0.6654243801145435, 'R(rel=2)@100': 0.7484215244188818, 'R(rel=2)@200': 0.8049411429260779, 'R(rel=2)@1000': 0.8827833847929204, 'P(rel=2)@1': 0.7962962962962963, 'P(rel=2)@3': 0.7345679012345678, 'P(rel=2)@5': 0.7111111111111111, 'P(rel=2)@10': 0.585185185185185, 'P(rel=2)@25': 0.3822222222222221, 'P(rel=2)@50': 0.26666666666666666, 'P(rel=2)@100': 0.16629629629629633, 'P(rel=2)@200': 0.09592592592592594, 'AP(rel=2)@1000': 0.5150890361806915, 'RR(rel=2)@10': 0.8703703703703703, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* 
