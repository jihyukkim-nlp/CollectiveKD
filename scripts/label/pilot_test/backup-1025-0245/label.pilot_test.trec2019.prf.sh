#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg

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
rerank_experiment=kmeans.prf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
rf_dir=experiments/pilot_test/trec2019/kmeans.k10.beta1.0.clusters10/label.py/$(ls experiments/pilot_test/trec2019/kmeans.k10.beta1.0.clusters10/label.py/)
fb_ranking=${rf_dir}/ranking.jsonl
expansion_pt=${rf_dir}/expansion.pt
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    \
    --prf --expansion_pt ${expansion_pt} \
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
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.48509411917614387, 'nDCG@10': 0.7580093307675281, 'RR(rel=2)@10': 0.8875968992248061, 'R(rel=2)@1000': 0.8186313217518287, 'nDCG@50': 0.690058487664221, 'nDCG@100': 0.6736547405126189, 'nDCG@200': 0.6746438941738635, 'nDCG@500': 0.7020247663221892, 'nDCG@1000': 0.7153258415765128, 'NumRet': 42962.0, 'num_q': 43.0}                      
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.49887998796146144, 'nDCG@10': 0.771203383971933, 'RR(rel=2)@10': 0.8875968992248061, 'R(rel=2)@1000': 0.8222543863892677, 'nDCG@50': 0.7032176645908726, 'nDCG@100': 0.6846113600143597, 'nDCG@200': 0.6847218426678843, 'nDCG@500': 0.7112681865664183, 'nDCG@1000': 0.7237936654604654, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5121928895953506, 'nDCG@10': 0.7754236875824102, 'RR(rel=2)@10': 0.9038759689922481, 'R(rel=2)@1000': 0.8206178903622142, 'nDCG@50': 0.706131180412931, 'nDCG@100': 0.6897943343722951, 'nDCG@200': 0.6897200974769683, 'nDCG@500': 0.7167891034567134, 'nDCG@1000': 0.727043604958444, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 5, beta: 0.25, fb_clusters: 24
#TODO
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5104360653933622, 'nDCG@10': 0.7893268474416049, 'RR(rel=2)@10': 0.9069767441860467, 'R(rel=2)@1000': 0.8224444827752049, 'nDCG@50': 0.7077795390087505, 'nDC$@100': 0.6898272563013605, 'nDCG@200': 0.6898887444777633, 'nDCG@500': 0.7170441374355436, 'nDCG@1000': 0.7300235824402256, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5235552314613287, 'nDCG@10': 0.8051365995220359, 'RR(rel=2)@10': 0.9069767441860467, 'R(rel=2)@1000': 0.8258450768823048, 'nDCG@50': 0.721157705434244, 'nDCG@100': 0.7009898788592565, 'nDCG@200': 0.7000421265886307, 'nDCG@500': 0.7262878107776798, 'nDCG@1000': 0.7385079580361239, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5342445938821306, 'nDCG@10': 0.8056178152741984, 'RR(rel=2)@10': 0.934108527131783, 'R(rel=2)@1000': 0.8242508415622907, 'nDCG@50': 0.7250236114282272, 'nDCG@100': 0.7063826710028805, 'nDCG@200': 0.7062344840914158, 'nDCG@500': 0.732438496506403, 'nDCG@1000': 0.7424389086106278, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 5, beta: 0.25, fb_clusters: 24
#TODO
