#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg
# 
rf_fb_k=$6 #TODO: input arg
rf_fb_clusters=$7 #TODO: input arg
rf_beta=$8 #TODO: input arg

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
# 
rerank_experiment=kmeans.rf_then_prf.rf-k${rf_fb_k}.beta${rf_beta}.clusters${rf_fb_clusters}.prf-docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
# 
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
# 
rf_exp=experiments/pilot_test/trec2020/kmeans.k${rf_fb_k}.beta${rf_beta}.clusters${rf_fb_clusters}
if [ ! -d "${rf_exp}" ];then
    echo "sh scripts/label/pilot_test/label.pilot_test.trec2020.sh ${device} ${rf_fb_k} ${rf_beta} ${rf_fb_clusters}"
    sh scripts/label/pilot_test/label.pilot_test.trec2020.sh ${device} ${rf_fb_k} ${rf_beta} ${rf_fb_clusters}
fi
[ ! -d "${rf_exp}" ] && echo "${rf_exp} does not exist." && return
rf_dir=${rf_exp}/label.py/$(ls ${rf_exp}/label.py/)
[ ! -d "${rf_dir}" ] && echo "${rf_dir} does not exist." && return
fb_ranking=${rf_dir}/ranking.jsonl
expansion_pt=${rf_dir}/expansion.pt
# 
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
qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# RF (fb_k: 10, beta: 0.5, fb_clusters: 10), PRF (fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24) #! default
#* {'nDCG@10': 0.7356660574503837, 'nDCG@25': 0.676131106808902, 'nDCG@50': 0.6475227448709021, 'nDCG@100': 0.6463119505860642, 'nDCG@200': 0.6688545165904878, 'nDCG@500': 0.6925981234382093, 'nDCG@1000': 0.7038846934649885, 'R(rel=2)@3': 0.18985945497670217, 'R(rel=2)@5': 0.2939860448010372, 'R(rel=2)@10': 0.4033786485121773, 'R(rel=2)@25': 0.5423667641163198, 'R(rel=2)@50': 0.646889274311512, 'R(rel=2)@100': 0.7320367732391381, 'R(rel=2)@200': 0.7849036664502402, 'R(rel=2)@1000': 0.8434366959159805, 'P(rel=2)@1': 0.7962962962962963, 'P(rel=2)@3': 0.7654320987654322, 'P(rel=2)@5': 0.725925925925926, 'P(rel=2)@10': 0.5666666666666665, 'P(rel=2)@25': 0.3725925925925926, 'P(rel=2)@50': 0.24851851851851847, 'P(rel=2)@100': 0.15462962962962962, 'P(rel=2)@200': 0.0883333333333333, 'AP(rel=2)@1000': 0.4857298222048089, 'RR(rel=2)@10': 0.8794753086419753, 'NumRet': 53951.0, 'num_q': 54.0}
# RF (fb_k: 5, beta: 0.5, fb_clusters: 10), PRF (fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24)
#* 
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# RF (fb_k: 10, beta: 0.5, fb_clusters: 10), PRF (fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24) #! default
#* {'nDCG@10': 0.7898274823781638, 'nDCG@25': 0.7164801349126698, 'nDCG@50': 0.6826205187020633, 'nDCG@100': 0.6787947899652066, 'nDCG@200': 0.7002799004464836, 'nDCG@500': 0.723389095866824, 'nDCG@1000': 0.7344069529814049, 'R(rel=2)@3': 0.20696423618121085, 'R(rel=2)@5': 0.30107308903155544, 'R(rel=2)@10': 0.4224257883134148, 'R(rel=2)@25': 0.5566712597307992, 'R(rel=2)@50': 0.6584793076172717, 'R(rel=2)@100': 0.7422709841323475, 'R(rel=2)@200': 0.7927457934645674, 'R(rel=2)@1000': 0.8466728364449971, 'P(rel=2)@1': 0.9074074074074074, 'P(rel=2)@3': 0.845679012345679, 'P(rel=2)@5': 0.7999999999999999, 'P(rel=2)@10': 0.625925925925926, 'P(rel=2)@25': 0.4007407407407407, 'P(rel=2)@50': 0.2648148148148149, 'P(rel=2)@100': 0.16333333333333333, 'P(rel=2)@200': 0.09277777777777778, 'AP(rel=2)@1000': 0.5368221846361124, 'RR(rel=2)@10': 0.9506172839506173, 'NumRet': 54000.0, 'num_q': 54.0}
# RF (fb_k: 5, beta: 0.5, fb_clusters: 10), PRF (fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24)
#* 
