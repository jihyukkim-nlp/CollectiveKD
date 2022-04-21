#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg

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
rerank_experiment=kmeans.prf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
rf_dir=experiments/pilot_test/trec2020/kmeans.k10.beta1.0.clusters10/label.py/$(ls experiments/pilot_test/trec2020/kmeans.k10.beta1.0.clusters10/label.py/)
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
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    \
    --prf --expansion_pt ${expansion_pt} \
    --fb_ranking ${fb_ranking} \
    --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
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
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.43219405591282756, 'nDCG@10': 0.6859934163708604, 'RR(rel=2)@10': 0.8606995884773662, 'R(rel=2)@1000': 0.8394907608167946, 'nDCG@50': 0.6050382899253393, 'nDCG@100': 0.6083468204333825, 'nDCG@200': 0.6355675415363252, 'nDCG@500': 0.6646960071468345, 'nDCG@1000': 0.6769766325565011, 'NumRet': 53951.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.4472354698336113, 'nDCG@10': 0.6991292568279918, 'RR(rel=2)@10': 0.8699588477366255, 'R(rel=2)@1000': 0.839381828354921, 'nDCG@50': 0.6226034674952573, 'nDCG@100': 0.61883772507595, 'nDCG@200': 0.6453612108482623, 'nDCG@500': 0.6736989090104563, 'nDCG@1000': 0.6843531801653853, 'NumRet': 53951.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.4523300491879823, 'nDCG@10': 0.7063153016541008, 'RR(rel=2)@10': 0.8819444444444444, 'R(rel=2)@1000': 0.8407319771781436, 'nDCG@50': 0.6255546730344469, 'nDCG@100': 0.6236211673953228, 'nDCG@200': 0.6490701606375993, 'nDCG@500': 0.6758919891911056, 'nDCG@1000': 0.6879842314581895, 'NumRet': 53951.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 5, beta: 0.25, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.46357296956850325, 'nDCG@10': 0.7186382229000644, 'RR(rel=2)@10': 0.8927469135802468, 'R(rel=2)@1000': 0.8407319771781436, 'nDCG@50': 0.6361994903195558, 'nDCG@100': 0.6318251362215064, 'nDCG@200': 0.6552314451201245, 'nDCG@500': 0.6817057896303197, 'nDCG@1000': 0.6931232431235458, 'NumRet': 53951.0, 'num_q': 54.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.4961045364875935, 'nDCG@10': 0.7543887248409713, 'RR(rel=2)@10': 0.9629629629629629, 'R(rel=2)@1000': 0.8427941038635426, 'nDCG@50': 0.6502533498622906, 'nDCG@100': 0.6496844176361385, 'nDCG@200': 0.6759448429340437, 'nDCG@500': 0.7042110523958082, 'nDCG@1000': 0.7161623238816373, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.513129821298323, 'nDCG@10': 0.7673314937392587, 'RR(rel=2)@10': 0.9907407407407407, 'R(rel=2)@1000': 0.8426959144758516, 'nDCG@50': 0.6702304068755809, 'nDCG@100': 0.6623768170098585, 'nDCG@200': 0.6877979759008519, 'nDCG@500': 0.715257644034329, 'nDCG@1000': 0.7256851330011936, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5164999893663792, 'nDCG@10': 0.7734580853588843, 'RR(rel=2)@10': 0.9907407407407407, 'R(rel=2)@1000': 0.8440266534273637, 'nDCG@50': 0.6720248692646, 'nDCG@100': 0.6659551220186806, 'nDCG@200': 0.6901467548460424, 'nDCG@500': 0.7163859424553869, 'nDCG@1000': 0.7280164916824716, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 5, beta: 0.25, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5259063456471531, 'nDCG@10': 0.7839033814190204, 'RR(rel=2)@10': 0.9907407407407407, 'R(rel=2)@1000': 0.8440266534273637, 'nDCG@50': 0.680600133283646, 'nDCG@100': 0.6722550755333653, 'nDCG@200': 0.694775584031614, 'nDCG@500': 0.7205232614284384, 'nDCG@1000': 0.7317438910037657, 'NumRet': 54000.0, 'num_q': 54.0}