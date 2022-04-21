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
# 
# Prepend labeled positives, to construct new feedback documents with RF + PRF 
fb_ranking=experiments/pilot_test/trec2020/wo_qe/label.py/$(ls experiments/pilot_test/trec2020/wo_qe/label.py/)/rfprf${fb_docs}.fd.jsonl
if [ ! -f ${fb_ranking} ];then
    python -m preprocessing.feedback_documents.construct_fd_file \
    --fb_docs ${fb_docs} \
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv \
    --ranking experiments/pilot_test/trec2020/wo_qe/label.py/$(ls experiments/pilot_test/trec2020/wo_qe/label.py/)/ranking.tsv \
    --output ${fb_ranking}
fi
# 
rerank_experiment=kmeans.rfprf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
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
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5084638838663701, 'nDCG@10': 0.7218733299597491, 'RR(rel=2)@10': 0.8719135802469137, 'R(rel=2)@1000': 0.8448475220315624, 'R(rel=2)@10': 0.4061923342682816, 'R(rel=2)@100': 0.7411199790068601, 'P(rel=2)@10': 0.548148148148148, 'P(rel=2)@100': 0.15907407407407403, 'nDCG@50': 0.6664544247101598, 'nDCG@100': 0.6619775986776306, 'nDCG@200': 0.6817576442034071, 'nDCG@500': 0.7058424595052211, 'nDCG@1000': 0.7139312936804039, 'NumRet': 53951.0, 'num_q': 54.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5360313312006737, 'nDCG@10': 0.750368676118759, 'RR(rel=2)@10': 0.8796296296296297, 'R(rel=2)@1000': 0.8480569792707295, 'R(rel=2)@10': 0.41729599251041444, 'R(rel=2)@100': 0.7469284506447723, 'P(rel=2)@10': 0.5944444444444444, 'P(rel=2)@100': 0.16777777777777778, 'nDCG@50': 0.6818077314250749, 'nDCG@100': 0.6765620938646334, 'nDCG@200': 0.6961684250706548, 'nDCG@500': 0.7194497842124501, 'nDCG@1000': 0.7273045703741933, 'NumRet': 54000.0, 'num_q': 54.0}

