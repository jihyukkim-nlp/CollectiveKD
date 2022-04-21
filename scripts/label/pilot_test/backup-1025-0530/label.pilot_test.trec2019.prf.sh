#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg
# 
# rf_fb_k=$6 #TODO: input arg
# rf_fb_clusters=$7 #TODO: input arg
# rf_beta=$8 #TODO: input arg

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
rerank_experiment=kmeans.rf_then_prf.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
rf_dir=experiments/pilot_test/trec2019/kmeans.k10.beta0.5.clusters10/label.py/$(ls experiments/pilot_test/trec2019/kmeans.k10.beta0.5.clusters10/label.py/)
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
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7900236250822773, 'nDCG@25': 0.7631403804229614, 'nDCG@50': 0.7389012703257533, 'nDCG@100': 0.7209788651724895, 'nDCG@200': 0.718721457868721, 'nDCG@500': 0.7375759756646981, 'nDCG@1000': 0.7470364893876701, 'R(rel=2)@3': 0.1677773553475498, 'R(rel=2)@5': 0.21676913353902474, 'R(rel=2)@10': 0.31885961001855595, 'R(rel=2)@25': 0.4920102286088171, 'R(rel=2)@50': 0.609290947208698, 'R(rel=2)@100': 0.6990480362583636, 'R(rel=2)@200': 0.7661202352937455, 'R(rel=2)@1000': 0.8238883621011537, 'P(rel=2)@3': 0.8294573643410852, 'P(rel=2)@5': 0.7534883720930234, 'P(rel=2)@10': 0.6790697674418604, 'P(rel=2)@25': 0.5367441860465114, 'P(rel=2)@50': 0.40558139534883725, 'P(rel=2)@100': 0.29162790697674423, 'P(rel=2)@200': 0.1813953488372093, 'AP(rel=2)@1000': 0.5449886868612445, 'RR(rel=2)@10': 0.9331395348837209, 'NumRet': 42962.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.8244753126325591, 'nDCG@25': 0.7843416089867287, 'nDCG@50': 0.7566950094876628, 'nDCG@100': 0.7373548747194245, 'nDCG@200': 0.7340323522081047, 'nDCG@500': 0.7526339405918907, 'nDCG@1000': 0.7619002075244491, 'R(rel=2)@3': 0.16979102950254868, 'R(rel=2)@5': 0.2252481151961091, 'R(rel=2)@10': 0.3295260307034517, 'R(rel=2)@25': 0.4974388376088103, 'R(rel=2)@50': 0.6156796426668021, 'R(rel=2)@100': 0.7044126665128654, 'R(rel=2)@200': 0.769879311474801, 'R(rel=2)@1000': 0.8274711684262395, 'P(rel=2)@3': 0.8992248062015502, 'P(rel=2)@5': 0.8186046511627909, 'P(rel=2)@10': 0.727906976744186, 'P(rel=2)@25': 0.5590697674418603, 'P(rel=2)@50': 0.4190697674418605, 'P(rel=2)@100': 0.29930232558139536, 'P(rel=2)@200': 0.18534883720930237, 'AP(rel=2)@1000': 0.5724874155414553, 'RR(rel=2)@10': 0.9689922480620154, 'NumRet': 43000.0, 'num_q': 43.0}

