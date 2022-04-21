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
rerank_experiment=kmeans.prf_only.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
fb_ranking=experiments/pilot_test/trec2019/wo_qe/label.py/$(ls experiments/pilot_test/trec2019/wo_qe/label.py/)/ranking.jsonl
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
qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7155511631108803, 'nDCG@25': 0.6809362305939063, 'nDCG@50': 0.6690359662000773, 'nDCG@100': 0.6589918121607751, 'nDCG@200': 0.6717044878741572, 'nDCG@500': 0.6967228494664418, 'nDCG@1000': 0.7099627427089363, 'R(rel=2)@3': 0.14232553402036863, 'R(rel=2)@5': 0.19044961455654547, 'R(rel=2)@10': 0.29329704170411003, 'R(rel=2)@25': 0.4342809061453624, 'R(rel=2)@50': 0.5595348187785693, 'R(rel=2)@100': 0.6431112613289401, 'R(rel=2)@200': 0.7386196985948202, 'R(rel=2)@1000': 0.8216206006777675, 'P(rel=2)@1': 0.813953488372093, 'P(rel=2)@3': 0.7131782945736436, 'P(rel=2)@5': 0.6697674418604651, 'P(rel=2)@10': 0.6162790697674417, 'P(rel=2)@25': 0.47069767441860444, 'P(rel=2)@50': 0.36279069767441857, 'P(rel=2)@100': 0.2620930232558139, 'P(rel=2)@200': 0.17593023255813955, 'AP(rel=2)@1000': 0.4991574822608041, 'RR(rel=2)@10': 0.8511904761904762, 'NumRet': 42963.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.7054803166079344, 'nDCG@25': 0.672923259129742, 'nDCG@50': 0.6607384895191615, 'nDCG@100': 0.6519166510968984, 'nDCG@200': 0.6604753983800941, 'nDCG@500': 0.689642293640753, 'nDCG@1000': 0.7042007834585767, 'R(rel=2)@3': 0.14038754952424456, 'R(rel=2)@5': 0.1914607369023493, 'R(rel=2)@10': 0.2853087122041526, 'R(rel=2)@25': 0.4308089546447973, 'R(rel=2)@50': 0.5540852794977343, 'R(rel=2)@100': 0.6389098828183223, 'R(rel=2)@200': 0.7236593842385501, 'R(rel=2)@1000': 0.822393274771521, 'P(rel=2)@3': 0.7054263565891473, 'P(rel=2)@5': 0.6744186046511628, 'P(rel=2)@10': 0.6069767441860464, 'P(rel=2)@25': 0.46604651162790695, 'P(rel=2)@50': 0.3581395348837209, 'P(rel=2)@100': 0.2604651162790697, 'P(rel=2)@200': 0.17255813953488372, 'AP(rel=2)@1000': 0.4910724824549166, 'RR(rel=2)@10': 0.850609080841639, 'NumRet': 42962.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7190334699112224, 'nDCG@25': 0.6839502718725956, 'nDCG@50': 0.6738609908412226, 'nDCG@100': 0.6639236371636662, 'nDCG@200': 0.6760336566659717, 'nDCG@500': 0.7002810879123601, 'nDCG@1000': 0.7132432985470754, 'R(rel=2)@3': 0.1316946066362192, 'R(rel=2)@5': 0.18579675923957195, 'R(rel=2)@10': 0.2963542575698357, 'R(rel=2)@25': 0.43963669315425813, 'R(rel=2)@50': 0.5634593635913152, 'R(rel=2)@100': 0.6511298978214967, 'R(rel=2)@200': 0.7445927505839498, 'R(rel=2)@1000': 0.825054253390824, 'P(rel=2)@1': 0.813953488372093, 'P(rel=2)@3': 0.7364341085271319, 'P(rel=2)@5': 0.6930232558139536, 'P(rel=2)@10': 0.6488372093023256, 'P(rel=2)@25': 0.4883720930232556, 'P(rel=2)@50': 0.37534883720930234, 'P(rel=2)@100': 0.26953488372093015, 'P(rel=2)@200': 0.18, 'AP(rel=2)@1000': 0.5057612857667395, 'RR(rel=2)@10': 0.8556201550387598, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.7047486542256665, 'nDCG@25': 0.6760746662192733, 'nDCG@50': 0.6646475601498076, 'nDCG@100': 0.6558469500080151, 'nDCG@200': 0.664943855821488, 'nDCG@500': 0.6931277853018166, 'nDCG@1000': 0.7074749098822583, 'R(rel=2)@3': 0.12917895368451973, 'R(rel=2)@5': 0.186765751487634, 'R(rel=2)@10': 0.281323294409313, 'R(rel=2)@25': 0.4331374851046292, 'R(rel=2)@50': 0.5573479560969042, 'R(rel=2)@100': 0.6430341047795333, 'R(rel=2)@200': 0.7307657180927005, 'R(rel=2)@1000': 0.8260142709059335, 'P(rel=2)@3': 0.7209302325581396, 'P(rel=2)@5': 0.6976744186046513, 'P(rel=2)@10': 0.6302325581395348, 'P(rel=2)@25': 0.4837209302325581, 'P(rel=2)@50': 0.3697674418604652, 'P(rel=2)@100': 0.26744186046511625, 'P(rel=2)@200': 0.1768604651162791, 'AP(rel=2)@1000': 0.49678467840601537, 'RR(rel=2)@10': 0.8533591731266151, 'NumRet': 43000.0, 'num_q': 43.0}
