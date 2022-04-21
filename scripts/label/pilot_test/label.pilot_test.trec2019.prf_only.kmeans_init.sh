#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg
kmeans_init=$6 #TODO: input arg: `avg_step_position`, `top1_step_position`, or `random`

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
#!@ original
# rerank_experiment=kmeans.prf_only.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
#!@ custom
rerank_experiment=kmeans.prf_only.${kmeans_init}.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
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
    --depth 1000 \
    \
    --kmeans_init ${kmeans_init} #!@ custom
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
# kmeans_init: avg_step_position, fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.4910724824549166, 'nDCG@10': 0.7054803166079344, 'RR(rel=2)@10': 0.850609080841639, 'R(rel=2)@1000': 0.822393274771521, 'nDCG@50': 0.6607384895191615, 'nDCG@100': 0.6519166510968984, 'nDCG@200': 0.6604753983800941, 'nDCG@500': 0.689642293640753, 'nDCG@1000': 0.7042007834585767, 'NumRet': 42962.0, 'num_q': 43.0}
# kmeans_init: avg_step_position, fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.4991574822608041, 'nDCG@10': 0.7155511631108803, 'RR(rel=2)@10': 0.8511904761904762, 'R(rel=2)@1000': 0.8216206006777675, 'nDCG@50': 0.6690359662000773, 'nDCG@100': 0.6589918121607751, 'nDCG@200': 0.6717044878741572, 'nDCG@500': 0.6967228494664418, 'nDCG@1000': 0.7099627427089363, 'NumRet': 42963.0, 'num_q': 43.0}
# kmeans_init: top1_step_position, fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5036655895488285, 'nDCG@10': 0.7110116893894763, 'RR(rel=2)@10': 0.8095238095238095, 'R(rel=2)@1000': 0.8224107414798698, 'R(rel=2)@10': 0.29117136880522154,'R(rel=2)@100': 0.6423044597388334, 'P(rel=2)@10': 0.6139534883720927, 'P(rel=2)@100': 0.26325581395348835, 'nDCG@50': 0.6726946368630609, 'nDCG@100': 0.6609077051499458, 'nDCG@200': 0.6718532492684336, 'nDCG@500': 0.6968195143379564, 'nDCG@1000': 0.7105260479273356, 'NumRet': 42963.0, 'num_q': 43.0}
# kmeans_init: random, fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.49566390973344965, 'nDCG@10': 0.7127769245664793, 'RR(rel=2)@10': 0.8480897009966777, 'R(rel=2)@1000': 0.8239894620386171, 'R(rel=2)@10': 0.2827508767509649,'R(rel=2)@100': 0.6407578710890075, 'P(rel=2)@10': 0.6046511627906976, 'P(rel=2)@100': 0.26209302325581396, 'nDCG@50': 0.6672202416028881, 'nDCG@100': 0.6582864407645576, 'nDCG@200': 0.6738782403861875, 'nDCG@500': 0.6975774973894795, 'nDCG@1000': 0.7109698525514536, 'NumRet': 42962.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# kmeans_init: avg_step_position, fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.49678467840601537, 'nDCG@10': 0.7047486542256665, 'RR(rel=2)@10': 0.8533591731266151, 'R(rel=2)@1000': 0.8260142709059335, 'nDCG@50': 0.6646475601498076, 'nDCG@100': 0.6558469500080151, 'nDCG@200': 0.664943855821488, 'nDCG@500': 0.6931277853018166, 'nDCG@1000': 0.7074749098822583, 'NumRet': 43000.0, 'num_q': 43.0}
# kmeans_init: avg_step_position, fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5057612857667395, 'nDCG@10': 0.7190334699112224, 'RR(rel=2)@10': 0.8556201550387598, 'R(rel=2)@1000': 0.825054253390824, 'nDCG@50': 0.6738609908412226, 'nDCG@100': 0.6639236371636662, 'nDCG@200': 0.6760336566659717, 'nDCG@500': 0.7002810879123601, 'nDCG@1000': 0.7132432985470754, 'NumRet': 43000.0, 'num_q': 43.0}
# kmeans_init: top1_step_position, fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.513660651380087, 'nDCG@10': 0.7159353850114576, 'RR(rel=2)@10': 0.8147286821705426, 'R(rel=2)@1000': 0.8258157938191985, 'R(rel=2)@10': 0.2918735102952485, 'R(rel=2)@100': 0.6504823525319922, 'P(rel=2)@10': 0.641860465116279, 'P(rel=2)@100': 0.2709302325581394, 'nDCG@50': 0.6805326713327922, 'nDCG@100': 0.667178650948344, 'nDCG@200': 0.6776308588385049, 'nDCG@500': 0.7017736535676481, 'nDCG@1000': 0.7152578111651159, 'NumRet': 43000.0, 'num_q': 43.0}
# kmeans_init: random, fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5029681351169358, 'nDCG@10': 0.7169086290877947, 'RR(rel=2)@10': 0.8525193798449614, 'R(rel=2)@1000': 0.8275768152374174, 'R(rel=2)@10': 0.28313354611943026,'R(rel=2)@100': 0.6488668162847218, 'P(rel=2)@10': 0.6348837209302326, 'P(rel=2)@100': 0.26953488372093015, 'nDCG@50': 0.6728148532468627, 'nDCG@100': 0.6630354972637909, 'nDCG@200': 0.6781496662025236, 'nDCG@500': 0.7011169418169425, 'nDCG@1000': 0.7143045214879845, 'NumRet': 43000.0, 'num_q': 43.0}

