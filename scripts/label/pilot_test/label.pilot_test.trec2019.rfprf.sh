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
# 
# Prepend labeled positives, to construct new feedback documents with RF + PRF 
fb_ranking=experiments/pilot_test/trec2019/wo_qe/label.py/$(ls experiments/pilot_test/trec2019/wo_qe/label.py/)/rfprf${fb_docs}.fd.jsonl
if [ ! -f ${fb_ranking} ];then
    python -m preprocessing.feedback_documents.construct_fd_file \
    --fb_docs ${fb_docs} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv \
    --ranking experiments/pilot_test/trec2019/wo_qe/label.py/$(ls experiments/pilot_test/trec2019/wo_qe/label.py/)/ranking.tsv \
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
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.4978452644011152, 'nDCG@10': 0.6959937463567007, 'RR(rel=2)@10': 0.8254152823920266, 'R(rel=2)@1000': 0.8254465831278177, 'R(rel=2)@10': 0.2768393778141213, 'R(rel=2)@100': 0.6611391833080983, 'P(rel=2)@10': 0.6023255813953488, 'P(rel=2)@100': 0.2765116279069767, 'nDCG@50': 0.6700111985352575, 'nDCG@100': 0.6633421059820407, 'nDCG@200': 0.6733270805291451, 'nDCG@500': 0.700346802378282, 'nDCG@1000': 0.7084211009072544, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_docs: 5, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.499823213632285, 'nDCG@10': 0.7156758388774513, 'RR(rel=2)@10': 0.8523809523809524, 'R(rel=2)@1000': 0.8220169874946666, 'R(rel=2)@10': 0.28603014121973913, 'R(rel=2)@100': 0.6555363669267485, 'P(rel=2)@10': 0.6116279069767443, 'P(rel=2)@100': 0.26325581395348835, 'nDCG@50': 0.6684542532489208, 'nDCG@100': 0.6619488919425556, 'nDCG@200': 0.6719273657372212, 'nDCG@500': 0.7015163109945477, 'nDCG@1000': 0.7108666976734987, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5109598947401603, 'nDCG@10': 0.7223797715643993, 'RR(rel=2)@10': 0.8382705795496492, 'R(rel=2)@1000': 0.8244694921723499, 'R(rel=2)@10': 0.2918264036707124, 'R(rel=2)@100': 0.6638022686248036, 'P(rel=2)@10': 0.6325581395348836, 'P(rel=2)@100': 0.2767441860465116, 'nDCG@50': 0.6847844195360554, 'nDCG@100': 0.6734228272863338, 'nDCG@200': 0.6859265539214026, 'nDCG@500': 0.7089988841580592, 'nDCG@1000': 0.7173775932306546, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_docs: 5, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5103731768758247, 'nDCG@10': 0.7194389309687806, 'RR(rel=2)@10': 0.8542635658914727, 'R(rel=2)@1000': 0.8222909477546033, 'R(rel=2)@10': 0.29427060305878644,'R(rel=2)@100': 0.6514242209247975, 'P(rel=2)@10': 0.6162790697674418, 'P(rel=2)@100': 0.263953488372093, 'nDCG@50': 0.6784972163283314, 'nDCG@100': 0.6652023206753621, 'nDCG@200': 0.6808307283061745, 'nDCG@500': 0.7066080978518093, 'nDCG@1000': 0.7157736318033489, 'NumRet': 42962.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5218810729830424, 'nDCG@10': 0.7314771968865688, 'RR(rel=2)@10': 0.8630490956072352, 'R(rel=2)@1000': 0.8290227317803313, 'R(rel=2)@10': 0.2882499656035269, 'R(rel=2)@100': 0.6668148076817371, 'P(rel=2)@10': 0.6465116279069768, 'P(rel=2)@100': 0.28325581395348826, 'nDCG@50': 0.690581517244764, 'nDCG@100': 0.6803407795813846, 'nDCG@200': 0.6901408037944763, 'nDCG@500': 0.7163807209027441, 'nDCG@1000': 0.724373549063644, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 5, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5104196473966182, 'nDCG@10': 0.7279328917517155, 'RR(rel=2)@10': 0.8587486157253598, 'R(rel=2)@1000': 0.8256466834557212, 'R(rel=2)@10': 0.2936862120880424, 'R(rel=2)@100': 0.6604406778998259, 'P(rel=2)@10': 0.6488372093023256, 'P(rel=2)@100': 0.26999999999999996, 'nDCG@50': 0.6758940660762419, 'nDCG@100': 0.6687164140521549, 'nDCG@200': 0.6784031243750178, 'nDCG@500': 0.7073630946526781, 'nDCG@1000': 0.7164595146609716, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5326912846804301, 'nDCG@10': 0.7521269259258266, 'RR(rel=2)@10': 0.8814599483204133, 'R(rel=2)@1000': 0.8280701762904678, 'R(rel=2)@10': 0.30502809553262294, 'R(rel=2)@100': 0.6704820143764301, 'P(rel=2)@10': 0.6790697674418604, 'P(rel=2)@100': 0.28465116279069763, 'nDCG@50': 0.7025489114516457, 'nDCG@100': 0.6888579875981995, 'nDCG@200': 0.700399157939467, 'nDCG@500': 0.7226773626927793, 'nDCG@1000': 0.7308739016927973, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 5, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'AP(rel=2)@1000': 0.5199395902138751, 'nDCG@10': 0.7279912794788675, 'RR(rel=2)@10': 0.8627906976744186, 'R(rel=2)@1000': 0.8259105786527264, 'R(rel=2)@10': 0.29787457130543693,'R(rel=2)@100': 0.6596798196873892, 'P(rel=2)@10': 0.6534883720930234, 'P(rel=2)@100': 0.27162790697674416, 'nDCG@50': 0.6870568450772758, 'nDCG@100': 0.6728334820031187, 'nDCG@200': 0.6870403893357087, 'nDCG@500': 0.7125629750678423, 'nDCG@1000': 0.7215178021903078, 'NumRet': 43000.0, 'num_q': 43.0}

