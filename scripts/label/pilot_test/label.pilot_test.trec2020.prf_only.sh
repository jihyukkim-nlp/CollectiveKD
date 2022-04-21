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
rerank_experiment=kmeans.prf_only.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
fb_ranking=experiments/pilot_test/trec2020/wo_qe/label.py/$(ls experiments/pilot_test/trec2020/wo_qe/label.py/)/ranking.jsonl
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
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.6812088897932649, 'nDCG@25': 0.6364810478277884, 'nDCG@50': 0.6156095086833714, 'nDCG@100': 0.6195653898224079, 'nDCG@200': 0.6502837536361388, 'nDCG@500': 0.6774135636799733, 'nDCG@1000': 0.6875149473113147, 'R(rel=2)@3': 0.18077317483971847, 'R(rel=2)@5': 0.2775563202028621, 'R(rel=2)@10': 0.3913723598674821, 'R(rel=2)@25': 0.5361520961860409, 'R(rel=2)@50': 0.6220426938129163, 'R(rel=2)@100': 0.7018414697253428, 'R(rel=2)@200': 0.7839384158205213, 'R(rel=2)@1000': 0.8440249357113083, 'P(rel=2)@3': 0.6666666666666666, 'P(rel=2)@5': 0.6185185185185186, 'P(rel=2)@10': 0.5037037037037037, 'P(rel=2)@25': 0.33925925925925926, 'P(rel=2)@50': 0.22777777777777775, 'P(rel=2)@100': 0.14629629629629626, 'P(rel=2)@200': 0.0872222222222222, 'AP(rel=2)@1000': 0.452776805545477, 'RR(rel=2)@10': 0.8200617283950619, 'NumRet': 53952.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.6915759622495223, 'nDCG@25': 0.651287132151812, 'nDCG@50': 0.6347513185439797, 'nDCG@100': 0.6394052585402135, 'nDCG@200': 0.6640421939911841, 'nDCG@500': 0.6884181584426999, 'nDCG@1000': 0.6986328311277356, 'R(rel=2)@3': 0.18077317483971847, 'R(rel=2)@5': 0.2766483339133053, 'R(rel=2)@10': 0.398856733018837, 'R(rel=2)@25': 0.5476083852596952, 'R(rel=2)@50': 0.645002851749166, 'R(rel=2)@100': 0.729839944655182, 'R(rel=2)@200': 0.7908900300948027, 'R(rel=2)@1000': 0.8439460425650956, 'P(rel=2)@3': 0.6666666666666666, 'P(rel=2)@5': 0.6111111111111112, 'P(rel=2)@10': 0.5111111111111111, 'P(rel=2)@25': 0.3481481481481482, 'P(rel=2)@50': 0.2359259259259259, 'P(rel=2)@100': 0.1522222222222222, 'P(rel=2)@200': 0.08814814814814813, 'AP(rel=2)@1000': 0.4728487912164106, 'RR(rel=2)@10': 0.8308641975308642, 'NumRet': 53951.0, 'num_q': 54.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.6885155204680559, 'nDCG@25': 0.6459207218472783, 'nDCG@50': 0.6251576248660592, 'nDCG@100': 0.6267060994294082, 'nDCG@200': 0.6568601222189001, 'nDCG@500': 0.6842351267567333, 'nDCG@1000': 0.6942496874574772, 'R(rel=2)@3': 0.15843835207024112, 'R(rel=2)@5': 0.27599721670000044, 'R(rel=2)@10': 0.38921134194923446, 'R(rel=2)@25': 0.5396545429141235, 'R(rel=2)@50': 0.6308315789493557, 'R(rel=2)@100': 0.7058827089149365, 'R(rel=2)@200': 0.7840546573129827, 'R(rel=2)@1000': 0.8461340321277788, 'P(rel=2)@3': 0.6790123456790123, 'P(rel=2)@5': 0.6666666666666669, 'P(rel=2)@10': 0.5388888888888889, 'P(rel=2)@25': 0.3599999999999998, 'P(rel=2)@50': 0.2414814814814815, 'P(rel=2)@100': 0.15333333333333332, 'P(rel=2)@200': 0.09120370370370372, 'AP(rel=2)@1000': 0.46920678181262254, 'RR(rel=2)@10': 0.8240740740740742, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7050928028350049, 'nDCG@25': 0.6617098002216392, 'nDCG@50': 0.643111928388147, 'nDCG@100': 0.645560913726691, 'nDCG@200': 0.6700730087032704, 'nDCG@500': 0.6944169843121778, 'nDCG@1000': 0.7045979669463018, 'R(rel=2)@3': 0.16461119157641396, 'R(rel=2)@5': 0.277456583645939, 'R(rel=2)@10': 0.4064567772014026, 'R(rel=2)@25': 0.5584931435597346, 'R(rel=2)@50': 0.6512125062199494, 'R(rel=2)@100': 0.7334187942894212, 'R(rel=2)@200': 0.7914281738072042, 'R(rel=2)@1000': 0.8471683391316858, 'P(rel=2)@3': 0.6851851851851853,'P(rel=2)@5': 0.6592592592592593, 'P(rel=2)@10': 0.5518518518518517, 'P(rel=2)@25': 0.3725925925925926, 'P(rel=2)@50': 0.25037037037037035, 'P(rel=2)@100': 0.1596296296296296, 'P(rel=2)@200': 0.09240740740740741, 'AP(rel=2)@1000': 0.4876691034852261, 'RR(rel=2)@10': 0.8339506172839507, 'NumRet': 54000.0, 'num_q': 54.0}
