#!/bin/bash
device=$1 #TODO: input arg

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
    #TODO: uncomment
    # CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    # --queries ${queries} \
    # --nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    # --checkpoint ${checkpoint} --root ${experiment_root} --experiment ${ann_experiment}
else
    echo "We have ANN search result at: \"${topk}\""
fi



# 2. Exact-NN Search, using query expansion.
topk=${experiment_root}/${ann_experiment}/retrieve.py/$(ls ${experiment_root}/${ann_experiment}/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
rerank_experiment=wo_qe #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    #TODO: uncomment
    # CUDA_VISIBLE_DEVICES=${device} python \
    # -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    # --topk ${topk} --batch --log-scores \
    # --queries ${queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    # --root ${experiment_root} --experiment ${rerank_experiment} \
    # --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    # --exp_embs 0 --exp_beta 0.0 --depth 1000
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
#* {'nDCG@10': 0.6884487503881044, 'nDCG@25': 0.650390907305978, 'nDCG@50': 0.6374881537047002, 'nDCG@100': 0.6333195347080279, 'nDCG@200': 0.6432465945383569, 'nDCG@500': 0.674583549264561, 'nDCG@1000': 0.6893402130910862, 'R(rel=2)@3': 0.1415566113353691, 'R(rel=2)@5': 0.19001514820210652, 'R(rel=2)@10': 0.28131172610629884, 'R(rel=2)@25': 0.40800364710177844, 'R(rel=2)@50': 0.5230915774010736, 'R(rel=2)@100': 0.624126813193234, 'R(rel=2)@200': 0.7047147253541708, 'R(rel=2)@1000': 0.8059874797801261, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.6976744186046513, 'P(rel=2)@5': 0.6697674418604652, 'P(rel=2)@10': 0.5976744186046511, 'P(rel=2)@25': 0.4465116279069766, 'P(rel=2)@50': 0.3423255813953487, 'P(rel=2)@100': 0.2502325581395348, 'P(rel=2)@200': 0.16313953488372093, 'AP(rel=2)@1000': 0.4629704540325707, 'RR(rel=2)@10': 0.8155684754521964, 'NumRet': 42964.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
#* {'nDCG@10': 0.6996191676961366, 'nDCG@25': 0.660308544811616, 'nDCG@50': 0.6468568119807828, 'nDCG@100': 0.6419724695294958, 'nDCG@200': 0.6514069510353638, 'nDCG@500': 0.6818872751153267, 'nDCG@1000': 0.6963564628761999, 'R(rel=2)@3': 0.13427849571086078, 'R(rel=2)@5': 0.1841068344211011, 'R(rel=2)@10': 0.2875825212503319, 'R(rel=2)@25': 0.4184246086356199, 'R(rel=2)@50': 0.5317990361876885, 'R(rel=2)@100': 0.6326877005713728, 'R(rel=2)@200': 0.7134365393099832, 'R(rel=2)@1000': 0.8113306150029239, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7286821705426357, 'P(rel=2)@5': 0.7069767441860467, 'P(rel=2)@10': 0.6325581395348838, 'P(rel=2)@25': 0.46604651162790695, 'P(rel=2)@50': 0.3548837209302326, 'P(rel=2)@100': 0.2574418604651163, 'P(rel=2)@200': 0.1672093023255814, 'AP(rel=2)@1000': 0.469969658251788, 'RR(rel=2)@10': 0.8330103359173127, 'NumRet': 43000.0, 'num_q': 43.0}
