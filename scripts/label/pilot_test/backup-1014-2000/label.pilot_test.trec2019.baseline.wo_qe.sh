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
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --exp_embs 0 --exp_beta 0.0 --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --ranking ${ranking} > ${result_path}
# python -m utility.evaluate.trec_passages --qrels data/trec2019/2019qrels-pass.txt --ranking ${ranking} > ${result_path} #?@ debugging
cat ${result_path} | tail -1
# 3-2. After filtering
qrels_exclude=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
for thr in 0 26 28 29 30;do
    python -m pilot_test.label.evaluate_labels \
    --k 9999 --thr ${thr} --ranking ${ranking} --qrels ${qrels} --qrels_exclude ${qrels_exclude}
    echo
done
# data/trec2019/2019qrels-pass.txt
# {'ndcg_cut_10': 0.6996191676961366, 'ndcg_cut_200': 0.6514069510353638, 'map_cut_1000': 0.46660020246228395, 'recall_100': 0.5302957761642422, 'recall_200': 0.6220485783655566, 'recall_500': 0.705068386116036, 'recall_1000': 0.7393163110224397}
# data/pilot_test/label/2019qrels-pass.test.tsv
# {'ndcg_cut_10': 0.6644615183326829, 'ndcg_cut_200': 0.6322917291274348, 'map_cut_1000': 0.441825862147553, 'recall_100': 0.5245841539656716, 'recall_200': 0.6173795630729204, 'recall_500': 0.7015405357413402, 'recall_1000': 0.7362681824411912}
# === k 9999, thr 0.0 ===                                                                                                                                                            
#          ranking       = experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/ranking.tsv                                                                            
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv                                                                                                            
#          # of queries = 43                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 42964 (999.16 per query)                                                                                                       
#         #> The # of labeled gold positives = 4059 (94.40 per query)                                                                                                                
#         #> precision 0.063, recall 0.669, F1 0.116                                                                                                                                 
                                                                                                                                                                                   
# === k 9999, thr 26.0 ===                                                                                                                                                           
#          ranking       = experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/ranking.tsv                                                                            
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv                                                                                                            
#          # of queries = 43                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 1039 (24.16 per query)                                                                                                         
#         #> The # of labeled gold positives = 4059 (94.40 per query)                                                                                                                
#         #> precision 0.802, recall 0.205, F1 0.327                                                                                                                                 
                                                                                                                                                                                   
# === k 9999, thr 28.0 ===                                                                                                                                                           
#          ranking       = experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/ranking.tsv                                                                            
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv                                                                                                            
#          # of queries = 43                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 231 (5.37 per query)                                                                                                           
#         #> The # of labeled gold positives = 4059 (94.40 per query)                                                                                                                
#         #> precision 0.913, recall 0.052, F1 0.098                                                                                                                                 
                                                                                                                                                                                   
# === k 9999, thr 29.0 ===                                                                                                                                                           
#          ranking       = experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/ranking.tsv                                                                            
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv                                                                                                            
#          # of queries = 43                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 61 (1.42 per query)                                                                                                            
#         #> The # of labeled gold positives = 4059 (94.40 per query)                                                                                                                
#         #> precision 0.951, recall 0.014, F1 0.028                                                                                                                                 
                                                                                                                                                                                   
# === k 9999, thr 30.0 ===                                                                                                                                                           
#          ranking       = experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/ranking.tsv                                                                            
#          qrels         = data/pilot_test/label/2019qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2019qrels-pass.train.tsv                                                                                                            
#          # of queries = 43                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 3 (0.07 per query)                                                                                                             
#         #> The # of labeled gold positives = 4059 (94.40 per query)                                                                                                                
#         #> precision 1.000, recall 0.001, F1 0.001                   
