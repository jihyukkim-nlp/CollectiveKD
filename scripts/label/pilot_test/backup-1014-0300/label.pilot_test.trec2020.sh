#!/bin/bash
device=$1 #TODO: input arg
exp_embs=$2 #TODO: input arg
exp_beta=$3 #TODO: input arg

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
rerank_experiment=maxsim.exp_embs${exp_embs}.exp_beta${exp_beta} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --exp_embs ${exp_embs} --exp_beta ${exp_beta} \
    --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --ranking ${ranking} > ${result_path}
# python -m utility.evaluate.trec_passages --qrels data/trec2020/2020qrels-pass.txt --ranking ${ranking} > ${result_path} #?@ debugging
cat ${result_path} | tail -1
# 3-2. After filtering
# qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
for thr in 0 32 34 35 36 37 38;do
    python -m pilot_test.label.evaluate_labels \
    --k 9999 --thr ${thr} --ranking ${ranking} --qrels ${qrels} --qrels_exclude ${qrels_exclude}
    echo
done
# data/trec2020/2020qrels-pass.txt
# {'ndcg_cut_10': 0.7447209680540077, 'ndcg_cut_200': 0.6895383136913082, 'map_cut_1000': 0.5021886471002371, 'recall_100': 0.6128673241789735, 'recall_200': 0.678330111170044, 'recall_500': 0.7296847968458751, 'recall_1000': 0.7579279179798751}
# data/pilot_test/label/2020qrels-pass.test.tsv
# {'ndcg_cut_10': 0.6487725730667521, 'ndcg_cut_200': 0.6380206563365192, 'map_cut_1000': 0.4463882235307737, 'recall_100': 0.6062299043784521, 'recall_200': 0.6729022173520504, 'recall_500': 0.7253247545905679, 'recall_1000': 0.7542258708418311}
# === k 9999, thr 0.0 ===                                                                                                                                                            
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv                                                    
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv                                                                                                            
#          # of queries = 54                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 53951 (999.09 per query)                                                                                                       
#         #> The # of labeled gold positives = 3552 (65.78 per query)                                                                                                                
#         #> precision 0.042, recall 0.645, F1 0.080                                                                                                                                 
                                                                                                                                                                                   
# === k 9999, thr 30.0 ===                                                                                                                                                           
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv                                                    
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv                                                                                                            
#          # of queries = 54                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 7011 (129.83 per query)                                                                                                        
#         #> The # of labeled gold positives = 3552 (65.78 per query)                                                                                                                
#         #> precision 0.213, recall 0.420, F1 0.283                                                                                                                                 
                                                                                                                                                                                   
# === k 9999, thr 31.0 ===                                                                                                                                                           
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv                                                    
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv                                                                                                             
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv                                                                                                            
#          # of queries = 54                                                                                                                                                         
#         #> The # of predicted unlabeled positives = 4271 (79.09 per query)                                                                                                         
#         #> The # of labeled gold positives = 3552 (65.78 per query)                                                                                                                
#         #> precision 0.288, recall 0.347, F1 0.315
                                                                                                                                                             
# === k 9999, thr 32.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 2631 (48.72 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.354, recall 0.262, F1 0.301

# === k 9999, thr 33.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 1497 (27.72 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.413, recall 0.174, F1 0.245

# === k 9999, thr 34.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 808 (14.96 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.490, recall 0.111, F1 0.182

# === k 9999, thr 35.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 432 (8.00 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.574, recall 0.070, F1 0.124

# === k 9999, thr 36.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 192 (3.56 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.688, recall 0.037, F1 0.071


# === k 9999, thr 37.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 74 (1.37 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 0.851, recall 0.018, F1 0.035

# === k 9999, thr 38.0 === 
#          ranking       = experiments/pilot_test/trec2020/maxsim.exp_embs10.exp_beta1.0/label.py/2021-10-08_11.02.29/ranking.tsv
#          qrels         = data/pilot_test/label/2020qrels-pass.test.tsv
#          qrels_exclude = data/pilot_test/label/2020qrels-pass.train.tsv
#          # of queries = 54
#         #> The # of predicted unlabeled positives = 26 (0.48 per query)
#         #> The # of labeled gold positives = 3552 (65.78 per query)
#         #> precision 1.000, recall 0.007, F1 0.015