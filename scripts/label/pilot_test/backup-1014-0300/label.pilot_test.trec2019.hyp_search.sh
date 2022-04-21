#!/bin/bash

DATA_DIR=/workspace/DataCenter/PassageRanking/MSMARCO # sonic
retriever=colbert.teacher
queries=data/queries.trec2019.tsv
index_root=experiments/${retriever}/MSMARCO-psg/index.py
checkpoint=data/checkpoints/colbert.teacher.dnn
experiment_root=experiments/pilot_test/trec2019
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return
# 1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
experiment=wo_qe
CUDA_VISIBLE_DEVICES=2 python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--queries ${queries} \
--nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--checkpoint ${checkpoint} --root ${experiment_root} --experiment ${experiment}



experiment=wo_qe
experiment_root=experiments/pilot_test/trec2019
topk=${experiment_root}/${experiment}/retrieve.py/$(ls ${experiment_root}/${experiment}/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
DATA_DIR=/workspace/DataCenter/PassageRanking/MSMARCO # sonic
retriever=colbert.teacher
queries=data/queries.trec2019.tsv
index_root=experiments/${retriever}/MSMARCO-psg/index.py
checkpoint=data/checkpoints/colbert.teacher.dnn
experiment_root=experiments/pilot_test/trec2019
# 2. Exact-NN search, using expanded queries by pseudo-relevance feedbacks (gold feedback documents & pseudo-relevance term selection )
# default values
exp_embs=10
exp_beta=1.0
exp_thr=0.0
exp_mmr_thr=1.0
# 
device=0
# for exp_embs in 10;do
for exp_beta in 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5;do
# for exp_thr in 0.0;do
# for exp_mmr_thr in 1.0;do
# experiment=impact.exp_embs${exp_embs}-exp_beta${exp_beta}-exp_thr${exp_thr}-exp_mmr_thr${exp_mmr_thr}
experiment=impact.exp_beta${exp_beta}
echo "CUDA_VISIBLE_DEVICES=${device} python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--topk ${topk} --batch --log-scores \
--queries ${queries} --index_root ${index_root} --index_name MSMARCO.L2.32x200k --checkpoint ${checkpoint} \
--root ${experiment_root} --experiment ${experiment} \
--qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection ${DATA_DIR}/collection.tsv \
--exp_embs ${exp_embs} --exp_beta ${exp_beta} --exp_thr ${exp_thr} --exp_mmr_thr ${exp_mmr_thr} \
--depth 1000"
device=$(expr ${device} + 1)
# done
# done
# done
done



# Evaluation
queries=data/queries.trec2019.tsv
experiment_root=experiments/pilot_test/trec2019
# default values
exp_embs=10
exp_beta=1.0
exp_thr=0.0
exp_mmr_thr=1.0
# 
# for exp_embs in 10;do
for exp_beta in 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5;do
# for exp_thr in 0.0;do
# for exp_mmr_thr in 1.0;do
# experiment=impact.exp_embs${exp_embs}-exp_beta${exp_beta}-exp_thr${exp_thr}-exp_mmr_thr${exp_mmr_thr}
experiment=impact.exp_beta${exp_beta}
# 
queries=data/queries.trec2019.tsv
ranking=${experiment_root}/${experiment}/label.py/$(ls ${experiment_root}/${experiment}/label.py)/queries.trec2019.ranking.jsonl
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
# sanity check
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
# evaluation
result_path=${experiment_root}/${experiment}/label.py/$(ls ${experiment_root}/${experiment}/label.py)/ranking.metrics
python -m utility.evaluate.trec_passages --qrels data/pilot_test/label/2019qrels-pass.test.tsv --ranking ${ranking} > ${result_path}
cat ${result_path} | tail -1
echo
# done
# done
# done
done



conda activate colbert
CUDA_VISIBLE_DEVICES=0 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.01 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.01 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=1 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.02 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.02 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=2 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.03 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.03 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=3 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.05 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.05 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=4 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.1 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.1 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=5 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.2 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.2 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=6 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.3 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv --exp_embs 10 --exp_beta 0.3 --exp_thr 0.0 --exp_mmr_thr 1.0 --depth 1000
CUDA_VISIBLE_DEVICES=7 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --experiment impact.exp_beta0.5 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.t




#?@ debugging: start #########
CUDA_VISIBLE_DEVICES=7 python -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 --topk experiments/pilot_test/trec2019/wo_qe/retrieve.py/2021-09-12_01.55.52/unordered.tsv --batch --log-scores --queries data/queries.trec2019.tsv --index_root experiments/colbert.teacher/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k --checkpoint data/checkpoints/colbert.teacher.dnn --root experiments/pilot_test/trec2019 --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--experiment debugging --exp_embs 10 --exp_beta 1.0 --depth 1000
# 
# Evaluation
queries=data/queries.trec2019.tsv
ranking=experiments/pilot_test/trec2019/debugging/label.py/2021-10-06_06.26.23/queries.trec2019.ranking.jsonl
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
result_path=experiments/pilot_test/trec2019/debugging/label.py/2021-10-06_06.26.23/ranking.metrics
python -m utility.evaluate.trec_passages --qrels data/pilot_test/label/2019qrels-pass.test.tsv --ranking ${ranking} > ${result_path}
cat ${result_path} | tail -1
{'ndcg_cut_10': 0.6805611088245922, 'ndcg_cut_200': 0.6622118552120205, 'map_cut_1000': 0.4735306587178119, 'recall_100': 0.5590749398368665, 'recall_200': 0.6492189368303629, 'recall_500': 0.7237083971913443, 'recall_1000': 0.7506573655887914}
# elapsed: 156.97200322151184
# rm -r /workspace/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/pilot_test/trec2019/exp_embs10-exp_beta1.0/
#?@ debugging: end   #########
