#!/bin/bash
# wc -l finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/retrieve.py/2021-10-30_13.42.32/unordered.tsv 
# 4222973109 finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/retrieve.py/2021-10-30_13.42.32/unordered.tsv
# wc -l data/queries.train.reduced.tsv 
# 502939 data/queries.train.reduced.tsv
# 4222973109 / 502939 = 8396.591055774159491

mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k
mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k

1. [pre-training   ] (300k iterations) "colbert.teacher.dnn", trained using BM25 negatives (official train triples)
- update ranking: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.[jsonl/tsv]"
- update kd_expansion_pt: "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
- update triples with HN: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"

2. [1st fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k", trained using HN from the pre-trained retrieval 
- training
mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
devices=0,1
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt
checkpoint=data/checkpoints/colbert.teacher.dnn
new_exp=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
maxsteps=50000
sh scripts/mutual_learning/msmarco_psg.training.stepwise.sh ${devices} ${kd_expansion_pt} ${checkpoint} ${new_exp} ${maxsteps}
# checkpoint: experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-50000.dnn
# cat experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/train.py/msmarco.psg.l2/logs/elapsed.txt 
# 20751.897797346115 = 5.764416054818365 hours

mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 25000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 50000
sh scripts/validation/msmarco_psg.sh 0 experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 25000 > experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/25000.log
sh scripts/validation/msmarco_psg.sh 1 experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 50000 > experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/50000.log


- update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
indexing: 
devices=0,1
exp_root=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
step=50000
sh scripts/indexing/indexing.sh ${devices} ${exp_root} ${step}
# index: experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/index.py
# du -hs experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/index.py
# 168G	experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/index.py
# cat experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/index.py/2021-*/logs/elapsed.txt
# 6211.149911403656
# cat experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/index_faiss.py/2021-*/logs/elapsed.txt
# 1654.5416250228882
# ==> total: 7865.6915364265442 = 2.18491431567404 hours


re-ranking:
device=0
topk=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.tsv # fixed for training efficiency
exp_root=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k # same with "new_exp" used for training script
step=50000
sh scripts/mutual_learning/msmarco_psg.nn_search.sh ${device} ${topk} ${exp_root} ${step}
# fb_ranking: experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train/label.py/ranking.[tsv/jsonl]
# du -hs experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train/label.py/ranking.*
# 141M	experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train/label.py/ranking.jsonl
# 171M	experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train/label.py/ranking.tsv
# cat experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train/label.py/2021-*/logs/elapsed.txt
# 1720.2617590427399
# 1732.6543273925781
# 1034.496298789978
# ==> total: 4487.412385225296 = 1.24650344034036 hours

#TODO:
- update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"
device=0
fb_k=10 #TODO: custom arg
beta=0.5 #TODO: custom arg
fb_clusters=24 #TODO: custom arg
fb_docs=3 #TODO: custom arg
exp_root=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k # same with "new_exp" used for training script
step=50000
sh scripts/mutual_learning/msmarco_psg.qe.sh ${device} ${fb_k} ${beta} ${fb_clusters} ${fb_docs} ${exp_root} ${step}
# expansion_pt=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/2021-11-02_23.48.24/expansion.pt
# cat experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/2021-*/logs/elapsed.txt
# 17307.66862130165 seconds = 4.807685728139347 hours
# du -hs experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/2021-*/expansion.pt
# 2.6G	experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/2021-11-02_23.48.24/expansion.pt

#! Update "exp_root" for every iteration
#! Update "maxsteps" for every iteration
#! Update "checkpoint" for every iteration
#! Update "kd_expansion_pt" for every iteration
3. [2nd fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k", trained using HN from the pre-trained retrieval 
- training
mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k
devices=0,1
kd_expansion_pt=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/2021-11-02_23.48.24/expansion.pt
checkpoint=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-50000.dnn
new_exp=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k
maxsteps=100000
sh scripts/mutual_learning/msmarco_psg.training.stepwise.sh ${devices} ${kd_expansion_pt} ${checkpoint} ${new_exp} ${maxsteps}

mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k 75000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k 100000
sh scripts/validation/msmarco_psg.sh 0 experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k 75000 > experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg/test.py/75000.log
sh scripts/validation/msmarco_psg.sh 1 experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k 100000 > experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg/test.py/100000.log

- update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
- update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"

#! Update "exp_root" for every iteration
#! Update "maxsteps" for every iteration
#! Update "checkpoint" for every iteration
#! Update "kd_expansion_pt" for every iteration
4. [3rd fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k", trained using HN from the pre-trained retrieval 
- update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
- update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"




# 3-3. Pilot test
#TODO: copy from scripts/kd_prf_2nd_iter/
sh scripts/kd_prf_2nd_iter/pilot_test/trec2019.sh 2
sh scripts/kd_prf_2nd_iter/pilot_test/trec2019.prf.sh 2 3 10 0.5 24
sh scripts/kd_prf_2nd_iter/pilot_test/trec2020.sh 2
sh scripts/kd_prf_2nd_iter/pilot_test/trec2020.prf.sh 2 3 10 0.5 24


