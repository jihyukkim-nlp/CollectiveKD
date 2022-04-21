#!/bin/bash
# wc -l finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/retrieve.py/2021-10-30_13.42.32/unordered.tsv 
# 4222973109 finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/retrieve.py/2021-10-30_13.42.32/unordered.tsv
# wc -l data/queries.train.reduced.tsv 
# 502939 data/queries.train.reduced.tsv
# 4222973109 / 502939 = 8396.591055774159491

mkdir -p experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
mkdir -p experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k
mkdir -p experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k

- 1. [pre-training   ] (300k iterations) "colbert.teacher.dnn", trained using BM25 negatives (official train triples)
- update ranking: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.[jsonl/tsv]"
- update kd_expansion_pt: "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
- update triples with HN: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"

#TODO:
2. [1st fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k", trained using HN from the pre-trained retrieval 
- training
devices=2,3
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt
checkpoint=data/checkpoints/colbert.teacher.dnn
new_exp=experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
mkdir -p ${new_exp}
sh scripts/mutual_learning/msmarco_psg.training.stepwise.mse.sh ${devices} ${kd_expansion_pt} ${checkpoint} ${new_exp}

mkdir -p experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 25000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 50000
sh scripts/validation/msmarco_psg.sh 0 experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 25000 > experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/25000.log
sh scripts/validation/msmarco_psg.sh 2 experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k 50000 > experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/50000.log


- update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
indexing: 
exp_root=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
step=50000
sh scripts/indexing/indexing.sh ${exp_root} ${step}

re-ranking:
device=0
topk=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.tsv # fixed for training efficiency
exp_root=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k # same with "new_exp" used for training script
step=50000
sh scripts/mutual_learning/msmarco_psg.nn_search.sh ${device} ${topk} ${exp_root} ${step}

- update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"
device=0
fb_k=10 #TODO: custom arg
beta=0.5 #TODO: custom arg
fb_clusters=24 #TODO: custom arg
fb_docs=3 #TODO: custom arg
exp_root=experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k # same with "new_exp" used for training script
step=50000
sh scripts/mutual_learning/msmarco_psg.qe.sh ${device} ${fb_k} ${beta} ${fb_clusters} ${fb_docs} ${exp_root} ${step}
> expansion_pt="experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/2021-*/expansion.pt"

#!  Update "exp_root" for every iteration
- 3. [2nd fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k", trained using HN from the pre-trained retrieval 
- update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
- update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"

#!  Update "exp_root" for every iteration
- 4. [3rd fine-tuning] (+50k iterations) "mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k", trained using HN from the pre-trained retrieval 
- update ranking: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/ranking.[jsonl/tsv]"
- update kd_expansion_pt: "experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k/MSMARCO-psg-train-prf.docs3.k10.beta0.5.clusters24/label.py/[]/expansion.pt"




# 3-3. Pilot test
#TODO: copy from scripts/kd_prf_2nd_iter/
sh scripts/kd_prf_2nd_iter/pilot_test/trec2019.sh 2
sh scripts/kd_prf_2nd_iter/pilot_test/trec2019.prf.sh 2 3 10 0.5 24
sh scripts/kd_prf_2nd_iter/pilot_test/trec2020.sh 2
sh scripts/kd_prf_2nd_iter/pilot_test/trec2020.prf.sh 2 3 10 0.5 24


