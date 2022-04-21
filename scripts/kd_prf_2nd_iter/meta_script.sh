#!/bin/bash

# 1. Indexing collection using the previously fine-tuned ColBERT
exp_root=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5
step=150000
sh scripts/indexing/indexing.sh ${exp_root} ${step}
# du -h experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index.py/MSMARCO.L2.32x200k/
# 163G    experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index.py/MSMARCO.L2.32x200k
# experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index_faiss.py/
# cat experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index.py/2021-10-30_11.26.32/logs/elapsed.txt 
# 3111.447897672653
# cat experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/index_faiss.py/2021-10-30_12.18.31/logs/elapsed.txt 
# 3687.5009772777557

# 2. Get nearest neighbor passages for each query in train set.
# : Those passages will be used as both ``hard negatives`` during the current fine-tuning step and ``feedback documents`` for PRF
device=0
sh scripts/kd_prf_2nd_iter/msmarco_psg.nn_search.sh ${device}
# ANN Search
# du -hs experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/retrieve.py/2021-10-30_13.42.32/unordered.tsv
# 71G
# cat experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/retrieve.py/2021-10-30_13.42.32/logs/elapsed.txt
# 11921.032135248184
# Exact-NN Search
# du -hs experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/*/*
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-30_22.10.03/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-30_22.10.03/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-30_22.10.03/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-30_23.13.12/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-30_23.13.12/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-30_23.13.12/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_00.18.48/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_00.18.48/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_00.18.48/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_01.27.08/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_01.27.08/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_01.27.08/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_02.33.18/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_02.33.18/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_02.33.18/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_03.42.26/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_03.42.26/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_03.42.26/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_04.48.17/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_04.48.17/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_04.48.17/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_05.55.51/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_05.55.51/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_05.55.51/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_07.01.58/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_07.01.58/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_07.01.58/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_08.11.06/logs
# 1.4G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_08.11.06/ranking.jsonl
# 1.8G	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_08.11.06/ranking.tsv
# 28K	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_09.16.32/logs
# 80M	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_09.16.32/ranking.jsonl
# 105M	experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/2021-10-31_09.16.32/ranking.tsv
# cat experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/*/logs/elapsed.txt 
# 3767.09325504303
# 3914.9775099754333
# 4084.604711532593
# 3950.223114013672
# 4132.094463586807
# 3935.650497674942
# 4033.286603450775
# 3950.8583920001984
# 4126.264284133911
# 3910.3445675373077
# 446.4878077507019
# du -hs experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/ranking.*    
# 14G     experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/ranking.jsonl
# 18G     experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/ranking.tsv

# 3-1. Construct new train triple using hard negatives.
hard_negatives=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/ranking.jsonl
n_negatives=1
sh scripts/label/msmarco_psg.triples.hn.sh ${hard_negatives} ${n_negatives}
# Construct new train triples using hard negatives (this will take about 8-10 minutes)
# #> Load positives
# [Oct 31, 10:06:49] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 31, 10:06:51] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.
# #> Elapsed time for loading positives: 1.9552910327911377
# #> Gather positive pairs
# #> Elapsed time for gathering positive pairs: 0.568547248840332
# #> Load negatives
# #> Load experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/ranking.jsonl
#         pid for 121352 => 1271786
#         pid for 634306 => 5686490
#         pid for 920825 => 3285660
#         pid for 510633 => 7526049
#         pid for 737889 => 5635608
#         pid for 674172 => 8408464
#         pid for 303205 => 5579677
#         pid for 570009 => 2862976
#         pid for 492875 => 7980567
#         pid for 54528 => 4368512
# #> Elapsed time for loading negatives: 218.9728808403015
# #> The # of positives: Min 1, Max 7, Mean 1.06, Median 1.0
# #> The # of negatives: Min 93, Max 100, Mean 99.00, Median 99.0
# #> Save [qid, ppid, npid#1, npid#2, ..., npid#N] to experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/triples.hn1.jsonl (# of samples = 40000000)
#> Elapsed time for saving triples: 162.89451956748962
#> output:               experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train/label.py/triples.hn1.jsonl


# 3-2. Get expansion embeddings & weights for KD
device=0
sh scripts/kd_prf_2nd_iter/msmarco_psg.qe.sh ${device}
# experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg-train-kmeans.prf.docs3.k10.beta0.5.clusters24/label.py/2021-10-31_11.12.15/


# 3-3. Pilot test
sh scripts/kd_prf_2nd_iter/pilot_test/trec2019.sh 2
sh scripts/kd_prf_2nd_iter/pilot_test/trec2019.prf.sh 2 3 10 0.5 24
sh scripts/kd_prf_2nd_iter/pilot_test/trec2020.sh 2
sh scripts/kd_prf_2nd_iter/pilot_test/trec2020.prf.sh 2 3 10 0.5 24


# 4. Fine-tune ColBERT student using newly updated ColBERT-PRF teacher
sh scripts/kd_prf_2nd_iter/msmarco_psg.training.sh

mkdir -p experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py

./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 25000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 50000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 75000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 100000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 150000
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 200000

sh scripts/validation/msmarco_psg.sh 2 experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 25000 > experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/25000.log
sh scripts/validation/msmarco_psg.sh 2 experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 50000 > experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/50000.log
sh scripts/validation/msmarco_psg.sh 0 experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 75000 > experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/75000.log
sh scripts/validation/msmarco_psg.sh 2 experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 100000 > experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/100000.log
sh scripts/validation/msmarco_psg.sh 2 experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 150000 > experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/150000.log
sh scripts/validation/msmarco_psg.sh 2 experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd 200000 > experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/200000.log